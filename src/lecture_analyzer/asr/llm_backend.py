"""用能听音频的大模型转写（如 Gemini）：按固定时长切片，逐片转写后拼回整体时间轴。

适用于任何声明了 audio = true 的模型档案，不绑定某一家。为适应长课：
- 每片结果按型号缓存在工作目录，重跑时跳过已完成的片段（缓存是否作废由调用方决定）；
- 某片被上游内容审核拦截时对半切开重试，最短切到 MIN_PIECE_SECONDS。拦截有几种表现：返回空结果或拦截说明
  （同一片段用同一型号重试仍被拦截），或者输出在中途被截断、带修正提示重试仍不完整（M4 验收实测）；
  最短的片段仍被主模型拦截，再依次换用备用模型（实测换个型号常能转写，但备用模型转写长片段会漏掉大量内容，
  所以只用在最短的片段上）；都失败的区间记入 skipped，由调用方决定是否用本地模型补转。
- 不可用的结果与被拦截同样处理，读缓存时也按这个标准：一半以上的句子是空的或「[听不清]」；有现场翻译的课里
  整段没有一句授课语言。M4 验收：备用型号会把老师的韩语原话写成「[听不清]」或干脆漏掉，只留下翻译的中文
  （病毒学的缓存里主型号 95 片每片都有韩语，备用型号 38 片里 14 片一句韩语都没有）；本地模型至少能转出韩语原文。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, Field

from ..course import CourseSettings
from ..llm import AudioPart, EmptyResponseError, LLMProvider, ModelProfile, StructuredOutputError, TextPart, Usage
from ..media import encode_mp3, halve_wav, split_wav, wav_duration_ms
from ..prompts import load_prompt
from ..timecode import parse_timecode
from .base import ASRBackend, ASRError, Segment

MIN_PIECE_SECONDS = 150  # 被拦截时最短切到 2.5 分钟，再短就放弃这一段
LANGUAGE_CHECK_LINES = 5  # 少于这么多句时只有翻译也正常（例如一节末尾的几句翻译）；漏掉原话的分片都有 8 句以上
_UNCLEAR = re.compile(r"[\s\[\]（）()【】]*([听聽]不清[楚]?[\s\[\]（）()【】]*)*")  # 整句只有「[听不清]」标记


class TranscriptLine(BaseModel):
    start: str = Field(description="开始时间，相对本段音频开头，格式 mm:ss")
    end: str = Field(description="结束时间，相对本段音频开头，格式 mm:ss")
    lang: str = Field(description="这一句的语种代码：zh、ko、en 等")
    text: str


class TranscriptChunk(BaseModel):
    lines: list[TranscriptLine]


class LLMTranscriber(ASRBackend):
    """alternates：最短的片段仍被主模型拦截时依次换用的备用模型（实测拦截与型号有关）。"""

    def __init__(
        self,
        provider: LLMProvider,
        work_dir: Path,
        chunk_minutes: float = 10,
        settings: CourseSettings | None = None,
        alternates: Sequence[LLMProvider] = (),
    ) -> None:
        self.providers = [provider, *alternates]
        for candidate in self.providers:
            if not candidate.profile.audio:
                raise ASRError(f"模型档案「{candidate.profile.name}」不支持音频输入（audio = false）")
        self.provider = provider
        self.work_dir = work_dir
        self.chunk_seconds = chunk_minutes * 60
        self.settings = settings or CourseSettings()
        self.name = f"llm:{provider.profile.name}"
        self.usage_by_model: dict[str, tuple[ModelProfile, Usage]] = {}  # 各型号的用量（含被拦截请求消耗的输入）
        self.skipped: list[tuple[int, int]] = []  # 没能转写的区间（毫秒）
        self._system = ""
        self._instruction = ""

    @property
    def usage(self) -> Usage:
        """各型号用量之和。"""
        total = Usage()
        for _, usage in self.usage_by_model.values():
            total = total + usage
        return total

    def _add_usage(self, provider: LLMProvider, usage: Usage) -> None:
        profile, previous = self.usage_by_model.get(provider.profile.model, (provider.profile, Usage()))
        self.usage_by_model[provider.profile.model] = (profile, previous + usage)

    def transcribe(self, wav: Path, hotwords: Sequence[str] = ()) -> list[Segment]:
        self.skipped = []
        self._system = load_prompt(
            "transcribe", subject=self.settings.subject, teaching=self.settings.teaching_description
        )
        self._instruction = "请转写这段录音。"
        if hotwords:
            self._instruction += "\n可能出现的专业术语：" + "、".join(hotwords)
        chunk_dir = self.work_dir / f"{wav.stem}_chunks_{int(self.chunk_seconds)}s"
        segments: list[Segment] = []
        pieces = split_wav(wav, self.chunk_seconds, chunk_dir)
        for index, (piece, offset_ms) in enumerate(pieces, start=1):
            self.notify(f"第 {index}/{len(pieces)} 片")
            segments.extend(self._transcribe_piece(piece, offset_ms))
        return segments

    def _transcribe_piece(self, piece: Path, offset_ms: int) -> list[Segment]:
        lines = self._load_cache(piece)
        if lines is None:
            # 先只用主模型；切到最短仍被拦截才换备用模型（实测备用模型转写 10 分钟的长片段会漏掉大量内容）
            shortest = wav_duration_ms(piece) <= MIN_PIECE_SECONDS * 1000
            lines = self._request_any(piece, self.providers if shortest else self.providers[:1])
        if lines is None:  # 被拦截：对半切开重试，已是最短就记为没能转写
            return self._split_and_retry(piece, offset_ms)
        return _to_segments(lines, offset_ms)

    def _request_any(self, piece: Path, providers: Sequence[LLMProvider]) -> list[TranscriptLine] | None:
        """依次用这些模型转写；成功就按该型号缓存，都被拦截时返回 None。"""
        for provider in providers:
            try:
                lines = self._request(provider, piece)
            except (EmptyResponseError, StructuredOutputError) as exc:  # 被拦截，或输出两次都在中途被截断
                self._add_usage(provider, exc.usage)
                continue
            if unusable(lines, self._source_languages):
                continue
            self._save_cache(provider, piece, lines)
            return lines
        return None

    def _split_and_retry(self, piece: Path, offset_ms: int) -> list[Segment]:
        duration_ms = wav_duration_ms(piece)
        if duration_ms <= MIN_PIECE_SECONDS * 1000:
            self.skipped.append((offset_ms, offset_ms + duration_ms))
            return []
        halves = halve_wav(piece, piece.parent / f"{piece.stem}_halves")
        return [seg for half, start in halves for seg in self._transcribe_piece(half, offset_ms + start)]

    def _request(self, provider: LLMProvider, piece: Path) -> list[TranscriptLine]:
        mp3 = piece.with_suffix(".mp3")
        if not mp3.is_file():
            encode_mp3(piece, mp3)
        result = provider.generate_structured(
            [AudioPart(mp3.read_bytes(), "audio/mp3"), TextPart(self._instruction)], TranscriptChunk,
            system=self._system,
        )
        self._add_usage(provider, result.usage)
        return result.output.lines

    @property
    def _source_languages(self) -> list[str]:
        """有现场翻译时的授课语言：结果里一句都没有，说明老师的原话被漏掉了。"""
        return self.settings.languages if self.settings.interpreted else []

    @staticmethod
    def _cache_path(provider: LLMProvider, piece: Path) -> Path:
        model = re.sub(r"[^\w.-]", "_", provider.profile.model)
        return piece.with_name(f"{piece.stem}.{model}.json")

    def _load_cache(self, piece: Path) -> list[TranscriptLine] | None:
        """按主模型、备用模型的顺序找已有的、可用的结果；损坏的缓存（例如写到一半时任务被取消）当作没有。"""
        for provider in self.providers:
            path = self._cache_path(provider, piece)
            if not path.is_file():
                continue
            try:
                lines = TranscriptChunk.model_validate_json(path.read_text(encoding="utf-8")).lines
            except ValueError:
                continue
            if not unusable(lines, self._source_languages):
                return lines
        return None

    def _save_cache(self, provider: LLMProvider, piece: Path, lines: Sequence[TranscriptLine]) -> None:
        payload = TranscriptChunk(lines=list(lines)).model_dump_json()
        self._cache_path(provider, piece).write_text(payload, encoding="utf-8")


def unusable(lines: Sequence[TranscriptLine], source_languages: Sequence[str] = ()) -> bool:
    """结果不可用：一半以上的句子是空的或只有「[听不清]」；或者给了授课语言（有现场翻译的课），清楚的句子不少，
    却没有一句授课语言。多半是上游过滤了老师的原话。没有句子（整段静音）不算。"""
    clear = [line for line in lines if not _UNCLEAR.fullmatch(line.text)]
    if len(clear) * 2 < len(lines):
        return True
    if not source_languages or len(clear) < LANGUAGE_CHECK_LINES:
        return False
    sources = {_base_language(code) for code in source_languages}
    return not any(_base_language(line.lang) in sources for line in clear)


def _base_language(code: str) -> str:
    return code.strip().lower().split("-")[0]


def _to_segments(lines: Sequence[TranscriptLine], offset_ms: int) -> list[Segment]:
    """换算成整段音频的时间；时间缺失或倒退时取不早于上一句的时间，保证时间轴单调。"""
    segments: list[Segment] = []
    last_end = offset_ms
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        start = _time_not_before(line.start, offset_ms, last_end)
        end = _time_not_before(line.end, offset_ms, start)
        segments.append(Segment(start, end, text, line.lang.strip().lower()))
        last_end = end
    return segments


def _time_not_before(value: str, offset_ms: int, floor_ms: int) -> int:
    try:
        return max(offset_ms + parse_timecode(value), floor_ms)
    except ValueError:
        return floor_ms
