"""本地语音识别：FunASR（需先运行 `uv sync --extra asr`，默认使用显卡）。"""

from __future__ import annotations

import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..media import SAMPLE_RATE
from .base import ASRBackend, ASRError, Segment, free_gpu_memory

MAX_SEGMENT_MS = 30_000  # VAD 单段最长 30 秒
VAD_MODEL = "fsmn-vad"  # 语音活动检测模型（FunASR 简称），Qwen3-ASR 切句也用它


@dataclass(frozen=True)
class Preset:
    model: str
    punc_model: str | None
    hotwords: bool  # 是否支持热词
    languages: tuple[str, ...] = ()  # 可以强制的语种；空表示不接受 language 参数（只认一种语言）
    generate_kwargs: dict[str, Any] = field(default_factory=dict)


PRESETS: dict[str, Preset] = {
    # SeACo-Paraformer（ModelScope 别名 paraformer-zh）：只认中文，支持热词，带标点和句级时间戳
    "paraformer": Preset(model="paraformer-zh", punc_model="ct-punc", hotwords=True),
    # SenseVoice-Small：速度快，支持中英日韩粤，自带标点（use_itn），按 VAD 切句
    "sensevoice": Preset(
        model="iic/SenseVoiceSmall", punc_model=None, hotwords=False, languages=("zh", "en", "yue", "ja", "ko"),
        generate_kwargs={"use_itn": True},
    ),
}

# SeACo 的热词按空格切分，且只认中文和数字（英文会变成 <unk>）；数量过多会干扰识别
_HOTWORD_PATTERN = re.compile(r"^[一-龥0-9]{2,12}$")
MAX_HOTWORDS = 100


def usable_hotwords(terms: Sequence[str]) -> list[str]:
    picked: list[str] = []
    for term in terms:
        term = term.strip()
        if _HOTWORD_PATTERN.match(term) and term not in picked:
            picked.append(term)
        if len(picked) >= MAX_HOTWORDS:
            break
    return picked


class FunASRBackend(ASRBackend):
    def __init__(self, preset: str = "paraformer", device: str = "cuda:0", language: str = "auto") -> None:
        if preset not in PRESETS:
            raise ASRError(f"未知的 FunASR 预设：{preset}（可选：{', '.join(PRESETS)}）")
        self.preset = PRESETS[preset]
        self.name = f"funasr:{preset}"
        self.device = device
        # 只对多语种预设生效；auto 表示自动识别，预设不支持的语言也改为自动识别
        self.language = language if language in self.preset.languages else "auto"
        self._model: Any = None  # 首次转写时才加载模型（加载耗时且占显存）

    def transcribe(self, wav: Path, hotwords: Sequence[str] = ()) -> list[Segment]:
        cfg: dict[str, Any] = {
            "batch_size_s": 60,
            "batch_size_threshold_s": 30,
            "sentence_timestamp": True,
            **self.preset.generate_kwargs,
        }
        if self.preset.languages:
            cfg["language"] = self.language
        if self.preset.punc_model:
            cfg["return_raw_text"] = True  # 自己切句时需要未加标点的逐字结果
        words = usable_hotwords(hotwords) if self.preset.hotwords else []
        if words:
            cfg["hotword"] = " ".join(words)
        results = self._load_model().generate(input=read_wav_float32(wav), **cfg)
        return _collect_segments(results)

    def warm_up(self) -> None:
        self._load_model()

    def close(self) -> None:
        self._model = None
        free_gpu_memory()

    def _load_model(self) -> Any:
        if self._model is None:
            kwargs: dict[str, Any] = {
                "model": self.preset.model,
                "vad_model": VAD_MODEL,
                "vad_kwargs": {"max_single_segment_time": MAX_SEGMENT_MS},
                "device": self.device,
                "disable_update": True,
            }
            if self.preset.punc_model:
                kwargs["punc_model"] = self.preset.punc_model
            self._model = load_automodel(**kwargs)
        return self._model


def load_automodel(**kwargs: Any) -> Any:
    """创建 FunASR AutoModel；没装本地依赖时给出安装提示。"""
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise ASRError("没有安装本地语音识别依赖，请先运行：uv sync --extra asr") from exc
    return AutoModel(**kwargs)


def read_wav_float32(path: Path) -> np.ndarray:
    """读取 16kHz 单声道 16 bit wav 为 float32 数组。

    直接传数组给 FunASR，避开 torchaudio 的文件加载（新版 torchaudio 需要额外安装 torchcodec）。
    """
    with wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getnchannels() != 1 or audio.getframerate() != SAMPLE_RATE:
            raise ASRError(f"需要 16kHz 单声道 16 bit wav，请先用 media.extract_audio 转换：{path}")
        frames = audio.readframes(audio.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _collect_segments(results: Sequence[dict[str, Any]]) -> list[Segment]:
    segments: list[Segment] = []
    for result in results:
        rebuilt = rebuild_sentences(
            str(result.get("text", "")), str(result.get("raw_text", "")).split(), result.get("timestamp") or []
        )
        if rebuilt is not None:
            segments.extend(rebuilt)
            continue
        for sentence in result.get("sentence_info", []):  # 对不上时退回 FunASR 自己的切句
            text = str(sentence.get("text", "")).strip()
            if text:
                segments.append(Segment(int(sentence["start"]), int(sentence["end"]), text))
    return segments


_SENTENCE_END = set("。？！?!")
_CLAUSE_END = set("，,；;")
MAX_SENTENCE_CHARS = 50  # 口语常有很长的一逗到底，超过这个长度时在逗号处断开


Timestamps = Sequence[Sequence[int]]
_Piece = tuple[str, int | None]  # （文本片段，对应的词序号；标点和空白为 None）


def rebuild_sentences(text: str, raw_tokens: Sequence[str], timestamps: Timestamps) -> list[Segment] | None:
    """用"带标点的整段文本 + 逐词时间戳"重新切句。

    FunASR 1.4.16 的 sentence_info 会把标点错位 1–2 个字（如"所发，生的效应"），
    而整段文本的标点和逐词时间戳本身是对的（已实测）。按顺序把 raw_text 的词对回带标点的文本，
    在句末标点处断句；任何一处对不上就返回 None，由调用方退回原结果。
    """
    if not raw_tokens or len(raw_tokens) != len(timestamps):
        return None
    pieces = _align_tokens(text, raw_tokens)
    return None if pieces is None else _group_sentences(pieces, timestamps)


def _align_tokens(text: str, raw_tokens: Sequence[str]) -> list[_Piece] | None:
    """把带标点的文本切成片段：识别出的词（附词序号）或标点、空白。"""
    pieces: list[_Piece] = []
    lowered = text.lower()
    position = token_index = 0
    while position < len(text):
        token = raw_tokens[token_index].lower() if token_index < len(raw_tokens) else ""
        if token and lowered.startswith(token, position):
            pieces.append((text[position : position + len(token)], token_index))
            position += len(token)
            token_index += 1
            continue
        char = text[position]
        if char.isalnum():
            return None  # 出现了对不上任何词的字，放弃重建
        pieces.append((char, None))
        position += 1
    return pieces if token_index == len(raw_tokens) else None


def _group_sentences(pieces: Sequence[_Piece], timestamps: Timestamps) -> list[Segment]:
    sentences: list[Segment] = []
    buffer, first, last = "", -1, -1
    for piece, index in pieces:
        if index is not None:
            first = index if first < 0 else first
            last = index
            buffer += piece
            continue
        if not piece.isspace() or (buffer and buffer[-1].isascii() and buffer[-1].isalnum()):
            buffer += piece  # 空白只保留英文词之间的
        if first >= 0 and _ends_sentence(piece, buffer):
            sentences.append(Segment(int(timestamps[first][0]), int(timestamps[last][1]), buffer.strip()))
            buffer, first = "", -1
    if first >= 0 and buffer.strip():
        sentences.append(Segment(int(timestamps[first][0]), int(timestamps[last][1]), buffer.strip()))
    return sentences


def _ends_sentence(char: str, buffer: str) -> bool:
    return char in _SENTENCE_END or (char in _CLAUSE_END and len(buffer) >= MAX_SENTENCE_CHARS)
