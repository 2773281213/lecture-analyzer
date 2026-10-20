"""第一步：单讲抽取。只看本讲的转写和课件，输出结构化笔记与考点信号。

一讲由多节合并而成时，句子编号全讲连续，每节开头插一行分隔；整讲放不进所选模型时，
按 memory.plan_windows 分段整理，再由代码合并（见 memory.merge_notes）。
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Sequence

from ..asr.base import Segment
from ..course import CourseSettings, language_name
from ..llm import LLMProvider, LLMResult, TextPart, Usage
from ..llm.base import Effort
from ..prompts import load_prompt
from ..schemas import LectureNotes, PageDigest
from ..textnorm import strip_foreign_asides
from ..timecode import format_ms
from .memory import Window, merge_notes, plan_windows


SLIDES_CHARS = 60_000  # 课件内容的总字数上限，超出时每页按比例截短
MIN_PAGE_CHARS = 300


@dataclass
class ExtractResult(LLMResult[LectureNotes]):
    windows: int = 1  # 分几段整理


def transcript_lines(segments: Sequence[Segment]) -> list[str]:
    """转写按句编号（从 1 开始，整讲连续），模型用这个编号标注出处；有语种时附上语种标记。"""
    return [
        f"[S{i}][{format_ms(seg.start_ms)}]{f'[{seg.lang}]' if seg.lang else ''} {seg.text}"
        for i, seg in enumerate(segments, start=1)
    ]


def part_starts(segments: Sequence[Segment], offsets: Sequence[int]) -> list[int]:
    """各节第一句的下标（从 0 开始）：按各节在整讲时间轴上的起点查找。"""
    starts = [seg.start_ms for seg in segments]
    return [bisect_left(starts, offset) for offset in offsets]


def render_transcript(segments: Sequence[Segment], starts: Sequence[int] = (), window: Window | None = None) -> str:
    """渲染转写；多节的讲在每节开头插一行「—— 第 K 节 ——」。window 指定只渲染其中一段。"""
    lines = transcript_lines(segments)
    window = window or Window(0, len(lines))
    markers = {start: k for k, start in enumerate(starts, start=1)} if len(starts) > 1 else {}
    rendered: list[str] = []
    for i in range(window.start, window.end):
        if i in markers:
            rendered.append(f"—— 第 {markers[i]} 节 ——")
        rendered.append(lines[i])
    return "\n".join(rendered)


def render_slides(digests: Sequence[PageDigest], sources: Sequence[str] = (), limit: int = SLIDES_CHARS) -> str:
    """课件页，每页以 [P页码] 开头，sources 给出时在括号里注明出处；总字数超过 limit 时每页按比例截短。"""
    total = sum(len(d.content_md) for d in digests)
    share = max(MIN_PAGE_CHARS, limit // max(len(digests), 1)) if total > limit else None
    blocks = []
    for index, digest in enumerate(digests):
        source = f"（{sources[index]}）" if index < len(sources) and sources[index] else ""
        content = digest.content_md
        if share is not None and len(content) > share:
            content = content[:share] + "…"
        blocks.append(f"[P{digest.page_no}]{source} {digest.title}\n{content}")
    return "\n\n".join(blocks)


# 以课件为参照的信号：没有课件的讲不产生（M4 验收：不提供课件的课也标出了「课件外补充」）
SLIDE_SIGNALS = ("off_slide", "skipped_slide")
NO_SLIDES = "（本讲没有课件：不要输出 off_slide、skipped_slide 信号，slide_pages 都填空列表）"


def instructions_block(instructions: str) -> str:
    """用户对本课程的补充要求（课程提示词），附加在各步骤的提示词里。"""
    if not instructions.strip():
        return ""
    return (
        "\n用户对本课程的补充要求如下（请遵守；与上面的格式要求冲突时以格式要求为准，例如 quote 仍须逐字照抄）：\n"
        f"{instructions.strip()}\n"
    )


def interpretation_block(settings: CourseSettings) -> str:
    """有现场翻译时，说明哪些句子是老师原话、哪些是翻译。"""
    if not settings.interpreted:
        return ""
    return load_prompt(
        "extract_interpretation",
        source="、".join(language_name(code) for code in settings.languages),
        source_tags="、".join(f"[{code}]" for code in settings.languages),
        target=language_name(settings.interpreted_to),
        target_tag=settings.interpreted_to,
    ) + "\n"


def build_system_prompt(settings: CourseSettings, instructions: str = "") -> str:
    interpretation = interpretation_block(settings)
    return load_prompt(
        "extract",
        subject=settings.subject,
        teaching=settings.teaching_description,
        note_language=settings.note_language_name,
        term_rule=settings.term_rule,
        interpretation=interpretation,
        instructions=instructions_block(instructions),
    )


def extract_lecture(
    provider: LLMProvider,
    segments: Sequence[Segment],
    digests: Sequence[PageDigest],
    settings: CourseSettings,
    *,
    effort: Effort | None = None,
    starts: Sequence[int] = (),
    instructions: str = "",
    sources: Sequence[str] = (),
) -> ExtractResult:
    """整理一讲的笔记。starts 为各节第一句的下标，用于插入节的分隔和按节分段；sources 为各页课件的出处。"""
    system = build_system_prompt(settings, instructions)
    slides = "## 课件\n" + (render_slides(digests, sources) or NO_SLIDES)
    lines = transcript_lines(segments)
    windows = plan_windows([len(line) + 1 for line in lines], starts, len(system) + len(slides), provider.profile)
    notes: list[LectureNotes] = []
    usage = Usage()
    for number, window in enumerate(windows, start=1):
        ask = f"请按要求输出本讲的结构化笔记，笔记正文用{settings.note_language_name}。"
        if len(windows) > 1:
            ask = (f"本讲较长，分 {len(windows)} 段整理，这是第 {number} 段（S{window.start + 1}–S{window.end}），"
                   f"只整理这一段的内容。{ask}")
        parts = [TextPart(slides), TextPart("## 课堂转写\n" + render_transcript(segments, starts, window)), TextPart(ask)]
        result = provider.generate_structured(parts, LectureNotes, system=system, effort=effort)
        notes.append(result.output)
        usage = usage + result.usage
    merged = chinese_only(merge_notes(notes))
    if not digests:
        merged.signals = [signal for signal in merged.signals if signal.type not in SLIDE_SIGNALS]
    return ExtractResult(merged, usage, provider.profile.model, windows=len(windows))


def chinese_only(notes: LectureNotes) -> LectureNotes:
    """去掉笔记里括号中的外语旁注（分析结果全中文）；原话 quote 是证据，保持原文不动。"""
    clean = strip_foreign_asides
    return notes.model_copy(update={
        "title": clean(notes.title),
        "outline": [clean(item) for item in notes.outline],
        "knowledge_points": [kp.model_copy(update={"name": clean(kp.name), "summary_md": clean(kp.summary_md)})
                             for kp in notes.knowledge_points],
        "examples": [ex.model_copy(update={"title": clean(ex.title), "problem_md": clean(ex.problem_md),
                                           "solution_md": clean(ex.solution_md)}) for ex in notes.examples],
        "homework": [clean(item) for item in notes.homework],
        "signals": [s.model_copy(update={"knowledge_point": clean(s.knowledge_point), "meaning": clean(s.meaning)})
                    for s in notes.signals],
    })
