"""按模型的上下文预算组装输入：整理笔记的分段与合并，以及跨讲整合、总复习输入的按序裁剪。

token 按"字数 × 档案的 tokens_per_char"估算（M0 实测的换算比例）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ..llm import ModelProfile
from ..schemas import KnowledgePointNote, LectureNotes
from ..textnorm import normalize_zh

OUTPUT_RATIO = 0.35  # 笔记输出约为转写输入的 0.3 倍（药理学、病毒学实测），留些余量
OUTPUT_BASE = 1500  # 笔记里与长度无关的部分（标题、大纲等）
SAFETY_TOKENS = 4000  # 估算误差与格式说明的余量


class BudgetError(RuntimeError):
    """输入放不进所选模型的上下文。"""


def estimate_tokens(text: str, profile: ModelProfile) -> int:
    return math.ceil(len(text) * profile.tokens_per_char)


def input_budget(profile: ModelProfile, cap: int | None = None) -> int:
    """可用于输入的 token：上下文上限减去输出上限和余量；cap 为配置里的预算上限。"""
    budget = profile.max_context - profile.max_output - SAFETY_TOKENS
    return min(budget, cap) if cap else budget


# ---------- 整理笔记的分段 ----------


@dataclass(frozen=True)
class Window:
    start: int  # 起始句子下标（从 0 开始，含）
    end: int  # 结束下标（不含）


def plan_windows(
    line_lengths: Sequence[int], part_starts: Sequence[int], fixed_chars: int, profile: ModelProfile
) -> list[Window]:
    """把一讲的转写分成能放进模型的若干段。

    line_lengths：每句渲染后的字数；part_starts：各节第一句的下标（第 1 节为 0）；
    fixed_chars：每段都要带的固定内容（系统提示词、课件）的字数。
    整讲放得下就一段；否则按节切，单节仍放不下再对半切，最后把相邻的段尽量合并，减少调用次数。
    """
    prefix = [0]
    for length in line_lengths:
        prefix.append(prefix[-1] + length)
    fixed = math.ceil(fixed_chars * profile.tokens_per_char)

    def fits(window: Window) -> bool:
        transcript = math.ceil((prefix[window.end] - prefix[window.start]) * profile.tokens_per_char)
        return (transcript + fixed <= input_budget(profile)
                and transcript * OUTPUT_RATIO + OUTPUT_BASE <= profile.max_output)

    whole = Window(0, len(line_lengths))
    if fits(whole):
        return [whole]
    bounds = sorted({0, *(s for s in part_starts if 0 < s < len(line_lengths)), len(line_lengths)})
    pieces: list[Window] = []
    for start, end in zip(bounds, bounds[1:]):
        pieces.extend(_split_until_fits(Window(start, end), fits))
    return _merge_adjacent(pieces, fits)


def _split_until_fits(window: Window, fits) -> list[Window]:
    if fits(window):
        return [window]
    if window.end - window.start <= 1:
        raise BudgetError("课件或单句转写过长，放不进所选模型的上下文，请换用上下文更大的模型")
    middle = (window.start + window.end) // 2
    return _split_until_fits(Window(window.start, middle), fits) + _split_until_fits(Window(middle, window.end), fits)


def _merge_adjacent(windows: list[Window], fits) -> list[Window]:
    merged = [windows[0]]
    for window in windows[1:]:
        candidate = Window(merged[-1].start, window.end)
        if fits(candidate):
            merged[-1] = candidate
        else:
            merged.append(window)
    return merged


def merge_notes(parts: Sequence[LectureNotes]) -> LectureNotes:
    """把分段整理的笔记合并成一讲：各项按顺序拼接，同名知识点合并要点、出处和时长。"""
    if len(parts) == 1:
        return parts[0]
    titles = list(dict.fromkeys(p.title.strip() for p in parts if p.title.strip()))
    return LectureNotes(
        title="；".join(titles[:3]),
        outline=[item for p in parts for item in p.outline],
        knowledge_points=_merge_points([kp for p in parts for kp in p.knowledge_points]),
        examples=[ex for p in parts for ex in p.examples],
        homework=list(dict.fromkeys(item for p in parts for item in p.homework)),
        signals=[s for p in parts for s in p.signals],
    )


def _merge_points(points: Sequence[KnowledgePointNote]) -> list[KnowledgePointNote]:
    """名称相同（归一化后）的知识点合并，保持首次出现的位置。"""
    merged: dict[str, KnowledgePointNote] = {}
    for kp in points:
        key = normalize_zh(kp.name) or kp.name
        old = merged.get(key)
        merged[key] = kp if old is None else KnowledgePointNote(
            name=old.name,
            summary_md=f"{old.summary_md}\n\n{kp.summary_md}",
            slide_pages=sorted({*old.slide_pages, *kp.slide_pages}),
            segment_ids=sorted({*old.segment_ids, *kp.segment_ids}),
            minutes=old.minutes + kp.minutes,
            question_types=list(dict.fromkeys([*old.question_types, *kp.question_types])),
        )
    return list(merged.values())


# ---------- 跨讲整合、总复习：按优先级裁剪 ----------


@dataclass
class Item:
    full: str
    short: str = ""  # 裁剪后剩下的文字；空串表示整条去掉
    keep: float = 0  # 同一部分里数值小的先裁剪


@dataclass
class Section:
    heading: str
    items: list[Item] = field(default_factory=list)
    trim_rank: int | None = None  # 不同部分的裁剪顺序：数字小的先裁；None 表示不能裁剪
    empty_text: str = "（无）"


@dataclass
class Assembled:
    text: str
    tokens: int
    trimmed: dict[str, int]  # 部分标题 → 被裁剪的条数


def assemble(sections: Sequence[Section], budget: int, profile: ModelProfile, fixed_chars: int = 0) -> Assembled:
    """按预算拼出输入文本。超出预算时按 (trim_rank, keep) 从小到大逐条裁剪；全部裁完仍超出就报错。"""
    fixed = math.ceil(fixed_chars * profile.tokens_per_char)
    current = [[item.full for item in section.items] for section in sections]
    total = fixed + sum(estimate_tokens(_render(s, texts), profile) for s, texts in zip(sections, current))
    order = sorted(
        (
            (section.trim_rank, item.keep, si, ii)
            for si, section in enumerate(sections) if section.trim_rank is not None
            for ii, item in enumerate(section.items) if item.short != item.full
        ),
    )
    trimmed: dict[str, int] = {}
    for _, _, si, ii in order:
        if total <= budget:
            break
        item = sections[si].items[ii]
        total -= estimate_tokens(item.full, profile) - estimate_tokens(item.short, profile)
        current[si][ii] = item.short
        trimmed[sections[si].heading] = trimmed.get(sections[si].heading, 0) + 1
    text = "\n\n".join(_render(s, texts) for s, texts in zip(sections, current))
    tokens = fixed + estimate_tokens(text, profile)
    if tokens > budget:
        raise BudgetError(
            f"输入约 {tokens} token，超出模型档案「{profile.name}」的预算 {budget} token，请换用上下文更大的模型"
        )
    return Assembled(text, tokens, trimmed)


def _render(section: Section, texts: Sequence[str]) -> str:
    lines = [text for text in texts if text]
    return f"## {section.heading}\n" + ("\n".join(lines) if lines else section.empty_text)


def trimmed_note(trimmed: dict[str, int]) -> str:
    """处理记录里的裁剪说明。"""
    if not trimmed:
        return ""
    return "；输入超出预算，已精简：" + "、".join(f"{name} {count} 条" for name, count in trimmed.items())


def snippet(text: str, limit: int) -> str:
    """压成一行并截断，用于知识点要点的简短说明。"""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
