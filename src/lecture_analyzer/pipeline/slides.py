"""匹配课件：找出一讲讲到了课程资料里的哪些页（课件、教材等按页的资料）。

一次模型调用（[steps] match，没写时与页面识别同一档案）：输入是全部页面的目录（[F文件序号-页码] 页面标题）
和这一讲转写的抽样，输出讲到的页面区间。代码校验页码；同一份文件里两段之间相隔不超过 GAP_FILL 页时补上中间的页，
整理笔记时模型能据此判断老师是否跳过了这些页。页面目录超出预算时，先按术语命中数保留最相关的文件。
资料和授课语言不同也能匹配（按含义），所以不靠关键词比对。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from ..asr.base import Segment
from ..course import CourseSettings
from ..library import Course, Library, PageRef
from ..llm import LLMProvider, TextPart, Usage
from ..llm.base import Effort
from ..materials import PAGE_KINDS, Material, MaterialStore
from ..prompts import load_prompt
from ..schemas import PageDigest, PageRange, SlideMatch
from ..textnorm import normalize_zh
from ..timecode import format_ms
from .memory import estimate_tokens, input_budget

GAP_FILL = 3  # 两段之间相隔不超过这么多页时补上中间的页
SAMPLE_CHARS = 12_000  # 转写抽样的字数上限
TOC_BUDGET = 80_000  # 页面目录的 token 上限（还受模型上下文约束）
TITLE_CHARS = 40  # 目录里每页标题的字数上限
HOTWORDS = 150  # 转写热词取整门课课件术语的前多少个


@dataclass(frozen=True)
class MatchFile:
    number: int  # 目录里的文件序号 F1、F2…
    material: Material
    pages: list[PageDigest]


@dataclass
class MatchResult:
    refs: list[PageRef]
    note: str
    usage: Usage
    dropped_files: int = 0  # 超出预算、按术语筛掉的文件数


def match_files(store: MaterialStore, course: Course) -> list[MatchFile]:
    """参与匹配的资料：按页的资料里已经识别过的页。"""
    files = []
    for material in store.list(course, PAGE_KINDS):
        pages = store.pages(material.id)
        if pages:
            files.append(MatchFile(len(files) + 1, material, pages))
    return files


def render_toc(files: Sequence[MatchFile]) -> str:
    blocks = []
    for item in files:
        material = item.material
        lines = [f"## [F{item.number}] {material.title}（{material.kind_label}，共 {material.page_count} 页）"]
        lines += [f"[F{item.number}-{page.page_no}] {_page_title(page)}" for page in item.pages]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _page_title(page: PageDigest) -> str:
    title = " ".join((page.title or page.content_md).split())
    return title[:TITLE_CHARS] or "（无标题）"


def sample_transcript(segments: Sequence[Segment], limit: int = SAMPLE_CHARS) -> str:
    """按时间顺序均匀抽样，总字数不超过 limit。"""
    total = sum(len(seg.text) + 10 for seg in segments)
    step = max(1, math.ceil(total / limit))
    return "\n".join(f"[{format_ms(seg.start_ms)}] {seg.text}" for seg in segments[::step])


def relevant_files(files: Sequence[MatchFile], segments: Sequence[Segment], keep: int) -> list[MatchFile]:
    """目录太长时只保留术语命中最多的 keep 份文件（按原顺序）。"""
    spoken = normalize_zh("".join(seg.text for seg in segments))
    scores = []
    for item in files:
        terms = {normalize_zh(term) for page in item.pages for term in page.terms}
        scores.append(sum(1 for term in terms if len(term) >= 2 and term in spoken))
    ranked = sorted(range(len(files)), key=lambda i: (-scores[i], i))[:keep]
    return [files[i] for i in sorted(ranked)]


def build_input(
    files: Sequence[MatchFile], segments: Sequence[Segment], provider: LLMProvider
) -> tuple[list[MatchFile], str]:
    """在预算内组装目录；超出时逐份去掉最不相关的文件。"""
    budget = input_budget(provider.profile, TOC_BUDGET)
    chosen = list(files)
    while len(chosen) > 1 and estimate_tokens(render_toc(chosen), provider.profile) > budget:
        chosen = relevant_files(chosen, segments, len(chosen) - 1)
    return chosen, render_toc(chosen)


def match_lecture(
    provider: LLMProvider,
    files: Sequence[MatchFile],
    segments: Sequence[Segment],
    settings: CourseSettings,
    effort: Effort | None = None,
) -> MatchResult:
    chosen, toc = build_input(files, segments, provider)
    system = load_prompt("match", subject=settings.subject, teaching=settings.teaching_description)
    parts = [
        TextPart("## 资料目录\n" + toc),
        TextPart("## 这一讲的转写抽样\n" + sample_transcript(segments)),
        TextPart("请输出这一讲讲到的页面区间。"),
    ]
    result = provider.generate_structured(parts, SlideMatch, system=system, effort=effort)
    refs = resolve_ranges(result.output.ranges, chosen)
    return MatchResult(refs, result.output.note.strip(), result.usage, dropped_files=len(files) - len(chosen))


def resolve_ranges(ranges: Sequence[PageRange], files: Sequence[MatchFile]) -> list[PageRef]:
    """校验模型给的区间：文件序号和页码必须存在，起止颠倒时对调；再补上相隔不远的中间页。"""
    by_number = {item.number: item.material for item in files}
    pages: dict[int, set[int]] = {}
    for item in ranges:
        material = by_number.get(item.file)
        if material is None:
            continue
        start, end = sorted((item.start, item.end))
        start, end = max(start, 1), min(end, material.page_count)
        if start <= end:
            pages.setdefault(material.id, set()).update(range(start, end + 1))
    return [PageRef(material_id, page_no)
            for material_id in sorted(pages) for page_no in fill_gaps(sorted(pages[material_id]))]


def fill_gaps(page_nos: Sequence[int], gap: int = GAP_FILL) -> list[int]:
    filled: list[int] = []
    for page_no in page_nos:
        if filled and 1 < page_no - filled[-1] <= gap + 1:
            filled.extend(range(filled[-1] + 1, page_no))
        filled.append(page_no)
    return filled


def parse_page_spec(spec: str, files: Sequence[Material]) -> list[PageRef]:
    """手动指定的页："1:3-30,2:1-5" 表示资料 1 的第 3–30 页和资料 2 的第 1–5 页；
    资料用 `lec material list` 里的编号，只写编号表示整份资料。"""
    by_id = {material.id: material for material in files}
    refs: list[PageRef] = []
    for chunk in (part.strip() for part in spec.replace("，", ",").split(",")):
        if not chunk:
            continue
        head, _, span = chunk.partition(":")
        material = by_id.get(int(head)) if head.strip().isdigit() else None
        if material is None:
            raise ValueError(f"「{chunk}」里的资料编号不存在（编号见课程页的资料表，或 `lec material list`）")
        refs += [PageRef(material.id, n) for n in _parse_span(span, material, chunk)]
    return sorted(set(refs))


def _parse_span(span: str, material: Material, chunk: str) -> range:
    if not span.strip():
        return range(1, material.page_count + 1)
    first, _, last = span.partition("-")
    try:
        start, end = int(first), int(last or first)
    except ValueError:
        raise ValueError(f"「{chunk}」的页码格式不对，应写成 资料编号:起始页-结束页") from None
    if not 1 <= start <= end <= material.page_count:
        raise ValueError(f"「{chunk}」超出范围：「{material.title}」共 {material.page_count} 页")
    return range(start, end + 1)


def describe_pages(refs: Sequence[PageRef], titles: dict[int, str]) -> str:
    """「ch1-介绍」第 1–21 页；「ch2」第 3–5、8 页。"""
    grouped: dict[int, list[int]] = {}
    for ref in refs:
        grouped.setdefault(ref.material_id, []).append(ref.page_no)
    parts = [f"「{titles.get(mid, f'资料 {mid}')}」第 {page_spans(nos)} 页" for mid, nos in grouped.items()]
    return "；".join(parts) if parts else "没有对应的页"


def page_spans(page_nos: Sequence[int]) -> str:
    """[1, 2, 3, 5, 8, 9] → "1–3、5、8–9"。"""
    spans: list[list[int]] = []
    for number in sorted(set(page_nos)):
        if spans and number == spans[-1][1] + 1:
            spans[-1][1] = number
        else:
            spans.append([number, number])
    return "、".join(f"{a}–{b}" if b > a else str(a) for a, b in spans)


def course_terms(library: Library, course: Course, limit: int = HOTWORDS) -> list[str]:
    """整门课课件里的术语，按出现的页数从多到少（同样多时按首次出现顺序），用作转写热词。

    顺序必须与进程无关：转写的分片缓存在热词变了时作废（见 run.py 的 _refresh_part_cache）。
    """
    store = MaterialStore(library)
    counts: Counter[str] = Counter()
    first: dict[str, int] = {}
    for material in store.list(course, PAGE_KINDS):
        for page in store.pages(material.id):
            for term in dict.fromkeys(term.strip() for term in page.terms if term.strip()):  # 按页内顺序去重，不用集合
                counts[term] += 1
                first.setdefault(term, len(first))
    return sorted(counts, key=lambda term: (-counts[term], first[term]))[:limit]
