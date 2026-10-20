"""总复习：按章节调用模型，把各讲对同一组知识点的讲解合并成复习讲义。

- 模型只写复习要点，用编号引用依据（讲解 L讲次N序号、原话 Q、例题 E）；原话和出处由代码按编号从书库填入，
  只来自通过校验的考点信号；正文里的编号旁注和开头重复知识点名称的标题保存前去掉（页面不显示编号）；
- 重要程度由代码按考点信号和课程资料（往年真题、模拟题、复习提纲、考试通知）计算（"模型抽信号、代码打分"），
  不让模型打分；
- 每章按未裁剪的材料算一个指纹，材料没变的章节不重新生成，变了的标为待更新。
- 考前报告的知识点要点也取自这里（update_review_for：只更新排进报告的知识点所在的章节）。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from pydantic import BaseModel

from ..config import AppConfig
from ..knowledge import CourseMemory, HistorySignal, KnowledgePoint, LectureCache, StoredReview
from ..library import Course, Library
from ..llm import LLMProvider, TextPart, create_provider
from ..llm.base import Effort
from ..prompts import load_prompt
from ..render import ROLE_LABELS, SIGNAL_LABELS, group_by_chapter, quote_html, zh_aliases
from ..schemas import LectureNotes, ReviewChapter, WorkedExample
from ..textnorm import normalize_zh, strip_citations, strip_foreign_asides
from ..timecode import format_ms
from .extract import instructions_block
from .integrate import parse_key
from .materials import ITEM_TYPES, MaterialHit, material_hits
from .memory import Item, Section, assemble, input_budget, snippet
from .scoring import excluded

STAR_TYPES = {"exam_hint", "scope_include"}
MATERIAL_STAR_KINDS = {"past_exam", "mock_exam", "review_outline"}  # 出现在这些资料里的知识点至少 3 星
_SOURCE = re.compile(r"^\s*L\s*(\d+)\s*N\s*(\d+)\s*$", re.I)
_NUMBER = re.compile(r"^\s*[A-Za-z]\s*(\d+)\s*$")
_TITLE_LINE = re.compile(r"\s*(?:#{1,6}[ \t]+(.+?)|\*\*([^*\n]+)\*\*)[ \t]*(?:\n|$)")  # 开头单独一行的标题或粗体
_PAREN = re.compile(r"[（(][^（）()]*[）)]")


class StoredEntry(BaseModel):
    key: int
    summary_md: str
    sources: list[list[int]]  # [讲的编号, 笔记中第几个知识点]
    signals: list[int]  # 模型引用的原话（信号编号），按引用顺序
    examples: list[list[int]]  # [讲的编号, 第几道例题]
    generated: bool  # False 表示模型漏写、由代码补的


class StoredChapter(BaseModel):
    overview_md: str
    entries: list[StoredEntry]


def kp_priority(signals: Sequence[HistorySignal], lecture_count: int, hits: Sequence[MaterialHit] = ()) -> int:
    """重要程度：3 提示考试/考试范围、强度 3 的强调，或出现在往年真题、模拟题、复习提纲里；
    2 强调、易错或在 2 讲以上出现；1 其他；0 老师明确说不考或考试通知说不考（且没有说要考）。"""
    if excluded(signals, hits):
        return 0
    types = {s.type for s in signals} | {h.item.item_type for h in hits if h.material.kind == "notice"}
    in_materials = any(h.material.kind in MATERIAL_STAR_KINDS for h in hits)
    if types & STAR_TYPES or in_materials or any(s.type == "emphasis" and s.strength >= 3 for s in signals):
        return 3
    if types & {"emphasis", "common_mistake"} or lecture_count >= 2:
        return 2
    return 1


# ---------- 材料 ----------


class CourseMaterial(LectureCache):
    """一门课写复习讲义所需的材料：在各讲笔记与转写的缓存之上，加课程知识点、已校验的信号和例题归属。"""

    def __init__(self, library: Library, course: Course) -> None:
        super().__init__(library)
        memory = CourseMemory(library)
        self.course = course
        self.tree = memory.tree(course)
        self.signals: dict[int, list[HistorySignal]] = {}
        for signal in memory.history_signals(course):
            self.signals.setdefault(signal.key, []).append(signal)
        self.examples = self._assign_examples(memory)
        self.hits = material_hits(library, course)

    def example(self, lecture_id: int, index: int) -> WorkedExample | None:
        notes = self.notes(lecture_id)
        return notes.examples[index - 1] if notes and 1 <= index <= len(notes.examples) else None

    def _assign_examples(self, memory: CourseMemory) -> dict[int, list[tuple[int, int]]]:
        """例题按句子编号归到离得最近的知识点：{K 编号: [(讲的编号, 第几道例题)]}。"""
        result: dict[int, list[tuple[int, int]]] = {}
        for lecture_id in sorted({a.lecture_id for kp in self.tree for a in kp.appearances}):
            notes = self.notes(lecture_id)
            if notes is None:
                continue
            keys = memory.kp_keys(lecture_id)
            for index, example in enumerate(notes.examples, start=1):
                note_index = nearest_kp(example.segment_ids, notes)
                if note_index in keys:
                    result.setdefault(keys[note_index], []).append((lecture_id, index))
        return result

    def fallback_summary(self, kp: KnowledgePoint) -> str:
        """模型没写时的要点：按讲次列出各讲笔记里的原文。"""
        parts = []
        for a in kp.appearances:
            note = self.note_kp(a.lecture_id, a.note_index)
            if note is not None:
                parts.append(f"**第{a.seq}讲**：{note.summary_md}")
        return "\n\n".join(parts)


def nearest_kp(segment_ids: Sequence[int], notes: LectureNotes) -> int | None:
    best, best_distance = None, None
    for index, kp in enumerate(notes.knowledge_points, start=1):
        distance = _distance(segment_ids, kp.segment_ids)
        if distance is not None and (best_distance is None or distance < best_distance):
            best, best_distance = index, distance
    return best


def _distance(a: Sequence[int], b: Sequence[int]) -> int | None:
    """两段句子编号区间的距离；重叠为 0，任一为空时无法比较。"""
    if not a or not b:
        return None
    if min(a) <= max(b) and min(b) <= max(a):
        return 0
    return min(abs(min(a) - max(b)), abs(min(b) - max(a)))


# ---------- 每章的输入 ----------


@dataclass
class ChapterInput:
    chapter: str
    points: list[KnowledgePoint]
    sections: list[Section]
    sources: dict[str, tuple[int, int]] = field(default_factory=dict)  # "L2N3" → (讲的编号, 第几个知识点)
    quotes: dict[int, int] = field(default_factory=dict)  # Q 编号 → 信号编号
    examples: dict[int, tuple[int, int]] = field(default_factory=dict)  # E 编号 → (讲的编号, 第几道例题)

    def fingerprint(self, system: str) -> str:
        """按未裁剪的材料计算，材料或提示词变了指纹就变。"""
        text = system + "\n".join(item.full for section in self.sections for item in section.items)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def labels(self) -> set[str]:
        """给模型的编号（K、L、Q、E）。"""
        return ({f"K{kp.key}" for kp in self.points} | set(self.sources) | {f"Q{n}" for n in self.quotes}
                | {f"E{n}" for n in self.examples})


def chapter_inputs(material: CourseMaterial) -> list[ChapterInput]:
    return [_chapter_input(material, chapter, points) for chapter, points in group_by_chapter(material.tree)]


def _chapter_input(material: CourseMaterial, chapter: str, points: Sequence[KnowledgePoint]) -> ChapterInput:
    data = ChapterInput(chapter, list(points), [])
    kp_items, lecture_items, quote_items, example_items = [], [], [], []
    for kp in points:
        line = f"- [K{kp.key}] {kp.name}" + (f"（又称：{'、'.join(kp.aliases)}）" if kp.aliases else "") + "｜" + \
            "、".join(f"第{a.seq}讲{ROLE_LABELS.get(a.role, a.role)}" for a in kp.appearances)
        kp_items.append(Item(line, line))
        lecture_items += _lecture_items(material, kp, data)
        quote_items += _quote_items(material, kp, data)
        example_items += _example_items(material, kp, data)
    data.sections = [
        Section("本章知识点", kp_items),
        Section("各讲讲解", lecture_items, trim_rank=3),
        Section("老师原话（已与录音转写核对）", quote_items, trim_rank=2),
        Section("例题", example_items, trim_rank=1),
    ]
    return data


def _lecture_items(material: CourseMaterial, kp: KnowledgePoint, data: ChapterInput) -> list[Item]:
    items = []
    for a in kp.appearances:
        note = material.note_kp(a.lecture_id, a.note_index)
        if note is None:
            continue
        label = f"L{a.seq}N{a.note_index}"
        data.sources[label] = (a.lecture_id, a.note_index)
        start = material.time(a.lecture_id, note.segment_ids)
        when = f" {format_ms(start)}" if start is not None else ""
        head = f"### [{label}] K{kp.key} {kp.name} · 第{a.seq}讲{when} · {ROLE_LABELS.get(a.role, a.role)}"
        items.append(Item(f"{head}\n{note.summary_md}", f"{head}\n{snippet(note.summary_md, 200)}"))
    return items


def _quote_items(material: CourseMaterial, kp: KnowledgePoint, data: ChapterInput) -> list[Item]:
    items = []
    for signal in material.signals.get(kp.key, []):
        number = len(data.quotes) + 1
        data.quotes[number] = signal.signal_id
        start = material.time(signal.lecture_id, signal.segment_ids)
        when = f" {format_ms(start)}" if start is not None else ""
        label = SIGNAL_LABELS.get(signal.type, signal.type)
        text = f"- [Q{number}] K{kp.key} · {label}·强度{signal.strength} · 第{signal.seq}讲{when}：「{signal.quote}」"
        items.append(Item(text, "", keep=signal.strength))
    return items


def _example_items(material: CourseMaterial, kp: KnowledgePoint, data: ChapterInput) -> list[Item]:
    items = []
    seq_of = {a.lecture_id: a.seq for a in kp.appearances}
    for lecture_id, index in material.examples.get(kp.key, []):
        example = material.example(lecture_id, index)
        if example is None:
            continue
        number = len(data.examples) + 1
        data.examples[number] = (lecture_id, index)
        head = f"### [E{number}] K{kp.key} · 第{seq_of.get(lecture_id, '?')}讲 · {example.title}"
        items.append(Item(f"{head}\n{example.problem_md}\n解答：{snippet(example.solution_md, 400)}", head))
    return items


# ---------- 生成 ----------


def build_system_prompt(course: Course) -> str:
    settings = course.settings
    return load_prompt(
        "review",
        subject=settings.subject,
        teaching=settings.teaching_description,
        note_language=settings.note_language_name,
        term_rule=settings.term_rule,
        instructions=instructions_block(course.instructions),
    )


@dataclass
class PlannedChapter:
    input: ChapterInput
    fingerprint: str
    needed: bool  # 需要（重新）生成


def plan_review(library: Library, course: Course, regenerate_all: bool = False) -> list[PlannedChapter]:
    material = CourseMaterial(library, course)
    system = build_system_prompt(course)
    stored = CourseMemory(library).reviews(course.id)
    planned = []
    for chapter in chapter_inputs(material):
        fingerprint = chapter.fingerprint(system)
        record = stored.get(chapter.chapter)
        planned.append(PlannedChapter(chapter, fingerprint,
                                      regenerate_all or record is None or record.fingerprint != fingerprint))
    return planned


@dataclass
class ReviewContext:
    library: Library
    provider: LLMProvider
    effort: Effort | None = None
    budget: int = 60_000
    report: Callable[[str], None] = lambda _: None  # 每章生成完的进度说明


def review_context(cfg: AppConfig, library: Library, report: Callable[[str], None],
                   model: str | None = None) -> ReviewContext:
    """按 [steps] review（或指定的模型档案）、推理强度和输入预算生成复习讲义（命令行、总复习任务和考前报告共用）。"""
    profile = cfg.profile_for_step("review", model)
    return ReviewContext(library, create_provider(profile), effort=cfg.effort_for_step("review"),
                         budget=cfg.memory.review_budget, report=report)


def update_review_for(cfg: AppConfig, library: Library, course: Course, keys: Sequence[int],
                      report: Callable[[str], None], generate: bool = True) -> dict[int, EntryView]:
    """先把这些知识点所在、需要（重新）生成的章节生成好（generate 为假时不生成），再返回它们的复习条目。

    传给 generate_review 的是完整的章节列表，只把相关章节标为需要生成——它会删掉列表里没有的章节。
    """
    wanted = set(keys)
    planned = plan_review(library, course)
    related = {p.input.chapter for p in planned if wanted & {kp.key for kp in p.input.points}}
    planned = [replace(p, needed=generate and p.needed and p.input.chapter in related) for p in planned]
    needed = sum(p.needed for p in planned)
    if needed:
        report(f"正在更新总复习里相关的 {needed} 章…")
        generate_review(review_context(cfg, library, report), course, planned)
    return {entry.key: entry for chapter in review_book(library, course) for entry in chapter.entries
            if entry.key in wanted}


def generate_review(ctx: ReviewContext, course: Course, planned: Sequence[PlannedChapter]) -> list[str]:
    """生成需要更新的章节，返回各章的处理说明；已经不存在的章节一并删除。"""
    memory = CourseMemory(ctx.library)
    profile = ctx.provider.profile
    system = build_system_prompt(course)
    notes = []
    for item in (p for p in planned if p.needed):
        ask = f"请为「{item.input.chapter}」这一章写复习讲义。"
        assembled = assemble(item.input.sections, input_budget(profile, ctx.budget), profile,
                             fixed_chars=len(system) + len(ask))
        result = ctx.provider.generate_structured(
            [TextPart(assembled.text), TextPart(ask)], ReviewChapter, system=system, effort=ctx.effort
        )
        stored = resolve_review(result.output, item.input)
        memory.save_review(course.id, item.input.chapter, item.fingerprint, stored.model_dump_json(), profile)
        ctx.library.log_call(None, "review", profile, result.usage, course_id=course.id)
        missing = sum(not entry.generated for entry in stored.entries)
        note = f"{item.input.chapter}：{len(stored.entries)} 个知识点" + (f"（{missing} 个由程序补全）" if missing else "")
        ctx.report(note)
        notes.append(note)
    memory.prune_reviews(course.id, [p.input.chapter for p in planned])
    return notes


def resolve_review(output: ReviewChapter, chapter: ChapterInput) -> StoredChapter:
    """只保留本章知识点的条目；编号引用换成书库里的位置，无效的丢弃；模型漏写的知识点由代码补上。"""
    names = {kp.key: kp.name for kp in chapter.points}
    labels = chapter.labels()
    entries: dict[int, StoredEntry] = {}
    for entry in output.entries:
        key = parse_key(entry.kp)
        if key not in names or key in entries:
            continue
        entries[key] = StoredEntry(
            key=key,
            summary_md=_without_title(_clean(entry.summary_md, labels), names[key]),
            sources=[list(chapter.sources[label]) for label in _labels(entry.sources) if label in chapter.sources],
            signals=[chapter.quotes[n] for n in _numbers(entry.emphasis) if n in chapter.quotes],
            examples=[list(chapter.examples[n]) for n in _numbers(entry.examples) if n in chapter.examples],
            generated=True,
        )
    ordered = [entries.get(kp.key) or StoredEntry(key=kp.key, summary_md="", sources=[], signals=[], examples=[],
                                                  generated=False)
               for kp in chapter.points]
    return StoredChapter(overview_md=_clean(output.overview_md, labels), entries=ordered)


def _clean(text: str, labels: set[str]) -> str:
    """保存前去掉外语旁注和引用编号：提示词要求了，模型仍会漏。"""
    return strip_citations(strip_foreign_asides(text), labels).strip()


def _without_title(summary_md: str, name: str) -> str:
    """去掉要点开头重复知识点名称的标题，页面已经显示名称：标题就是名称（不看括号），或名称是「标题的……」
    （「**第V类：负极性单链RNA病毒**」对「第V类：负极性单链RNA病毒的增殖策略」）。
    小节标题保留：「### 1. 崩解」，「**噬菌体药物**」对「噬菌体药物与噬菌体裂解酶」。"""
    match = _TITLE_LINE.match(summary_md)
    if match is None:
        return summary_md
    title, wanted = (normalize_zh(_PAREN.sub("", text)) for text in (match.group(1) or match.group(2), name))
    repeated = bool(title) and (wanted == title or wanted.startswith(title + "的"))
    return summary_md[match.end():].lstrip() if repeated else summary_md


def _labels(values: Sequence[str]) -> list[str]:
    labels = []
    for value in values:
        match = _SOURCE.match(value or "")
        if match:
            labels.append(f"L{int(match.group(1))}N{int(match.group(2))}")
    return labels


def _numbers(values: Sequence[str]) -> list[int]:
    return [int(m.group(1)) for m in (_NUMBER.match(v or "") for v in values) if m]


# ---------- 展示（网页与 总复习.md 共用） ----------


@dataclass
class EntryView:
    key: int
    name: str
    aliases: tuple[str, ...]
    norm_name: str
    priority: int
    summary_md: str
    generated: bool
    reviewed: bool
    sources: list[dict[str, Any]]
    quotes: list[dict[str, Any]]
    examples: list[dict[str, Any]]
    materials: list[dict[str, Any]] = field(default_factory=list)  # 在课程资料里出现的题目或条目


@dataclass
class ChapterView:
    chapter: str
    overview_md: str
    entries: list[EntryView]
    state: str  # ready 已生成 / stale 待更新 / missing 还没有生成
    generated_at: str
    model: str


def review_book(library: Library, course: Course) -> list[ChapterView]:
    """整本书的复习讲义：已生成的章节用模型写的要点，没生成的用各讲笔记原文；重要程度和原话都由代码填入。"""
    material = CourseMaterial(library, course)
    system = build_system_prompt(course)
    stored = CourseMemory(library).reviews(course.id)
    reviewed = library.reviewed_names(course.id)
    views = []
    for chapter in chapter_inputs(material):
        record = stored.get(chapter.chapter)
        data = StoredChapter.model_validate_json(record.result_json) if record else None
        by_key = {entry.key: entry for entry in data.entries} if data else {}
        entries = [_entry_view(material, kp, by_key.get(kp.key), reviewed) for kp in chapter.points]
        entries.sort(key=lambda e: e.priority == 0)  # 老师说不考的排在最后，其余保持讲授顺序
        views.append(ChapterView(chapter.chapter, data.overview_md if data else "", entries,
                                 _chapter_state(record, chapter.fingerprint(system)),
                                 record.created_at if record else "", record.model if record else ""))
    return views


def _chapter_state(record: StoredReview | None, fingerprint: str) -> str:
    if record is None:
        return "missing"
    return "ready" if record.fingerprint == fingerprint else "stale"


def _entry_view(
    material: CourseMaterial, kp: KnowledgePoint, entry: StoredEntry | None, reviewed: set[str]
) -> EntryView:
    """模型写过的条目用它的要点和引用；没写过的用各讲笔记原文。原话总是列出这个知识点全部通过校验的信号。"""
    entry = entry if entry and entry.generated else None
    signals = material.signals.get(kp.key, [])
    hits = material.hits.get(kp.key, [])
    return EntryView(
        key=kp.key, name=kp.name, aliases=kp.aliases, norm_name=normalize_zh(kp.name),
        priority=kp_priority(signals, len({a.seq for a in kp.appearances}), hits),
        summary_md=(entry.summary_md if entry else "") or material.fallback_summary(kp),
        generated=entry is not None, reviewed=bool(kp.norm_names & reviewed),
        sources=[_source_view(material, kp, *source) for source in _entry_sources(kp, entry)],
        quotes=[_quote_view(material, s) for s in _ordered_quotes(signals, entry.signals if entry else [])],
        examples=_example_views(material, kp, entry),
        materials=[hit_view(hit) for hit in hits],
    )


def hit_view(hit: MaterialHit) -> dict[str, Any]:
    """资料里的一条题目：「模拟题（样例）」一、3（名词解释，5 分）。"""
    item = hit.item
    extra = [ITEM_TYPES.get(item.item_type, "其他")] + ([f"{item.score:g} 分"] if item.score else [])
    return {"material_id": hit.material.id, "title": hit.material.title, "kind": hit.material.kind_label,
            "label": item.label or f"第 {item.idx} 条", "type": "，".join(extra), "content_md": item.content_md,
            "pages": item.pages}


def _entry_sources(kp: KnowledgePoint, entry: StoredEntry | None) -> list[tuple[int, int]]:
    """模型引用的讲解；没有引用时列出这个知识点出现过的每一讲。"""
    if entry and entry.sources:
        return [(source[0], source[1]) for source in entry.sources]
    return [(a.lecture_id, a.note_index) for a in kp.appearances]


def _example_views(material: CourseMaterial, kp: KnowledgePoint, entry: StoredEntry | None) -> list[dict[str, Any]]:
    views = (_example_view(material, kp, *example) for example in _entry_examples(material, kp, entry))
    return [view for view in views if view]


def _ordered_quotes(signals: Sequence[HistorySignal], cited: Sequence[int]) -> list[HistorySignal]:
    """模型引用的原话按引用顺序排在前面，其余按强度从高到低。"""
    rank = {signal_id: position for position, signal_id in enumerate(cited)}
    return sorted(signals, key=lambda s: (rank.get(s.signal_id, len(rank)), -s.strength, s.seq))


def _entry_examples(material: CourseMaterial, kp: KnowledgePoint, entry: StoredEntry | None) -> list[tuple[int, int]]:
    """模型引用的例题在前，再补上按句子编号归到这个知识点的其他例题。"""
    examples = [(e[0], e[1]) for e in entry.examples] if entry else []
    return examples + [e for e in material.examples.get(kp.key, []) if e not in examples]


def _source_view(material: CourseMaterial, kp: KnowledgePoint, lecture_id: int, note_index: int) -> dict[str, Any]:
    appearance = next((a for a in kp.appearances if a.lecture_id == lecture_id), None)
    note = material.note_kp(lecture_id, note_index)
    return {
        "seq": appearance.seq if appearance else 0,
        "lecture_id": lecture_id,
        "role": ROLE_LABELS.get(appearance.role, "") if appearance else "",
        "t": material.time(lecture_id, note.segment_ids) if note else None,
    }


def _quote_view(material: CourseMaterial, signal: HistorySignal) -> dict[str, Any]:
    return {"label": SIGNAL_LABELS.get(signal.type, signal.type), "type": signal.type, "strength": signal.strength,
            "quote": signal.quote, "meaning": signal.meaning, "seq": signal.seq, "lecture_id": signal.lecture_id,
            "t": material.time(signal.lecture_id, signal.segment_ids)}


def _example_view(material: CourseMaterial, kp: KnowledgePoint, lecture_id: int, index: int) -> dict[str, Any] | None:
    example = material.example(lecture_id, index)
    if example is None:
        return None
    seq = next((a.seq for a in kp.appearances if a.lecture_id == lecture_id), 0)
    return {"title": example.title, "problem_md": example.problem_md, "solution_md": example.solution_md,
            "seq": seq, "lecture_id": lecture_id, "t": material.time(lecture_id, example.segment_ids)}


def _quote_line(quote: dict[str, Any]) -> str:
    when = f" {format_ms(quote['t'])}" if quote["t"] is not None else ""
    return f"- {quote['label']}（第{quote['seq']}讲{when}）：{quote_html(quote['quote'], quote.get('meaning', ''))}"


def render_review_markdown(course: Course, chapters: Sequence[ChapterView]) -> str:
    """总复习.md：按章节列出知识点、重要程度、复习要点、老师原话（附讲次和时间）和例题。"""
    lines = [f"# {course.name} · 总复习", "",
             "重要程度：★★★ 老师提示考试或明确划重点，或出现在往年真题、模拟题、复习提纲里；★★ 强调、易错或多次讲到；"
             "★ 一般；（不考）老师或考试通知说不考。"]
    for chapter in chapters:
        lines += ["", f"## {chapter.chapter}"]
        if chapter.state != "ready":
            lines.append("> " + ("本章内容有更新，复习讲义待重新生成。" if chapter.state == "stale" else "本章还没有生成复习讲义，下面是各讲笔记原文。"))
        if chapter.overview_md:
            lines += ["", chapter.overview_md]
        for entry in chapter.entries:
            stars = "（不考）" if entry.priority == 0 else "★" * entry.priority
            shown = zh_aliases(entry.aliases)
            lines += ["", f"### {entry.name} {stars}", *([f"又称：{'、'.join(shown)}"] if shown else []), "",
                      entry.summary_md]
            lines += [_quote_line(q) for q in entry.quotes]
            lines += [f"- 例题：{e['title']}（第{e['seq']}讲）" for e in entry.examples]
            lines += [f"- 资料：「{m['title']}」{m['label']}（{m['type']}）" for m in entry.materials]
    return "\n".join(lines) + "\n"
