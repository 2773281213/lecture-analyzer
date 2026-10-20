"""第二步：跨讲整合。把本讲笔记归并进课程知识点体系（L2），并找出与前几讲的联系。

模型只判断"本讲的知识点对应哪个已有知识点、属于哪一章、和前几讲有什么联系"；
编号分配、校验纠错（resolve）和写入由代码完成。课程记忆按讲次顺序叠加，
前面已是最新的讲直接重放整合日志，不重复调用模型（knowledge.CourseMemory）。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Sequence

from rapidfuzz import fuzz, process

from ..asr.base import Segment
from ..course import recap_pattern
from ..knowledge import (
    CourseMemory, HistorySignal, KnowledgePoint, LectureCache, ResolvedIntegration, ResolvedKP, ResolvedReference,
)
from ..library import CheckedSignal, Course, Lecture, Library
from ..llm import LLMProvider, TextPart
from ..llm.base import Effort
from ..prompts import load_prompt
from ..render import ROLE_LABELS, SIGNAL_LABELS, first_time
from ..schemas import CrossReference, IntegrationResult, KnowledgePointMapping, LectureNotes, Signal
from ..textnorm import normalize_zh, strip_foreign_asides
from ..timecode import format_ms
from .extract import instructions_block
from .memory import Item, Section, assemble, input_budget, snippet, trimmed_note

MAX_RECAP_CUES = 40
CANDIDATE_CUTOFF = 60  # 名称模糊匹配的最低分（partial_ratio）
SIGNAL_MATCH_CUTOFF = 80
_KEY = re.compile(r"^\s*[Kk]\s*(\d+)\s*$")


@dataclass
class IntegrateContext:
    library: Library
    memory: CourseMemory
    provider: LLMProvider
    effort: Effort | None = None
    budget: int = 60_000  # 配置里的输入预算上限（还受模型上下文约束）
    recent_lectures: int = 2
    report: Callable[[str], None] = field(default=lambda message: None)  # 进度（每整合一讲报告一次）


@dataclass
class IntegrateOutcome:
    integrated: list[Lecture] = field(default_factory=list)  # 本次调用模型整合的讲
    replayed: int = 0  # 直接重放日志的讲数
    stale: list[Lecture] = field(default_factory=list)  # 被标记为待更新的后续讲
    details: dict[int, str] = field(default_factory=dict)  # 讲的编号 → 处理说明


@dataclass
class ResolveStats:
    new: int = 0
    linked: int = 0
    references: int = 0
    unmatched_signals: int = 0

    def describe(self, model: str) -> str:
        text = f"新增 {self.new} 个知识点、关联已有 {self.linked} 个、跨讲联系 {self.references} 条"
        if self.unmatched_signals:
            text += f"、{self.unmatched_signals} 条信号未归类"
        return f"{text}（{model}）"


# ---------- 按讲次推进课程记忆 ----------


def up_to_date(memory: CourseMemory, lecture: Lecture) -> bool:
    return lecture.step_state("integrate") == "done" and memory.log(lecture.id) is not None


def eligible_lectures(library: Library, course: Course) -> list[Lecture]:
    """可以整合的讲：原话校验已完成。"""
    return [lecture for lecture in library.lectures(course) if lecture.step_state("validate") == "done"]


def integrate_through(ctx: IntegrateContext, course: Course, target: Lecture) -> IntegrateOutcome:
    """让课程记忆覆盖到 target 这一讲（含）。

    从头重建：前面已是最新的讲直接重放日志；从第一讲需要整合的讲起，到 target 为止逐讲调用模型
    （target 本身总会重新整合）；target 之后已整合过的讲标记为待更新。
    """
    library, memory = ctx.library, ctx.memory
    memory.reset(course.id)
    outcome = IntegrateOutcome()
    applied: list[Lecture] = []
    pending = False
    for lecture in eligible_lectures(library, course):
        if lecture.seq > target.seq:
            if mark_stale(library, course, lecture):
                outcome.stale.append(lecture)
            continue
        log = memory.log(lecture.id)
        if not pending and lecture.id != target.id and log is not None and lecture.step_state("integrate") == "done":
            memory.apply(course, lecture.id, log)
            outcome.replayed += 1
        else:
            pending = True
            ctx.report(f"整合课程记忆：第 {lecture.seq} 讲…")
            detail = _integrate_with_status(ctx, course, lecture, applied)
            outcome.integrated.append(lecture)
            outcome.details[lecture.id] = detail
        applied.append(lecture)
    return outcome


def plan_rebuild(library: Library, memory: CourseMemory, course: Course, from_seq: int | None) -> list[Lecture]:
    """重建时需要调用模型的讲：从第一讲需要更新的讲（或 from_seq）起到最后一讲。"""
    lectures = eligible_lectures(library, course)
    for index, lecture in enumerate(lectures):
        if (from_seq is not None and lecture.seq >= from_seq) or not up_to_date(memory, lecture):
            return lectures[index:]
    return []


def rebuild(ctx: IntegrateContext, course: Course, from_seq: int | None = None) -> IntegrateOutcome | None:
    """重建课程记忆；已是最新时返回 None。"""
    planned = plan_rebuild(ctx.library, ctx.memory, course, from_seq)
    if not planned:
        return None
    for lecture in planned:
        ctx.library.reset_steps(lecture.id, ["integrate"])
    return integrate_through(ctx, course, planned[-1])


def restore_prefix(library: Library, memory: CourseMemory, course: Course, before_seq: int) -> list[Lecture]:
    """重置课程记忆，按讲次重放 before_seq 之前、一直是最新的讲；其后整合过的讲标记为待更新并返回。

    用于不调用模型就让课程记忆回到一致状态，例如把一讲移出或移入这门课之后。
    """
    memory.reset(course.id)
    stale: list[Lecture] = []
    consistent = True
    for lecture in eligible_lectures(library, course):
        log = memory.log(lecture.id)
        if consistent and lecture.seq < before_seq and log is not None and lecture.step_state("integrate") == "done":
            memory.apply(course, lecture.id, log)
            continue
        consistent = False
        if mark_stale(library, course, lecture):
            stale.append(lecture)
    return stale


def mark_stale(library: Library, course: Course, lecture: Lecture) -> bool:
    """整合过的讲标记为待更新；还没整合过的不用标记。"""
    if lecture.step_state("integrate") not in ("done", "stale"):
        return False
    detail = f"前面的讲有更新，课程记忆待更新：在课程页点「重建课程记忆」（命令行：`lec rebuild {course.name}`）"
    library.set_step(lecture.id, "integrate", "stale", detail)
    return True


def _integrate_with_status(ctx: IntegrateContext, course: Course, lecture: Lecture, applied: list[Lecture]) -> str:
    try:
        detail = integrate_lecture(ctx, course, lecture, applied)
    except Exception as exc:  # 记录失败原因后照常抛出
        ctx.library.set_step(lecture.id, "integrate", "failed", f"{type(exc).__name__}: {exc}")
        raise
    ctx.library.set_step(lecture.id, "integrate", "done", detail)
    return detail


# ---------- 整合一讲 ----------


@dataclass
class LectureInput:
    lecture: Lecture
    notes: LectureNotes
    segments: list[Segment]
    checked: list[CheckedSignal]


def integrate_lecture(ctx: IntegrateContext, course: Course, lecture: Lecture, applied: Sequence[Lecture]) -> str:
    """调用模型整合一讲，写入整合日志和课程记忆，返回处理说明。"""
    library, memory, profile = ctx.library, ctx.memory, ctx.provider.profile
    notes = library.notes(lecture.id)
    if notes is None:
        raise RuntimeError(f"第 {lecture.seq} 讲还没有笔记，无法整合")
    item = LectureInput(lecture, notes, library.segments(lecture.id), library.signals(lecture.id))
    tree = memory.tree(course)
    system = build_system_prompt(course)
    ask = f"请按要求输出第 {lecture.seq} 讲的整合结果。"
    recent = applied[-ctx.recent_lectures:] if ctx.recent_lectures else []
    sections = build_sections(memory, course, item, tree, recent)
    assembled = assemble(sections, input_budget(profile, ctx.budget), profile, fixed_chars=len(system) + len(ask))
    result = ctx.provider.generate_structured(
        [TextPart(assembled.text), TextPart(ask)], IntegrationResult, system=system, effort=ctx.effort
    )
    resolved, stats = resolve(result.output, notes, lecture.seq, len(item.segments), tree)
    memory.save_log(lecture.id, resolved, profile)
    memory.apply(course, lecture.id, resolved)
    library.log_call(lecture.id, "integrate", profile, result.usage)
    return stats.describe(profile.model) + trimmed_note(assembled.trimmed)


def build_system_prompt(course: Course) -> str:
    settings = course.settings
    return load_prompt(
        "integrate",
        subject=settings.subject,
        teaching=settings.teaching_description,
        note_language=settings.note_language_name,
        term_rule=settings.term_rule,
        instructions=instructions_block(course.instructions),
    )


def build_sections(
    memory: CourseMemory, course: Course, item: LectureInput, tree: Sequence[KnowledgePoint], recent: Sequence[Lecture]
) -> list[Section]:
    """整合的输入，按显示顺序排列；裁剪顺序见各部分的 trim_rank。"""
    candidates = candidate_keys(item.notes, tree, recent[-1].id if recent else None)
    cache = LectureCache(memory.library)
    return [
        _tree_section(tree, candidates, cache),
        _recent_section(memory, recent, cache),
        _history_section(memory.history_signals(course), tree, candidates, cache),
        _lecture_section(item),
        _cue_section(item.segments, recap_pattern(course.settings.spoken_languages)),
    ]


def candidate_keys(notes: LectureNotes, tree: Sequence[KnowledgePoint], previous_lecture_id: int | None) -> set[int]:
    """可能与本讲有关的已有知识点：名称（含别名）与本讲知识点、信号名称模糊匹配，再加上一讲的全部知识点。"""
    keys = {kp.key for kp in tree if any(a.lecture_id == previous_lecture_id for a in kp.appearances)}
    labels = [(kp.key, name) for kp in tree for name in kp.norm_names if name]
    choices = [name for _, name in labels]
    queries = [kp.name for kp in notes.knowledge_points] + [s.knowledge_point for s in notes.signals]
    for query in (normalize_zh(q) for q in queries):
        if not query or not choices:
            continue
        for _, _, index in process.extract(query, choices, scorer=fuzz.partial_ratio, limit=5,
                                           score_cutoff=CANDIDATE_CUTOFF):
            keys.add(labels[index][0])
    return keys


def _tree_section(tree: Sequence[KnowledgePoint], candidates: set[int], cache: LectureCache) -> Section:
    items = []
    for kp in tree:
        head = _kp_line(kp)
        first = cache.note_kp(kp.appearances[0].lecture_id, kp.appearances[0].note_index)
        full = f"{head}\n    要点：{snippet(first.summary_md, 120)}" if first and first.summary_md else head
        items.append(Item(full, head, keep=1 if kp.key in candidates else 0))
    return Section(f"课程知识点（截至上一讲，共 {len(tree)} 个）", items, trim_rank=3,
                   empty_text="（这是本课程第一次整合，还没有已有知识点）")


def _kp_line(kp: KnowledgePoint) -> str:
    parts = [f"[K{kp.key}] {kp.name}"]
    if kp.chapter:
        parts.append(f"章节：{kp.chapter}")
    if kp.aliases:
        parts.append("别名：" + "、".join(kp.aliases))
    parts.append("、".join(f"第{a.seq}讲{ROLE_LABELS.get(a.role, a.role)}" for a in kp.appearances))
    return "｜".join(parts)


def _recent_section(memory: CourseMemory, recent: Sequence[Lecture], cache: LectureCache) -> Section:
    items = []
    for lecture in recent:
        notes = cache.notes(lecture.id)
        if notes is None:
            continue
        keys = memory.kp_keys(lecture.id)
        lines = [f"### 第 {lecture.seq} 讲：{lecture.title or notes.title}", "大纲：" + "；".join(notes.outline[:15])]
        for index, kp in enumerate(notes.knowledge_points, start=1):
            label = f"[K{keys[index]}] " if index in keys else ""
            lines.append(f"- {label}{kp.name}：{snippet(kp.summary_md, 100)}")
        items.append(Item("\n".join(lines), "", keep=lecture.seq))
    return Section("最近几讲的笔记摘要", items, trim_rank=1)


def _history_section(
    history: Sequence[HistorySignal], tree: Sequence[KnowledgePoint], candidates: set[int], cache: LectureCache
) -> Section:
    names = {kp.key: kp.name for kp in tree}
    items = []
    for number, signal in enumerate((s for s in history if s.key in candidates), start=1):
        start = cache.time(signal.lecture_id, signal.segment_ids)
        when = f" {format_ms(start)}" if start is not None else ""
        label = SIGNAL_LABELS.get(signal.type, signal.type)
        text = (f"[H{number}] 第{signal.seq}讲{when}·{label}·强度{signal.strength}"
                f" → [K{signal.key}] {names.get(signal.key, '')}：「{signal.quote}」")
        items.append(Item(text, "", keep=signal.strength * 1000 + signal.seq))
    return Section("相关的历史原话", items, trim_rank=2)


def _lecture_section(item: LectureInput) -> Section:
    lines = ["### 知识点"]
    for index, kp in enumerate(item.notes.knowledge_points, start=1):
        start = first_time(kp.segment_ids, item.segments)
        when = f"，{format_ms(start)} 起" if start is not None else ""
        lines.append(f"[N{index}] {kp.name}（约 {kp.minutes:g} 分钟{when}）\n    要点：{snippet(kp.summary_md, 300)}")
    lines.append("### 考点信号")
    for index, checked in enumerate(item.checked, start=1):
        lines.append(_signal_line(index, checked, item.segments))
    if not item.checked:
        lines.append("（无）")
    return Section(f"本讲笔记（第 {item.lecture.seq} 讲）", [Item(line, line) for line in lines])


def _signal_line(index: int, checked: CheckedSignal, segments: Sequence[Segment]) -> str:
    signal = checked.signal
    start = first_time(checked.segment_ids, segments)
    when = f"·{format_ms(start)}" if start is not None else ""
    mark = "" if checked.valid else "（原话未找到）"
    label = SIGNAL_LABELS.get(signal.type, signal.type)
    return f"[G{index}] {label}·强度{signal.strength}{when}·知识点「{signal.knowledge_point}」：「{signal.quote}」{mark}"


def _cue_section(segments: Sequence[Segment], cues: re.Pattern[str] | None) -> Section:
    """本讲转写中提到以前内容的句子；授课语言没有回顾词表（cues 为 None）时这一部分为空。"""
    items = []
    for index, segment in enumerate(segments if cues is not None else [], start=1):
        if len(items) >= MAX_RECAP_CUES:
            break
        if cues.search(segment.text):
            line = f"[S{index}][{format_ms(segment.start_ms)}] {segment.text}"
            items.append(Item(line, "", keep=-len(items)))  # 越靠后的线索越先裁剪
    return Section("回顾线索（本讲转写中提到以前内容的句子）", items, trim_rank=4)


# ---------- 校验与归一 ----------


def resolve(
    result: IntegrationResult, notes: LectureNotes, seq: int, segment_count: int, tree: Sequence[KnowledgePoint]
) -> tuple[ResolvedIntegration, ResolveStats]:
    """把模型输出变成可写入的整合结果：补全缺漏、纠正无效编号、按笔记顺序分配新编号。"""
    stats = ResolveStats()
    kps = _resolve_kps(result.knowledge_points, notes, tree, stats)
    note_keys = {kp.note_index: kp.key for kp in kps}
    signal_keys = _resolve_signals(result, notes, note_keys, stats)
    references = _resolve_references(result.references, notes, seq, segment_count, tree)
    stats.references = len(references)
    return ResolvedIntegration(kps=kps, signal_keys=signal_keys, references=references), stats


def _resolve_kps(
    mappings: Sequence[KnowledgePointMapping], notes: LectureNotes, tree: Sequence[KnowledgePoint], stats: ResolveStats
) -> list[ResolvedKP]:
    by_index: dict[int, KnowledgePointMapping] = {}
    for mapping in mappings:
        by_index.setdefault(mapping.note_kp, mapping)
    existing = {kp.key for kp in tree}
    by_name = {name: kp.key for kp in tree for name in kp.norm_names if name}
    new_keys: dict[str, int] = {}
    next_key = max(existing, default=0) + 1
    resolved = []
    for index, note_kp in enumerate(notes.knowledge_points, start=1):
        mapping = by_index.get(index)
        chapter = strip_foreign_asides(mapping.chapter).strip() if mapping else ""
        aliases = [note_kp.name, *(mapping.aliases if mapping else [])]
        key = _existing_key(mapping, note_kp.name, existing, by_name)
        if key is not None:
            role = mapping.role if mapping and mapping.role != "first" else "deepen"
            resolved.append(ResolvedKP(note_index=index, key=key, is_new=False, name=note_kp.name, chapter=chapter,
                                       aliases=aliases, role=role, minutes=note_kp.minutes))
            stats.linked += 1
            continue
        name = strip_foreign_asides(mapping.name if mapping and mapping.name.strip() else note_kp.name).strip()
        norm = normalize_zh(name) or name
        is_new = norm not in new_keys
        if is_new:
            new_keys[norm] = next_key
            next_key += 1
            stats.new += 1
        resolved.append(ResolvedKP(note_index=index, key=new_keys[norm], is_new=is_new, name=name, chapter=chapter,
                                   aliases=aliases, role="first", minutes=note_kp.minutes))
    return resolved


def _existing_key(
    mapping: KnowledgePointMapping | None, note_name: str, existing: set[int], by_name: dict[str, int]
) -> int | None:
    """模型给的 K 编号有效就用；否则名称（规范名称或本讲叫法）与已有知识点的名称、别名相同时并入。"""
    if mapping is not None:
        key = parse_key(mapping.target)
        if key in existing:
            return key
    for name in ((mapping.name if mapping else ""), note_name):
        norm = normalize_zh(name)
        if norm and norm in by_name:
            return by_name[norm]
    return None


def parse_key(text: str) -> int | None:
    match = _KEY.match(text or "")
    return int(match.group(1)) if match else None


def _resolve_signals(
    result: IntegrationResult, notes: LectureNotes, note_keys: dict[int, int], stats: ResolveStats
) -> list[int | None]:
    chosen: dict[int, int] = {}
    for mapping in result.signals:
        chosen.setdefault(mapping.signal, mapping.note_kp)
    keys = []
    for index, signal in enumerate(notes.signals, start=1):
        key = note_keys.get(chosen.get(index, 0)) or _match_signal(signal, notes, note_keys)
        if key is None:
            stats.unmatched_signals += 1
        keys.append(key)
    return keys


def _match_signal(signal: Signal, notes: LectureNotes, note_keys: dict[int, int]) -> int | None:
    """模型没给或给错时，按信号写的知识点名称匹配本讲知识点。"""
    names = [normalize_zh(kp.name) for kp in notes.knowledge_points]
    query = normalize_zh(signal.knowledge_point)
    if not query or not names:
        return None
    match = process.extractOne(query, names, scorer=fuzz.partial_ratio, score_cutoff=SIGNAL_MATCH_CUTOFF)
    return note_keys.get(match[2] + 1) if match else None


def _resolve_references(
    references: Sequence[CrossReference], notes: LectureNotes, seq: int, segment_count: int,
    tree: Sequence[KnowledgePoint],
) -> list[ResolvedReference]:
    earlier = {kp.key: sorted({a.seq for a in kp.appearances if a.seq < seq}) for kp in tree}
    resolved: list[ResolvedReference] = []
    seen: set[tuple[int, int, int, str]] = set()
    for ref in references:
        key = parse_key(ref.target)
        seqs = earlier.get(key, []) if key is not None else []
        if not seqs or not 1 <= ref.note_kp <= len(notes.knowledge_points):
            continue
        target_seq = ref.lecture_seq if ref.lecture_seq in seqs else seqs[-1]
        identity = (ref.note_kp, key, target_seq, ref.relation)
        if identity in seen:
            continue
        seen.add(identity)
        ids = [i for i in ref.segment_ids if 1 <= i <= segment_count]
        if not ids:
            ids = [i for i in notes.knowledge_points[ref.note_kp - 1].segment_ids if 1 <= i <= segment_count]
        resolved.append(ResolvedReference(note_index=ref.note_kp, key=key, target_seq=target_seq,
                                          relation=ref.relation, explanation=strip_foreign_asides(ref.explanation).strip(),
                                          segment_ids=ids))
    return resolved
