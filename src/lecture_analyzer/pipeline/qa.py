"""按课程连续问答：先找出相关知识点，再由代码整理证据，最后由模型依据证据回答并用编号注明出处。

1. 检索（[steps] qa_retrieve，默认与课件匹配同一档案）：输入课程知识点目录（含别名、出现的讲、对应的资料题目）、
   各讲大纲和最近几轮对话，输出相关知识点（最多 8 个）和讲次。提问和笔记可以是不同语言，由模型按含义判断。
2. 证据（代码）：沿用考前报告的编号——原话 Q、资料题目 M、课件页 P、例题 E，另加笔记要点 N 和转写片段 T
   （知识点开始讲的地方，带时间）。限定某一讲时只用这一讲的内容，不用资料题目。
3. 回答（[steps] qa，默认 Claude）：用提问的语言回答，只写证据编号；代码去掉不存在的编号，
   结果里只保留回答引用的证据，网页按编号显示出处（原话跳到录音的时间点、课件页弹出页面图片）。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..asr.base import Segment
from ..config import AppConfig
from ..conversations import QATurn
from ..exam_plans import ExamPlanStore, plan_block
from ..knowledge import CourseMemory, KnowledgePoint, LectureCache
from ..library import Course, Lecture, Library, LibraryError
from ..llm import TextPart, create_provider
from ..prompts import load_prompt
from ..schemas import QAAnswer, QARetrieval
from ..textnorm import strip_foreign_asides
from ..timecode import format_ms
from .extract import instructions_block
from .integrate import parse_key
from .materials import MaterialHit, material_hits
from .memory import Item, Section, assemble, input_budget, snippet, trimmed_note
from .report import normalize_id
from .report_input import Evidence, build_evidence, evidence_items, evidence_sections
from .scoring import Scope, Scoring, score_course

Report = Callable[[str], None]
MAX_KPS = 8
MAX_LECTURES = 3
MAX_APPEARANCES = 3  # 每个知识点最多取几讲的笔记和转写
HISTORY_TURNS = 3  # 追问时带上最近几轮对话
EXCERPT_CHARS = 800  # 转写片段的长度
QA_BUDGET = 40_000  # 每次请求的输入上限（token），还受模型上下文约束
MAX_QUESTION_CHARS = 2000
_CITATION = re.compile(r"\[\s*([A-Za-z]\s*\d+(?:\s*[,，、]\s*[A-Za-z]\s*\d+)*)\s*\]")
_SEPARATORS = re.compile(r"[,，、]")


@dataclass
class QAContext:
    cfg: AppConfig
    library: Library
    course: Course
    question: str
    lecture: Lecture | None  # 限定某一讲；None 表示整门课
    history: Sequence[QATurn]  # 这段对话之前的几轮

    @property
    def lectures(self) -> list[Lecture]:
        return [self.lecture] if self.lecture else self.library.lectures(self.course)


def answer_question(
    cfg: AppConfig, library: Library, course: Course, question: str, lecture: Lecture | None = None,
    history: Sequence[QATurn] = (), report: Report = print,
) -> dict[str, Any]:
    """回答一个问题，返回要存进问答记录的结果（回答、补充、用到的知识点和证据、模型）。"""
    ctx = QAContext(cfg, library, course, _checked_question(question), lecture, history[-HISTORY_TURNS:])
    if lecture is not None and lecture.course_id != course.id:
        raise LibraryError(f"第 {lecture.seq} 讲不属于「{course.name}」")
    tree = _scoped_tree(ctx)
    if not tree:
        where = f"第 {lecture.seq} 讲" if lecture else f"「{course.name}」"
        raise LibraryError(f"{where}还没有整合进课程记忆的内容，先处理这一讲再提问")
    keys, seqs, retrieve_model = _retrieve(ctx, tree, report)
    evidence, kps = _collect(ctx, keys, seqs)
    answer, answer_model, note = _answer(ctx, evidence, kps, seqs, report)
    text, cited = clean_citations(strip_foreign_asides(answer.answer_md), set(evidence.views))
    return {
        "question": ctx.question,
        "answer_md": text.strip(),
        "supplement_md": clean_citations(strip_foreign_asides(answer.supplement_md), set())[0].strip(),
        "found": answer.found,
        "kps": [{"key": kp.key, "name": kp.name} for kp in kps],
        "lectures": seqs,
        "scope": {"lecture_id": lecture.id, "seq": lecture.seq} if lecture else None,
        "evidence": {eid: evidence.views[eid] for eid in cited},
        "models": {"retrieve": retrieve_model, "answer": answer_model},
        "note": note,
    }


def _checked_question(question: str) -> str:
    text = question.strip()
    if not text:
        raise LibraryError("问题不能为空")
    if len(text) > MAX_QUESTION_CHARS:
        raise LibraryError(f"问题最多 {MAX_QUESTION_CHARS} 字")
    return text


def clean_citations(text: str, valid: set[str]) -> tuple[str, list[str]]:
    """只保留存在的证据编号，统一写成 [N2][T1] 的形式；返回（处理后的文字，引用过的编号）。"""
    cited: list[str] = []

    def replace(match: re.Match[str]) -> str:
        kept = [eid for eid in (normalize_id(part) for part in _SEPARATORS.split(match.group(1))) if eid in valid]
        cited.extend(kept)
        return "".join(f"[{eid}]" for eid in kept)

    return _CITATION.sub(replace, text), list(dict.fromkeys(cited))


# ---------- 检索 ----------


def _scoped_tree(ctx: QAContext) -> list[KnowledgePoint]:
    tree = CourseMemory(ctx.library).tree(ctx.course)
    if ctx.lecture is None:
        return tree
    return [kp for kp in tree if any(a.lecture_id == ctx.lecture.id for a in kp.appearances)]


def _retrieve(ctx: QAContext, tree: list[KnowledgePoint], report: Report) -> tuple[list[int], list[int], str]:
    """返回（相关知识点的 K 编号，针对整讲提问时的讲次，模型型号）；只保留范围内存在的。"""
    profile = ctx.cfg.profile_for_step("qa_retrieve")
    system = _prompt(ctx, "qa_retrieve")
    ask = f"学生的问题：{ctx.question}"
    hits = {} if ctx.lecture else material_hits(ctx.library, ctx.course)
    sections = [
        Section("提问范围", [Item(_scope_text(ctx))]),
        Section("最近几轮对话", _history_items(ctx.history, 200)),
        Section("课程知识点", [Item(_catalog_line(kp, hits.get(kp.key, []))) for kp in tree]),
        Section("各讲大纲", [Item(line) for line in _outlines(ctx, ctx.lectures)], trim_rank=1),
    ]
    assembled = assemble(sections, input_budget(profile, QA_BUDGET), profile, fixed_chars=len(system) + len(ask))
    report("正在查找和问题相关的知识点…")
    result = create_provider(profile).generate_structured(
        [TextPart(assembled.text), TextPart(ask)], QARetrieval, system=system,
        effort=ctx.cfg.effort_for_step("qa_retrieve"),
    )
    ctx.library.log_call(None, "qa_retrieve", profile, result.usage, course_id=ctx.course.id)
    valid = {kp.key for kp in tree}
    parsed = dict.fromkeys(parse_key(text) for text in result.output.knowledge_points)
    keys = [key for key in parsed if key is not None and key in valid]
    in_scope = {lecture.seq for lecture in ctx.lectures}
    seqs = [seq for seq in dict.fromkeys(result.output.lectures) if seq in in_scope]
    return keys[:MAX_KPS], seqs[:MAX_LECTURES], profile.model


def _catalog_line(kp: KnowledgePoint, hits: Sequence[MaterialHit]) -> str:
    parts = [f"[K{kp.key}] {kp.name}", *([kp.chapter] if kp.chapter else [])]
    if kp.aliases:
        parts.append("别名：" + "、".join(kp.aliases))
    seqs = sorted({a.seq for a in kp.appearances})
    parts.append("第 " + "、".join(map(str, seqs)) + " 讲" if seqs else "课上没讲到")
    if hits:
        parts.append("资料：" + "、".join(f"「{h.material.title}」{h.item.label or f'第 {h.item.idx} 条'}"
                                          for h in hits[:3]))
    return "｜".join(parts)


def _scope_text(ctx: QAContext) -> str:
    if ctx.lecture is None:
        return f"整门课（共 {len(ctx.lectures)} 讲）"
    return f"只限第 {ctx.lecture.seq} 讲「{ctx.lecture.title or '未命名'}」"


def _history_items(history: Sequence[QATurn], answer_chars: int) -> list[Item]:
    items = []
    for turn in history:
        kps = "、".join(f"K{kp['key']}" for kp in turn.result.get("kps", []))
        text = f"问：{turn.question}\n答：{snippet(turn.result.get('answer_md', ''), answer_chars)}"
        items.append(Item(text + (f"\n涉及知识点：{kps}" if kps else ""), f"问：{turn.question}"))
    return items


def _outlines(ctx: QAContext, lectures: Sequence[Lecture]) -> list[str]:
    cache = LectureCache(ctx.library)
    lines = []
    for lecture in lectures:
        notes = cache.notes(lecture.id)
        title = lecture.title or (notes.title if notes else "")
        outline = "；".join(notes.outline) if notes else "（还没有笔记）"
        lines.append(f"第 {lecture.seq} 讲「{title}」：{snippet(outline, 400)}")
    return lines


# ---------- 证据 ----------


class _LectureEvidence:
    """笔记要点 N 和转写片段 T：同一讲同一个知识点只编一次号，再被引用时只关联。"""

    def __init__(self, evidence: Evidence, cache: LectureCache) -> None:
        self.evidence = evidence
        self.cache = cache
        self.numbers: dict[tuple[str, int, int], str] = {}

    def add(self, key: int, lecture_id: int, seq: int, note_index: int, excerpt: bool) -> None:
        note = self.cache.note_kp(lecture_id, note_index)
        if note is None:
            return
        segments = self.cache.segments(lecture_id)
        self._add("N", key, (lecture_id, note_index), lambda: {
            "label": f"第{seq}讲笔记", "name": note.name, "summary_md": note.summary_md, "seq": seq,
            "lecture_id": lecture_id, "note_index": note_index, "t": self.cache.time(lecture_id, note.segment_ids)})
        ids = [i for i in note.segment_ids if 1 <= i <= len(segments)]
        if excerpt and ids:
            text, start = transcript_excerpt(segments, min(ids))
            self._add("T", key, (lecture_id, note_index), lambda: {
                "label": f"第{seq}讲 {format_ms(start)}", "text": text, "seq": seq, "lecture_id": lecture_id,
                "t": start})

    def _add(self, prefix: str, key: int, where: tuple[int, int], view: Callable[[], dict[str, Any]]) -> None:
        number = self.numbers.get((prefix, *where))
        if number is None:
            self.numbers[(prefix, *where)] = self.evidence.add(prefix, key, view())
        else:
            self.evidence.link(key, number)


def transcript_excerpt(segments: Sequence[Segment], first_id: int, limit: int = EXCERPT_CHARS) -> tuple[str, int]:
    """从第 first_id 句（从 1 起）的前一句开始连续取句子，到 limit 字为止；返回（文字，开始时间）。"""
    start = max(first_id - 2, 0)
    lines: list[str] = []
    total = 0
    for segment in segments[start:]:
        if lines and total + len(segment.text) > limit:
            break
        lines.append(segment.text)
        total += len(segment.text)
    return " ".join(lines), segments[start].start_ms


def _collect(ctx: QAContext, keys: Sequence[int], seqs: Sequence[int]) -> tuple[Evidence, list[KnowledgePoint]]:
    """相关知识点的证据（Q、M、P、E、N、T），以及针对整讲提问时这几讲各知识点的笔记要点。"""
    scope = Scope(frozenset({ctx.lecture.id}), frozenset()) if ctx.lecture else Scope()
    by_key = score_course(ctx.library, ctx.course, ctx.cfg.scoring, scope).by_key()
    items = [by_key[key] for key in keys if key in by_key]
    picked = Scoring([item for item in items if not item.excluded], [item for item in items if item.excluded])
    evidence = build_evidence(ctx.library, ctx.course, picked, scope)
    lecture_evidence = _LectureEvidence(evidence, LectureCache(ctx.library))
    for item in items:
        for appearance in item.kp.appearances[:MAX_APPEARANCES]:
            lecture_evidence.add(item.kp.key, appearance.lecture_id, appearance.seq, appearance.note_index, True)
    memory = CourseMemory(ctx.library)
    for lecture in (lecture for lecture in ctx.lectures if lecture.seq in seqs):
        # 笔记里第几个知识点 → K 编号；还没整合进课程记忆的知识点不用
        for index, key in sorted(memory.kp_keys(lecture.id).items()):
            lecture_evidence.add(key, lecture.id, lecture.seq, index, False)
    return evidence, [item.kp for item in items]


# ---------- 回答 ----------


def _answer(
    ctx: QAContext, evidence: Evidence, kps: Sequence[KnowledgePoint], seqs: Sequence[int], report: Report
) -> tuple[QAAnswer, str, str]:
    """返回（回答，模型型号，输入被精简时的说明）。"""
    profile = ctx.cfg.profile_for_step("qa")
    system = _prompt(ctx, "qa_answer")
    ask = f"学生的问题：{ctx.question}"
    sections = [
        Section("提问范围", [Item(_scope_text(ctx))]),
        Section("之前的对话", _history_items(ctx.history, 600), trim_rank=2),
        Section("相关知识点", [Item(_kp_line(kp)) for kp in kps], empty_text="（没有找到相关的知识点）"),
        Section("各讲大纲", [Item(line) for line in _outlines(ctx, [x for x in ctx.lectures if x.seq in seqs])]),
        Section("笔记要点", evidence_items(evidence, "N", _note_text), trim_rank=3),
        Section("转写片段（老师讲课的原文）", evidence_items(evidence, "T", _excerpt_text), trim_rank=1),
        *evidence_sections(evidence),
    ]
    assembled = assemble(sections, input_budget(profile, QA_BUDGET), profile, fixed_chars=len(system) + len(ask))
    report(f"{profile.model} 正在回答…")
    result = create_provider(profile).generate_structured(
        [TextPart(assembled.text), TextPart(ask)], QAAnswer, system=system, effort=ctx.cfg.effort_for_step("qa"),
    )
    ctx.library.log_call(None, "qa", profile, result.usage, course_id=ctx.course.id)
    return result.output, profile.model, trimmed_note(assembled.trimmed)


def _kp_line(kp: KnowledgePoint) -> str:
    seqs = "、".join(str(seq) for seq in sorted({a.seq for a in kp.appearances}))
    chapter = f"｜{kp.chapter}" if kp.chapter else ""
    return f"[K{kp.key}] {kp.name}{chapter}｜" + (f"第 {seqs} 讲" if seqs else "课上没讲到（只在资料里）")


def _note_text(view: dict[str, Any]) -> str:
    return f"{view['label']}「{view['name']}」：{snippet(view['summary_md'], 600)}"


def _excerpt_text(view: dict[str, Any]) -> str:
    return f"{view['label']}：{view['text']}"


def _prompt(ctx: QAContext, name: str) -> str:
    settings = ctx.course.settings
    return load_prompt(name, subject=settings.subject or ctx.course.name, teaching=settings.teaching_description,
                       note_language=settings.note_language_name, term_rule=settings.term_rule,
                       instructions=instructions_block(ctx.course.instructions),
                       exam_plan=plan_block(ExamPlanStore(ctx.library).active(ctx.course), "qa"))
