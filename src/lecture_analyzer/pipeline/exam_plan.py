"""考试方案的分析与关卡：先弄清这门课怎么考，再分析知识点。

- 来源：用户提供的考试说明（最可信）、考试通知 / 复习提纲资料的条目、往年试题的题型统计（只作参考），
  以及来源讲（默认讲次最小的一讲）转写里和考试有关的部分——开头若干分钟，加上提到考试、题型、分值等
  关键词的句子前后各若干分钟。按预算组装，长讲也只调用一次模型。
- 模型只整理、不计算：题型的合计分值和占比由代码算（exam_plans.ExamPlan.computed）；课堂出处的原话
  与转写核对（与考点信号同一套规则），核对不上的保留并标出，由用户确认时判断。
- 关卡（ensure_exam_plan）：一讲要整理笔记时，考试方案必须已确认或已跳过。还没有方案时先分析，
  然后停下等用户在课程页确认（配置 auto_confirm 时直接采用）；停下不算失败，确认后接着处理。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..asr.base import Segment
from ..config import AppConfig
from ..exam_plans import (
    MAX_ITEMS, MAX_TEXT, MAX_TYPES, ExamPlan, ExamPlanStore, PlanComponent, PlanEvidence, PlanQuestionType,
    PlanRule, PlanScope, normalize_type,
)
from ..library import Course, Lecture, Library, LibraryError
from ..llm import TextPart, Usage, create_provider
from ..materials import MaterialStore
from ..prompts import load_prompt
from ..schemas import ExamPlanOutput
from ..textnorm import normalize_zh
from .extract import instructions_block, transcript_lines
from .memory import Item, Section, assemble, input_budget, snippet, trimmed_note
from .validate import TranscriptIndex, check_quote

Report = Callable[[str], None]
PLAN_MATERIAL_KINDS = ("notice", "review_outline")
# 和考试安排有关的说法（「重点」「题目」课上太常见，不算）
KEYWORDS = re.compile(
    r"考试|考核|考查|期末|期中|题型|分值|满分|总分|卷面|闭卷|开卷|平时成绩|平时分|总评|及格|出题|考题|试卷|"
    r"名词解释|简答|论述|病例分析|案例分析|选择题|填空题|判断题|计算题|答题|得分|扣分|不考|要考|必考"
)
ITEM_CHARS = 400  # 资料条目每条最多带多少字


class AwaitingExamPlan(LibraryError):
    """考试方案还没确认：停在整理笔记之前（不算失败），用户在课程页确认后接着处理。"""


@dataclass
class PlanAnalysis:
    course: Course
    plan: ExamPlan
    usage: Usage
    model: str
    summary: str


# ---------- 来源 ----------


def default_source(library: Library, course: Course, lecture_id: int | None = None) -> Lecture | None:
    """分析用的来源讲：指定的讲，否则讲次最小的一讲（开课第一讲通常会交代怎么考）。"""
    lectures = library.lectures(course)
    if lecture_id is not None:
        chosen = next((lecture for lecture in lectures if lecture.id == lecture_id), None)
        if chosen is not None:
            return chosen
    return lectures[0] if lectures else None


def transcript_excerpt(segments: Sequence[Segment], lead_minutes: float, window_minutes: float) -> list[Item]:
    """转写里和考试有关的部分：开头 lead_minutes 分钟一段，再加上关键词句子前后 window_minutes 分钟的各段。

    相邻、重叠的段合并；每段是一条可裁剪的输入，按相关程度（关键词个数，开头一段最优先）决定超出预算时先去掉谁。
    """
    if not segments:
        return []
    lead_ms, window_ms = lead_minutes * 60_000, window_minutes * 60_000
    origin = segments[0].start_ms
    hits = [i for i, seg in enumerate(segments) if KEYWORDS.search(seg.text)]
    chosen = {i for i, seg in enumerate(segments) if seg.start_ms - origin < lead_ms}
    for i in hits:
        center = segments[i].start_ms
        chosen.update(j for j, seg in enumerate(segments) if abs(seg.start_ms - center) <= window_ms)
    lines = transcript_lines(segments)
    blocks: list[list[int]] = []
    for i in sorted(chosen):
        if blocks and i == blocks[-1][-1] + 1:
            blocks[-1].append(i)
        else:
            blocks.append([i])
    hit_set = set(hits)
    items = []
    for block in blocks:
        text = "\n".join(lines[i] for i in block)
        relevance = sum(i in hit_set for i in block)
        keep = 1_000 + relevance if block[0] == 0 else relevance  # 开头一段最后才裁
        items.append(Item(text + "\n……", "", keep=keep))
    return items


def material_items(library: Library, course: Course) -> tuple[list[str], list[str]]:
    """考试通知、复习提纲的条目（已拆成条目的资料）；返回（条目文字，还没拆成条目的资料名）。"""
    store = MaterialStore(library)
    lines: list[str] = []
    pending: list[str] = []
    for material in store.list(course, PLAN_MATERIAL_KINDS):
        items = store.items(material.id)
        if not items:
            pending.append(material.title)
            continue
        for item in items:
            label = item.label or f"第 {item.idx} 条"
            score = f"（{item.score:g} 分）" if item.score else ""
            lines.append(f"「{material.title}」{label}{score}：{snippet(item.content_md, ITEM_CHARS)}")
    return lines, pending


def type_stat_lines(library: Library, course: Course) -> list[str]:
    from .report_input import type_stats  # 报告模块依赖较多，用到时再导入

    return [f"{s['type']}：共 {s['count']} 题" + (f"，合计 {s['score']:g} 分" if s["score"] else "")
            for s in type_stats(library, course, None)]


# ---------- 分析 ----------


def build_system_prompt(course: Course) -> str:
    settings = course.settings
    return load_prompt("exam_plan", subject=settings.subject, teaching=settings.teaching_description,
                       note_language=settings.note_language_name,
                       instructions=instructions_block(course.instructions))


def analyze_exam_plan(
    cfg: AppConfig, library: Library, course: Course, lecture_id: int | None = None, user_text: str | None = None,
    report: Report = print,
) -> PlanAnalysis:
    """分析考试方案并保存为待确认（草稿）。lecture_id、user_text 不给时沿用上次保存的。"""
    store = ExamPlanStore(library)
    stored = store.get(course)
    text = user_text if user_text is not None else (stored.user_text if stored else "")
    source = default_source(library, course, lecture_id or (stored.source_lecture_id if stored else None))
    store.save_inputs(course, source.id if source else None, text)
    segments = library.segments(source.id) if source else []
    materials, pending = material_items(library, course)
    stats = type_stat_lines(library, course)
    if not (text.strip() or segments or materials or stats):
        where = f"第 {source.seq} 讲还没有转写，" if source else "还没有添加任何一讲，"
        raise AwaitingExamPlan(f"「{course.name}」还没有可以分析考试方案的内容：{where}可以先处理这一讲，"
                               "或在课程页填写老师的考试说明，也可以跳过考试方案")

    profile = cfg.profile_for_step("exam_plan")
    system = build_system_prompt(course)
    excerpt = transcript_excerpt(segments, cfg.exam_plan.lead_minutes, cfg.exam_plan.window_minutes)
    sections = [
        Section("用户提供的考试说明", [Item(text.strip())] if text.strip() else [], empty_text="（没有提供）"),
        Section("考试通知、复习提纲的条目", [Item(line, "", keep=1) for line in materials], trim_rank=2,
                empty_text="（没有）"),
        Section("往年真题、模拟题、习题的题型统计（只作参考）", [Item(line) for line in stats], empty_text="（没有）"),
        Section(f"课堂转写摘录（第 {source.seq} 讲）" if source and segments else "课堂转写摘录", excerpt,
                trim_rank=1, empty_text="（没有）"),
    ]
    assembled = assemble(sections, input_budget(profile, cfg.exam_plan.budget), profile, fixed_chars=len(system))
    report(f"分析考试方案（{profile.model}）：" + "、".join(
        name for name, present in (("考试说明", text.strip()), ("资料条目", materials), ("往年题型", stats),
                                   (f"第 {source.seq} 讲的转写" if source else "", segments)) if present))
    result = create_provider(profile).generate_structured(
        [TextPart(assembled.text), TextPart("请整理这门课的考试方案。")], ExamPlanOutput, system=system,
        effort=cfg.effort_for_step("exam_plan"),
    )
    library.log_call(None, "exam_plan", profile, result.usage, course_id=course.id)
    plan = to_plan(result.output, source, segments).computed()
    course = store.save_draft(course, plan, profile.model)
    unverified = sum(not e.verified for e in plan_evidences(plan))
    notes = [f"题型 {len(plan.question_types)} 种、答题要求 {len(plan.answer_rules)} 条、考试范围 {len(plan.scope)} 条"]
    if unverified:
        notes.append(f"{unverified} 条课堂原话没能在转写里核对上，请确认时留意")
    if plan.is_empty:
        notes.append("来源里没有找到考试安排的说法，可以在课程页填写或跳过")
    if pending:
        notes.append("这些资料还没拆成条目，没有参与分析：" + "、".join(pending))
    if assembled.trimmed:
        notes.append(trimmed_note(assembled.trimmed))
    return PlanAnalysis(course, plan, result.usage, profile.model, "；".join(notes))


def to_plan(output: ExamPlanOutput, source: Lecture | None, segments: Sequence[Segment]) -> ExamPlan:
    """模型输出 → 方案：截断过长的内容，课堂出处的原话与转写核对（核对上时出处改成原话实际所在的句子）。"""
    index = TranscriptIndex(segments) if segments else None

    def evidence(kind: str, quote: str, segment_ids: Sequence[int]) -> PlanEvidence:
        quote = quote.strip()[:MAX_TEXT]
        if kind != "lecture":
            return PlanEvidence(source=kind, quote=quote)  # type: ignore[arg-type]
        if index is None or source is None:
            return PlanEvidence(source="lecture", quote=quote, verified=False)
        method, ids = check_quote(quote, segment_ids, index)
        if method in ("window", "fuzzy"):  # 编号可能偏了一两句：能逐字找到时，跳转录音用原话实际所在的句子
            ids = index.locate(normalize_zh(quote)) or ids
        valid_ids = [i for i in ids if 1 <= i <= len(segments)]
        start = segments[valid_ids[0] - 1].start_ms if valid_ids else None
        return PlanEvidence(source="lecture", quote=quote, lecture_id=source.id, seq=source.seq,
                            segment_ids=valid_ids, t=start, verified=bool(method))

    types: list[PlanQuestionType] = []
    for item in output.question_types:
        name = item.type.strip()[:40]
        if not name or any(t.name == name for t in types):
            continue
        types.append(PlanQuestionType(
            name=name, key=normalize_type(name), count=max(item.count, 0), points_each=max(item.points_each, 0),
            total_points=max(item.total_points, 0), note=item.note.strip()[:MAX_TEXT],
            evidence=evidence(item.source, item.quote, item.segment_ids)))
    return ExamPlan(
        exam_form=output.exam_form.strip()[:60],
        duration_minutes=max(output.duration_minutes, 0),
        total_points=max(output.total_points, 0),
        components=[PlanComponent(name=c.name.strip()[:40], percent=max(c.percent, 0),
                                  evidence=evidence(c.source, c.quote, c.segment_ids))
                    for c in output.components if c.name.strip()][:MAX_ITEMS],
        question_types=types[:MAX_TYPES],
        answer_rules=[PlanRule(text=r.text.strip()[:MAX_TEXT], question_type=r.question_type.strip()[:40],
                               evidence=evidence(r.source, r.quote, r.segment_ids))
                      for r in output.answer_rules if r.text.strip()][:MAX_ITEMS],
        scope=[PlanScope(kind=s.kind, text=s.text.strip()[:MAX_TEXT], evidence=evidence(s.source, s.quote, s.segment_ids))
               for s in output.scope if s.text.strip()][:MAX_ITEMS],
        not_mentioned=[x.strip()[:20] for x in output.not_mentioned if x.strip()][:10],
    )


def plan_evidences(plan: ExamPlan) -> list[PlanEvidence]:
    return [x.evidence for group in (plan.components, plan.question_types, plan.answer_rules, plan.scope)
            for x in group]


# ---------- 关卡 ----------


def ensure_exam_plan(cfg: AppConfig, library: Library, course: Course, report: Report = print) -> Course:
    """整理笔记前：方案已确认或已跳过时放行；还没有方案时先分析；待确认时停下（auto_confirm 时直接采用）。"""
    course = library.course_by_id(course.id)
    if course.exam_plan in ("confirmed", "skipped"):
        return course
    store = ExamPlanStore(library)
    if course.exam_plan == "unset":
        stored = store.get(course)
        source = default_source(library, course, stored.source_lecture_id if stored else None)
        has_text = bool(stored and stored.user_text.strip())
        if source is not None and not library.segments(source.id) and not has_text:
            raise AwaitingExamPlan(
                f"整理笔记前要先确定「{course.name}」的考试方案（怎么考、题型分值、答题要求），它要从第 {source.seq} 讲"
                f"分析，但第 {source.seq} 讲还没有转写：请先处理第 {source.seq} 讲，或在课程页填写老师的考试说明、"
                "换一讲作为来源，也可以跳过考试方案")
        analysis = analyze_exam_plan(cfg, library, course, report=report)
        report(f"考试方案：{analysis.summary}")
        course = analysis.course
    if cfg.exam_plan.auto_confirm:
        course, _ = store.confirm(course)
        report("已按配置自动采用考试方案（auto_confirm）")
        return course
    raise AwaitingExamPlan(f"已分析出「{course.name}」的考试方案，请到课程页核对、修改并确认，确认后接着整理笔记"
                           "（之后的知识点分析会以它为依据）")
