"""课程页的「考试方案」：分析（后台任务）、表格修改、确认、跳过。

- 分析前先保存来源讲和用户提供的考试说明，再提交「分析考试方案」任务；结果是待确认的草稿；
- 表格每一行带着原来那一条的编号（ref）：内容没改的行保留原来的出处（课堂原话、资料），改过或新增的行记为
  用户填写；清空的行删除；
- 确认：方案内容和上次确认的不同、且已有讲整理过笔记时，自动从「整理笔记」起重新处理全部讲；
  第一次确认（或内容没变）时只接着处理还没完成的讲。跳过同样接着处理还没完成的讲。
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from ..exam_plans import (
    MAX_ITEMS, MAX_TEXT, MAX_TYPES, QUESTION_TYPES, SCOPE_LABELS, ExamPlan, ExamPlanStore, PlanComponent,
    PlanEvidence, PlanQuestionType, PlanRule, PlanScope, clean_user_text,
)
from ..library import Course, Library
from .forms import LONG_FORM_BYTES, Form, enqueue, flash, int_field, library_of, read_form, redirect, submit_job

router = APIRouter()
PLAN_FORM_FIELDS = 600  # 题型 12 行 × 7 列，答题要求、范围、成绩构成各 30 行 × 3 列
CATEGORY_FILLED: dict[str, Callable[[ExamPlan], bool]] = {
    "考核方式": lambda p: bool(p.exam_form),
    "考试时长": lambda p: bool(p.duration_minutes),
    "成绩构成": lambda p: bool(p.components),
    "题型": lambda p: bool(p.question_types),
    "分值": lambda p: any(t.total_points for t in p.question_types),
    "答题要求": lambda p: bool(p.answer_rules),
    "考试范围": lambda p: bool(p.scope),
}
T = TypeVar("T")


def _anchor(course_id: int) -> str:
    return f"/courses/{course_id}#exam-plan"


@router.post("/courses/{course_id}/exam-plan/analyze")
async def analyze(request: Request, course_id: int) -> RedirectResponse:
    """保存来源讲和考试说明，提交分析任务。"""
    form = await read_form(request, LONG_FORM_BYTES)
    lecture_id = int_field(form, "lecture", "来源讲", required=False)
    with library_of(request) as library:
        course = library.course_by_id(course_id)
        if lecture_id is not None and library.lecture_by_id(lecture_id).course_id != course.id:
            raise HTTPException(400, "所选的讲不属于这门课")
        ExamPlanStore(library).save_inputs(course, lecture_id, clean_user_text(form.get("user_text", "")))
        return submit_job(request, library, course, "exam_plan", {"lecture": lecture_id})


@router.post("/courses/{course_id}/exam-plan/save")
async def save(request: Request, course_id: int) -> RedirectResponse:
    """保存表格；action=confirm 时同时确认，并按需要接着（重新）处理各讲。"""
    form = await read_form(request, LONG_FORM_BYTES, PLAN_FORM_FIELDS)
    with library_of(request) as library:
        course = library.course_by_id(course_id)
        store = ExamPlanStore(library)
        stored = store.get(course)
        previous = stored.plan if stored and stored.plan else ExamPlan()
        plan = parse_plan(form, previous)
        store.save_edit(course, plan)
        if form.get("action") != "confirm":
            flash(request, "已保存考试方案（还没有确认）")
            return redirect(_anchor(course_id))
        course, changed = store.confirm(course)
        job = _continue(request, library, course, rerun=changed)
    if job is None:
        flash(request, "已确认考试方案" + ("" if changed else "（内容没有变化）"))
        return redirect(_anchor(course_id))
    return redirect(f"/jobs/{job}")


@router.post("/courses/{course_id}/exam-plan/skip")
async def skip(request: Request, course_id: int) -> RedirectResponse:
    """跳过考试方案：接着处理还没完成的讲（已整理的笔记不重做）。"""
    await read_form(request)
    with library_of(request) as library:
        course = ExamPlanStore(library).skip(library.course_by_id(course_id))
        job = _continue(request, library, course, rerun=False)
    if job is None:
        flash(request, "已跳过考试方案：后续分析不使用考试方案，可以随时回来分析或填写")
        return redirect(_anchor(course_id))
    return redirect(f"/jobs/{job}")


def _continue(request: Request, library: Library, course: Course, rerun: bool) -> int | None:
    """确认或跳过之后：有讲整理过笔记且方案变了时，从整理笔记起重新处理全部讲；否则只接着处理没完成的讲。
    没有需要处理的讲时返回 None。"""
    from ..pipeline.run import FINISHED, STEPS

    lectures = library.lectures(course)
    if not lectures or course.slides == "unset":  # 课件关卡还没过：等用户在课程页选择后再处理
        return None
    noted = [lecture for lecture in lectures if library.notes(lecture.id) is not None]
    if rerun and noted:
        return enqueue(request, library, course, "run", {"lectures": [], "force": ["extract"]})
    if all(lecture.step_state(step) in FINISHED for lecture in lectures for step in STEPS):
        return None
    return enqueue(request, library, course, "run", {"lectures": [], "force": []})


# ---------- 表格 → 方案 ----------


def parse_plan(form: Form, previous: ExamPlan) -> ExamPlan:
    """读表格。每行的 ref 是原来那一条的序号：内容没改时保留原来的出处，改过或新增的行记为用户填写。"""
    components = _rows(form, "comp", ("name", "percent"), previous.components, MAX_ITEMS,
                       lambda row, old: _component(row, old))
    types = _rows(form, "qt", ("name", "key", "count", "points", "total", "note"), previous.question_types, MAX_TYPES,
                  lambda row, old: _question_type(row, old))
    rules = _rows(form, "rule", ("text", "type"), previous.answer_rules, MAX_ITEMS, lambda row, old: _rule(row, old))
    scope = _rows(form, "scope", ("kind", "text"), previous.scope, MAX_ITEMS, lambda row, old: _scope(row, old))
    plan = ExamPlan(
        exam_form=_text(form.get("exam_form", ""), 60),
        duration_minutes=int(_number(form.get("duration_minutes", ""), "考试时长")),
        total_points=_number(form.get("total_points", ""), "卷面总分"),
        components=components, question_types=types, answer_rules=rules, scope=scope,
    )
    filled = {name for name, check in CATEGORY_FILLED.items() if check(plan)}
    return plan.model_copy(update={"not_mentioned": [x for x in previous.not_mentioned if x not in filled]})


def _rows(form: Form, prefix: str, columns: tuple[str, ...], previous: list[T], limit: int,
          build: Callable[[dict[str, str], T | None], T | None]) -> list[T]:
    values = {column: form.getlist(f"{prefix}_{column}") for column in columns}
    refs = form.getlist(f"{prefix}_ref")
    count = max((len(v) for v in values.values()), default=0)
    result: list[T] = []
    for i in range(count):
        row = {column: (values[column][i] if i < len(values[column]) else "").strip() for column in columns}
        ref = refs[i] if i < len(refs) else ""
        old = previous[int(ref)] if ref.isdigit() and int(ref) < len(previous) else None
        item = build(row, old)
        if item is not None:
            result.append(item)
    if len(result) > limit:
        raise HTTPException(400, f"最多 {limit} 行")
    return result


def _evidence(old: Any, same: bool) -> PlanEvidence:
    """内容没改时沿用原来的出处，否则记为用户填写。"""
    return old.evidence if old is not None and same else PlanEvidence(source="user")


def _component(row: dict[str, str], old: PlanComponent | None) -> PlanComponent | None:
    name = _text(row["name"], 40)
    if not name:
        return None
    percent = _number(row["percent"], "成绩占比")
    same = old is not None and (old.name, old.percent) == (name, percent)
    return PlanComponent(name=name, percent=percent, evidence=_evidence(old, same))


def _question_type(row: dict[str, str], old: PlanQuestionType | None) -> PlanQuestionType | None:
    name = _text(row["name"], 40)
    key = row["key"] if row["key"] in QUESTION_TYPES else ""
    if not name and key:
        name = QUESTION_TYPES[key]
    if not name:
        return None
    count = int(_number(row["count"], "题数"))
    points = _number(row["points"], "每题分值")
    total = _number(row["total"], "合计分值")
    note = _text(row["note"], MAX_TEXT)
    same = old is not None and (old.name, old.count, old.points_each, old.note) == (name, count, points, note)
    if count > 0 and points > 0:  # 合计由代码算；只给合计时才用填写的合计
        total = 0
    elif same and old is not None and not total:
        total = old.total_points
    return PlanQuestionType(name=name, key=key or "other", count=count, points_each=points, total_points=total,
                            note=note, evidence=_evidence(old, same))


def _rule(row: dict[str, str], old: PlanRule | None) -> PlanRule | None:
    text = _text(row["text"], MAX_TEXT)
    if not text:
        return None
    kind = _text(row["type"], 40)
    same = old is not None and (old.text, old.question_type) == (text, kind)
    return PlanRule(text=text, question_type=kind, evidence=_evidence(old, same))


def _scope(row: dict[str, str], old: PlanScope | None) -> PlanScope | None:
    text = _text(row["text"], MAX_TEXT)
    if not text:
        return None
    kind = row["kind"] if row["kind"] in SCOPE_LABELS else "include"
    same = old is not None and (old.kind, old.text) == (kind, text)
    return PlanScope(kind=kind, text=text, evidence=_evidence(old, same))  # type: ignore[arg-type]


def _text(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) > limit:
        raise HTTPException(400, f"有一项内容超过 {limit} 字，请精简")
    return text


def _number(value: str, label: str) -> float:
    text = value.strip()
    if not text:
        return 0
    try:
        number = float(text)
    except ValueError:
        raise HTTPException(400, f"{label}要填数字（现在是「{text}」）") from None
    if number < 0 or number > 100_000:
        raise HTTPException(400, f"{label}不正确：{text}")
    return number


def plan_context(library: Library, course: Course) -> dict[str, Any]:
    """课程页考试方案卡片要用的数据。"""
    stored = ExamPlanStore(library).get(course)
    lectures = library.lectures(course)
    noted = sum(library.notes(lecture.id) is not None for lecture in lectures)
    source_id = stored.source_lecture_id if stored and stored.source_lecture_id else (
        lectures[0].id if lectures else None)
    plan = stored.plan if stored else None
    return {
        "exam_plan": plan,
        "exam_plan_stored": stored,
        "exam_plan_source": source_id,
        "exam_plan_noted": noted,
        "exam_plan_changed": bool(stored and plan and plan.fingerprint() != stored.confirmed_fingerprint),
        "QUESTION_TYPES": QUESTION_TYPES,
        "SCOPE_LABELS": SCOPE_LABELS,
    }
