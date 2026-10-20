"""网页书架的小型写操作：给讲加删标签、把讲移到别的课程、调整课程在书架上的顺序、记录知识点的复习进度，
提交考前报告和总复习任务，取消任务。

都用普通表单 POST，处理完重定向回页面（303）。表单读取与校验、任务提交见 forms.py；
后台任务的参数先完整校验（课程、讲次、资料、模型都存在）再入队。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from ..books import remove_report
from ..jobs import JobStore
from ..library import LibraryError
from ..pipeline.organize import move_lecture
from ..pipeline.report import build_plan
from ..reports import ReportStore
from ..textnorm import normalize_zh
from .forms import MAX_NAME_CHARS, flash, int_field, library_of, numbers, read_form, redirect, submit_job

router = APIRouter()
REPORT_FORM_BYTES = 16 * 1024  # 考前分析的表单有很多复选框
REPORT_FORM_FIELDS = 400


@router.post("/lectures/{lecture_id}/tags")
async def add_tag(request: Request, lecture_id: int) -> RedirectResponse:
    form = await read_form(request)
    with library_of(request) as library:
        library.add_tag(lecture_id, form.get("tag", ""))
    return redirect(f"/lectures/{lecture_id}#organize")


@router.post("/lectures/{lecture_id}/tags/remove")
async def remove_tag(request: Request, lecture_id: int) -> RedirectResponse:
    form = await read_form(request)
    with library_of(request) as library:
        library.lecture_by_id(lecture_id)
        library.remove_tag(lecture_id, form.get("tag", ""))
    return redirect(f"/lectures/{lecture_id}#organize")


@router.post("/lectures/{lecture_id}/move")
async def move(request: Request, lecture_id: int) -> RedirectResponse:
    form = await read_form(request)
    course_id = int_field(form, "course_id", "课程编号")
    seq = int_field(form, "seq", "讲次", required=False)
    with library_of(request) as library:
        lecture = library.lecture_by_id(lecture_id)
        move_lecture(library, request.app.state.cfg.data_dir, lecture, library.course_by_id(course_id), seq)
    return redirect(f"/lectures/{lecture_id}")


@router.post("/courses/{course_id}/position")
async def position(request: Request, course_id: int) -> RedirectResponse:
    form = await read_form(request)
    direction = form.get("direction")
    if direction not in ("up", "down"):
        raise HTTPException(400, "只能上移或下移")
    with library_of(request) as library:
        library.move_course(course_id, -1 if direction == "up" else 1)
    return redirect("/?sort=custom")


@router.post("/courses/{course_id}/reviewed")
async def reviewed(request: Request, course_id: int) -> RedirectResponse:
    form = await read_form(request)
    name = normalize_zh(form.get("name", ""))
    if not name or len(name) > MAX_NAME_CHARS:
        raise HTTPException(400, "知识点名称不正确")
    key = int_field(form, "key", "知识点编号")
    with library_of(request) as library:
        library.course_by_id(course_id)
        library.set_reviewed(course_id, name, form.get("reviewed") == "1")
    page = f"/courses/{course_id}/review" if form.get("back") == "review" else f"/courses/{course_id}"
    return redirect(f"{page}#k-{key}")


# ---------- 考前报告、总复习、取消任务 ----------


@router.post("/courses/{course_id}/report")
async def submit_report(request: Request, course_id: int) -> RedirectResponse:
    form = await read_form(request, REPORT_FORM_BYTES, REPORT_FORM_FIELDS)
    params = {
        "lectures": numbers(form.getlist("lecture"), "讲次"),
        "tag": form.get("tag", "").strip()[:MAX_NAME_CHARS],
        "materials": numbers(form.getlist("material"), "资料编号"),
        "models": [name.strip() for name in form.getlist("model") if name.strip()][:8],
    }
    cfg = request.app.state.cfg
    with library_of(request) as library:
        course = library.course_by_id(course_id)
        try:  # 入队前先完整校验，参数不对时直接提示，而不是等任务失败
            build_plan(cfg, library, course, params["lectures"], params["tag"], params["materials"], params["models"])
        except ValueError as exc:  # ConfigError：模型档案不存在或能力不够
            raise LibraryError(str(exc)) from None
        return submit_job(request, library, course, "report", params)


@router.post("/reports/{report_id}/delete")
async def delete_report(request: Request, report_id: int) -> RedirectResponse:
    await read_form(request)
    with library_of(request) as library:
        report = ReportStore(library).get(report_id)
        course = library.course_by_id(report.course_id)
        remove_report(library, request.app.state.cfg.data_dir, course, report)
    flash(request, f"已删除「{report.title}」")
    return redirect(f"/courses/{course.id}#analysis")


@router.post("/courses/{course_id}/review/generate")
async def submit_review(request: Request, course_id: int) -> RedirectResponse:
    form = await read_form(request)
    with library_of(request) as library:
        course = library.course_by_id(course_id)
        return submit_job(request, library, course, "review", {"all": form.get("all") == "1"})


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: int) -> RedirectResponse:
    await read_form(request)
    with library_of(request) as library:
        JobStore(library).get(job_id)  # 没有这个任务时提示
    request.app.state.jobs.cancel(job_id)  # 已经结束的任务不受影响
    return redirect(f"/jobs/{job_id}")
