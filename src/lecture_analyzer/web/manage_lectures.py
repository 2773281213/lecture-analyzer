"""网页里管理讲：添加一讲（可附一份固定对应的课件、添加后立即处理）、修改标题日期和类型、追加一节、
处理或从某一步重做、手动指定课件页、删除一讲。耗时的操作都提交成后台任务。"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from ..books import export_course
from ..jobs import ensure_idle
from ..library import LECTURE_KINDS
from ..materials import MaterialStore
from ..pipeline.organize import delete_lecture, next_free_seq
from ..pipeline.slides import parse_page_spec
from .forms import (
    LONG_FORM_BYTES, Form, enqueue, flash, gate_url, int_field, library_of, path_list, read_form, redirect,
    run_params, submit_job, text_field,
)

router = APIRouter()
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PAGES = re.compile(r"^\d+(-\d+)?$")


def _lecture_fields(form: Form) -> dict[str, Any]:
    kind = form.get("kind", "normal")
    if kind not in LECTURE_KINDS:
        raise HTTPException(400, "讲的类型不正确")
    date = form.get("date", "").strip()
    if date and not _DATE.match(date):
        raise HTTPException(400, "日期要写成 2026-10-16 这样")
    return {"title": text_field(form, "title", "标题"), "date": date, "kind": kind}


@router.post("/courses/{course_id}/lectures")
async def add_lecture(request: Request, course_id: int) -> RedirectResponse:
    form = await read_form(request, LONG_FORM_BYTES)
    media = path_list(form.get("paths", ""), "录音或录像文件", files_only=True)
    slides = path_list(form["slides"], "课件", files_only=True) if form.get("slides", "").strip() else []
    fields = _lecture_fields(form)
    with library_of(request) as library:
        course = library.course_by_id(course_id)
        seq = int_field(form, "seq", "讲次", required=False) or next_free_seq(library, course)
        lecture = library.add_lecture(course, seq, media, **fields)
        if slides:  # 导入后固定对应到这一讲（后台任务，导入完才开始处理）
            enqueue(request, library, course, "import_material",
                    {"paths": [str(slides[0])], "kind": "slides", "title": "", "process": True,
                     "pin_lecture": lecture.id})
        if form.get("run") == "1":
            params = {"lectures": [lecture.id], "force": []}
            if course.slides == "unset" and not slides:
                return redirect(gate_url(f"/lectures/{lecture.id}", params))
            enqueue(request, library, course, "run", params)
    return redirect(f"/lectures/{lecture.id}")


@router.post("/lectures/{lecture_id}/edit")
async def edit_lecture(request: Request, lecture_id: int) -> RedirectResponse:
    form = await read_form(request)
    fields = _lecture_fields(form)
    with library_of(request) as library:
        lecture = library.lecture_by_id(lecture_id)
        library.update_lecture(lecture_id, **fields)
        if lecture.step_state("book") == "done":  # 已上架的讲重新导出，标题等跟着更新
            export_course(library, request.app.state.cfg.data_dir, library.course_by_id(lecture.course_id))
    return redirect(f"/lectures/{lecture_id}#manage")


@router.post("/lectures/{lecture_id}/parts")
async def add_parts(request: Request, lecture_id: int) -> RedirectResponse:
    form = await read_form(request, LONG_FORM_BYTES)
    media = path_list(form.get("paths", ""), "录音或录像文件", files_only=True)
    with library_of(request) as library:
        lecture = library.lecture_by_id(lecture_id)
        ensure_idle(library, library.course_by_id(lecture.course_id), "追加节")
        for path in media:
            lecture = library.add_part(lecture_id, path)
    flash(request, f"已追加 {len(media)} 节，现在共 {len(lecture.parts)} 节。点「处理」重新处理这一讲（已转写的节不会重新转写）")
    return redirect(f"/lectures/{lecture_id}#manage")


@router.post("/lectures/{lecture_id}/run")
async def run_lecture(request: Request, lecture_id: int) -> RedirectResponse:
    """处理这一讲；force 为从哪一步起重做。课程还没选择是否提供课件时先弹出选择框。"""
    form = await read_form(request)
    with library_of(request) as library:
        lecture = library.lecture_by_id(lecture_id)
        course = library.course_by_id(lecture.course_id)
        params = {**run_params(form, library, course), "lectures": [lecture_id]}
        if course.slides == "unset":
            return redirect(gate_url(f"/lectures/{lecture_id}", params))
        return submit_job(request, library, course, "run", params)


@router.post("/lectures/{lecture_id}/pages")
async def set_pages(request: Request, lecture_id: int) -> RedirectResponse:
    """手动指定这一讲对应的课件页（一份资料的某个页码区间），或恢复自动匹配。"""
    form = await read_form(request)
    with library_of(request) as library:
        lecture = library.lecture_by_id(lecture_id)
        store = MaterialStore(library)
        if form.get("mode") == "auto":
            store.set_lecture_pages(lecture_id, [], "auto")
        else:
            material_id = int_field(form, "material", "资料")
            pages = form.get("pages", "").strip().replace(" ", "")
            if pages and not _PAGES.match(pages):
                raise HTTPException(400, "页码要写成 3-30 这样（不填表示整份资料）")
            materials = store.list(library.course_by_id(lecture.course_id))
            try:
                refs = parse_page_spec(f"{material_id}:{pages}" if pages else str(material_id), materials)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from None
            store.set_lecture_pages(lecture_id, refs, "manual")
        library.reset_steps(lecture_id, ["slides"])
    flash(request, "已修改对应的课件页：点「处理」按新的课件页重新整理笔记（课件页没变时不会重新整理）")
    return redirect(f"/lectures/{lecture_id}#manage")


@router.post("/lectures/{lecture_id}/delete")
async def remove_lecture(request: Request, lecture_id: int) -> RedirectResponse:
    form = await read_form(request)
    if form.get("confirm") != "1":
        raise HTTPException(400, "请先勾选确认，再删除")
    with library_of(request) as library:
        lecture = library.lecture_by_id(lecture_id)
        stale = delete_lecture(library, request.app.state.cfg.data_dir, lecture)
    note = ("；这些讲的课程记忆待更新：" + "、".join(f"第 {x.seq} 讲" for x in stale) + "（在课程页点「重建课程记忆」）"
            if stale else "")
    flash(request, f"已删除第 {lecture.seq} 讲（原始音视频没有删除）{note}")
    return redirect(f"/courses/{lecture.course_id}")
