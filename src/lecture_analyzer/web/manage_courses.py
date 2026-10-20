"""网页里管理课程和资料：新建课程、修改设置、删除课程、课件关卡（选择是否提供课件）、处理与重建课程记忆、
导入和删除资料。耗时的操作都提交成后台任务。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ..asr import ASR_BACKENDS, language_problem
from ..config import AppConfig
from ..course import LANGUAGE_NAMES, CourseSettings, SettingsError, build_settings
from ..jobs import ensure_idle
from ..library import Course
from ..materials import MATERIAL_KINDS, MaterialStore
from ..pipeline.ingest import remove_material
from ..pipeline.organize import delete_course, rerun_step_for_settings
from ..pipeline.run import STEP_LABELS
from .forms import (
    LONG_FORM_BYTES, Form, enqueue, flash, gate_url, int_field, library_of, path_list, read_form, redirect,
    run_params, submit_job, text_field,
)
from .views import spine_color

router = APIRouter()
MAX_COURSE_NAME = 60
SETTINGS_KEYS = ("subject", "teaching", "interpreted_to")
COMMON_LANGUAGES = ("zh", "ko", "en", "ja")  # 课程表单里直接勾选的语言；其他语言填代码
LOCAL_ASR = {
    "funasr:paraformer": "本地 Paraformer（只支持中文，支持热词）",
    "funasr:sensevoice": "本地 SenseVoice（中、英、日、韩、粤）",
    "qwen3:1.7b": "本地 Qwen3-ASR 1.7B（多语种，较准）",
    "qwen3:0.6b": "本地 Qwen3-ASR 0.6B（多语种，较快）",
}


def asr_choices(cfg: AppConfig) -> dict[str, str]:
    """课程可选的转写方案：空表示用设置页的默认方案。"""
    choices = {"": f"默认（设置页的方案：{ASR_BACKENDS.get(cfg.asr.backend, cfg.asr.backend)}）",
               "auto": ASR_BACKENDS["auto"]}
    choices.update({f"llm:{name}": f"大模型转写：{name}（{profile.model or '未填型号'}）"
                    for name, profile in cfg.models.items() if profile.audio})
    choices.update(LOCAL_ASR)
    return choices


def _form_context(request: Request, course: Course | None, **extra: Any) -> dict[str, Any]:
    settings = course.settings if course else CourseSettings()
    return {"course": course, "asr_choices": asr_choices(request.app.state.cfg), "LANGUAGE_NAMES": LANGUAGE_NAMES,
            "COMMON_LANGUAGES": COMMON_LANGUAGES, "defaults": settings,
            "teaching_other": ",".join(code for code in settings.languages if code not in COMMON_LANGUAGES),
            "current_asr": (course.asr or "") if course else "",
            "spine": spine_color(course) if course else "var(--accent)", **extra}


def _settings_fields(form: Form, cfg: AppConfig) -> dict[str, Any]:
    """表单里的课程设置：勾选的授课语言加上填写的其他语言代码；取值由 build_settings 校验，
    转写方案不支持课程的语言时拦下。"""
    asr = form.get("asr", "")
    if asr not in asr_choices(cfg):
        raise HTTPException(400, f"未知的转写方案：{asr}")
    other = text_field(form, "teaching_other", "其他授课语言", 60)
    fields = {"subject": text_field(form, "subject", "学科", 40) or "医学",
              "teaching": ",".join([*form.getlist("teaching"), other]),
              "interpreted_to": text_field(form, "interpreted_to", "现场翻译成的语言", 20)}
    try:
        settings = build_settings(**fields)
    except SettingsError as exc:
        raise HTTPException(400, f"课程设置不正确：{exc}") from None
    problem = language_problem(cfg, asr or None, settings)
    if problem:
        raise HTTPException(400, problem)
    return {**fields, "asr": asr or None, "instructions": form.get("instructions", "")}


@router.get("/courses/new", response_class=HTMLResponse)
def new_course(request: Request) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(request, "course_form.html", _form_context(request, None))


@router.post("/courses")
async def create_course(request: Request) -> RedirectResponse:
    form = await read_form(request, LONG_FORM_BYTES)
    name = text_field(form, "name", "课程名", MAX_COURSE_NAME)
    if not name:
        raise HTTPException(400, "请填写课程名")
    fields = _settings_fields(form, request.app.state.cfg)
    with library_of(request) as library:
        settings = build_settings(**{key: fields[key] for key in SETTINGS_KEYS})
        course = library.add_course(name, settings, fields["asr"], fields["instructions"])
        if form.get("slides") == "none":
            library.update_course(course.id, slides="none")
    return redirect(f"/courses/{course.id}")


@router.get("/courses/{course_id}/settings", response_class=HTMLResponse)
def course_settings(request: Request, course_id: int, saved: str = "") -> HTMLResponse:
    with library_of(request) as library:
        course = library.course_by_id(course_id)
    rerun = saved if saved in STEP_LABELS else ""
    context = _form_context(request, course, saved=bool(saved), rerun=rerun,
                            rerun_label=STEP_LABELS.get(rerun, ""))
    return request.app.state.templates.TemplateResponse(request, "course_form.html", context)


@router.post("/courses/{course_id}/settings")
async def save_course_settings(request: Request, course_id: int) -> RedirectResponse:
    form = await read_form(request, LONG_FORM_BYTES)
    fields = _settings_fields(form, request.app.state.cfg)
    with library_of(request) as library:
        old = library.course_by_id(course_id)
        new = library.update_course(course_id, **fields)
        processed = any(lecture.status for lecture in library.lectures(new))
    step = rerun_step_for_settings(old, new) if processed else None
    return redirect(f"/courses/{course_id}/settings?saved={step or 'none'}")


@router.post("/courses/{course_id}/delete")
async def remove_course(request: Request, course_id: int) -> RedirectResponse:
    form = await read_form(request)
    with library_of(request) as library:
        course = library.course_by_id(course_id)
        if form.get("confirm", "").strip() != course.name:
            raise HTTPException(400, "请输入完整的课程名来确认删除")
        delete_course(library, request.app.state.cfg.data_dir, course)
    flash(request, f"已删除课程「{course.name}」（原始音视频没有删除）")
    return redirect("/")


# ---------- 课件关卡、处理、重建 ----------


@router.post("/courses/{course_id}/slides")
async def no_slides(request: Request, course_id: int) -> RedirectResponse:
    """课件关卡：声明这门课不提供课件；then_run 时接着提交原本要做的处理。"""
    form = await read_form(request)
    with library_of(request) as library:
        course = library.update_course(course_id, slides="none")
        if form.get("then_run") == "1":
            return submit_job(request, library, course, "run", run_params(form, library, course))
    return redirect(f"/courses/{course_id}")


@router.post("/courses/{course_id}/run")
async def run_course(request: Request, course_id: int) -> RedirectResponse:
    form = await read_form(request)
    with library_of(request) as library:
        course = library.course_by_id(course_id)
        params = run_params(form, library, course)
        if course.slides == "unset":
            return redirect(gate_url(f"/courses/{course_id}", params))
        return submit_job(request, library, course, "run", params)


@router.post("/courses/{course_id}/rebuild")
async def rebuild_course(request: Request, course_id: int) -> RedirectResponse:
    form = await read_form(request)
    with library_of(request) as library:
        course = library.course_by_id(course_id)
        return submit_job(request, library, course, "rebuild",
                          {"from_seq": int_field(form, "from_seq", "讲次", required=False)})


# ---------- 资料 ----------


@router.post("/courses/{course_id}/materials")
async def import_material(request: Request, course_id: int) -> RedirectResponse:
    """导入资料（后台任务）；从课件关卡来的（then_run）导入后接着处理原本要处理的讲。"""
    form = await read_form(request, LONG_FORM_BYTES)
    kind = form.get("kind", "")
    if kind not in MATERIAL_KINDS:
        raise HTTPException(400, "请选择资料类型")
    params = {"paths": [str(path) for path in path_list(form.get("paths", ""), "资料")], "kind": kind,
              "title": text_field(form, "title", "标题"), "process": form.get("process") == "1"}
    with library_of(request) as library:
        course = library.course_by_id(course_id)
        job_id = enqueue(request, library, course, "import_material", params)
        if form.get("then_run") == "1":  # 同一条队列，导入完才开始处理
            enqueue(request, library, course, "run", run_params(form, library, course))
            return redirect("/jobs")
    return redirect(f"/jobs/{job_id}")


@router.post("/courses/{course_id}/materials/process")
async def process_materials(request: Request, course_id: int) -> RedirectResponse:
    await read_form(request)
    with library_of(request) as library:
        course = library.course_by_id(course_id)
        return submit_job(request, library, course, "process_material", {"materials": []})


@router.post("/materials/{material_id}/delete")
async def delete_material(request: Request, material_id: int) -> RedirectResponse:
    await read_form(request)
    with library_of(request) as library:
        material = MaterialStore(library).get(material_id)
        course = library.course_by_id(material.course_id)
        ensure_idle(library, course, "删除资料")
        affected = MaterialStore(library).lectures_using(material_id)
        remove_material(library, request.app.state.cfg.data_dir, material_id)
    note = f"；{len(affected)} 讲用到了这份资料，下次处理时会重新匹配课件" if affected else ""
    flash(request, f"已删除资料「{material.title}」{note}")
    return redirect(f"/courses/{course.id}#materials")
