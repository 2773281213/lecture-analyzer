"""本地网页书架：书架（全部课程）→ 书（课程目录、知识点体系）→ 章（讲的阅读页，点时间戳播放原音视频）。

还有课程资料、考前报告（可打印、可下载 Markdown）和后台任务的页面。后台任务在子进程里执行（job_runner.py），
随应用启动和关闭。

写操作分在几个模块里：actions.py（标签、移动、报告等）、manage_courses.py 和 manage_lectures.py（课程、讲、资料）、
settings_page.py（设置）、qa_page.py（问答）。

安全边界：
- 只监听本机，并检查 Host 头（TrustedHostMiddleware，防 DNS rebinding 读取转写内容）；
- 写操作要带每次启动生成的令牌（见 forms.py）；
- 媒体文件、资料页面图片只按编号提供，不接受任意路径；
- 模型输出渲染成 HTML 时禁用原始 HTML，公式原样转义后交给浏览器端的 KaTeX（见 rendering.py）。
"""

from __future__ import annotations

import mimetypes
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from markupsafe import Markup
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..books import build_course_book, build_lecture, report_filename
from ..config import AppConfig
from ..job_runner import JobRunner
from ..jobs import ACTIVE_STATES, Job, JobStore
from ..jobs import STATE_LABELS as JOB_STATES
from ..knowledge import CourseMemory
from ..library import EXAM_PLAN_LABELS, SLIDES_LABELS, Course, Lecture, Library, LibraryError
from ..materials import MATERIAL_KINDS, Material, MaterialStore
from ..pipeline.ingest import page_image
from ..pipeline.report_render import evidence_text, render_report_markdown, type_brief
from ..pipeline.run import STEP_LABELS as RUN_STEP_LABELS
from ..pipeline.run import STEPS
from ..render import KIND_LABELS, SIGNAL_LABELS, quote_html, zh_aliases
from ..reports import ReportStore, StoredReport
from ..timecode import format_ms
from . import actions, exam_plan_page, manage_courses, manage_lectures, qa_page, settings_page
from .forms import take_flash
from .rendering import cite_links, evidence_url, kp_url, render_markdown
from .views import (
    SORTS, course_view, job_view, jobs_view, lecture_state, lecture_view, material_view, report_view, review_view,
    shelf_books,
)

__all__ = ["create_app", "render_markdown"]

WEB_DIR = Path(__file__).parent
ALLOWED_HOSTS = ("127.0.0.1", "localhost")
STATE_LABELS = {"done": "完成", "skipped": "跳过", "failed": "失败", "stale": "待更新"}


def create_app(cfg: AppConfig, allowed_hosts: Sequence[str] = ALLOWED_HOSTS) -> FastAPI:
    db_path = cfg.data_dir / "lecture.db"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.jobs.start()  # 成为执行器时，先把上次没执行完的任务标为失败
        yield
        app.state.jobs.shutdown()

    app = FastAPI(title="课程书架", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    app.state.cfg = cfg
    app.state.csrf_token = secrets.token_urlsafe(32)
    app.state.jobs = JobRunner(cfg, db_path)
    app.state.flash = None
    templates = _templates(app.state.csrf_token, db_path)
    app.state.templates = templates
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    for module in (actions, manage_courses, manage_lectures, settings_page, qa_page, exam_plan_page):
        app.include_router(module.router)
    _register_errors(app, templates)
    _register_pages(app, templates, db_path)
    _register_analysis_pages(app, templates, db_path)
    _register_data(app, db_path)
    return app


def _templates(csrf_token: str, db_path: Path) -> Jinja2Templates:
    templates = Jinja2Templates(directory=WEB_DIR / "templates")
    templates.env.filters.update(md=render_markdown, ts=format_ms, evidence=evidence_text, cite=cite_links)

    def active_job_count() -> int:
        """顶栏显示的、还没结束的后台任务数。"""
        with Library(db_path) as library:
            return len(JobStore(library).active_jobs())

    templates.env.globals.update(
        SIGNAL_LABELS=SIGNAL_LABELS, KIND_LABELS=KIND_LABELS, STATE_LABELS=STATE_LABELS, SORTS=SORTS,
        SLIDES_LABELS=SLIDES_LABELS, EXAM_PLAN_LABELS=EXAM_PLAN_LABELS, MATERIAL_KINDS=MATERIAL_KINDS,
        JOB_STATES=JOB_STATES,
        RUN_STEPS=[(step, RUN_STEP_LABELS[step]) for step in STEPS], lecture_state=lecture_state,
        evidence_url=evidence_url, kp_url=kp_url, type_brief=type_brief, take_flash=take_flash, active_job_count=active_job_count,
        csrf_token=csrf_token, static_version=_static_version(),
        quote_html=lambda quote, meaning="": Markup(quote_html(quote, meaning)),  # 内容已转义
    )
    templates.env.filters["zh_aliases"] = zh_aliases
    return templates


def gate_request(gate: int, lectures: list[int], force: str) -> dict[str, Any] | None:
    """页面地址里的课件关卡参数（原本要处理的讲和步骤），用来弹出选择框。"""
    if not gate:
        return None
    return {"lectures": lectures[:200], "force": force if force in (*STEPS, "all") else ""}


def _static_version() -> str:
    """静态文件的版本号（按最近的修改时间），加在地址后面：更新代码后浏览器会重新加载样式和脚本，不用旧缓存。"""
    files = [path for path in (WEB_DIR / "static").iterdir() if path.is_file()]
    return format(max((path.stat().st_mtime_ns for path in files), default=0), "x")


def _register_errors(app: FastAPI, templates: Jinja2Templates) -> None:
    """出错时显示中文说明页，而不是 JSON。"""

    @app.exception_handler(LibraryError)
    async def library_error(request: Request, exc: LibraryError) -> Response:
        return templates.TemplateResponse(request, "error.html", {"message": str(exc)}, status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        return templates.TemplateResponse(request, "error.html", {"message": exc.detail}, status_code=exc.status_code)


def _register_pages(app: FastAPI, templates: Jinja2Templates, db_path: Path) -> None:
    @app.get("/", response_class=HTMLResponse)
    def shelves(request: Request, sort: str = "custom") -> HTMLResponse:
        sort = sort if sort in SORTS else "custom"
        with Library(db_path) as library:
            books = shelf_books(library, sort)
        return templates.TemplateResponse(request, "shelves.html", {"books": books, "sort": sort})

    @app.get("/courses/{course_id}", response_class=HTMLResponse)
    def course_page(request: Request, course_id: int, tag: str | None = None, gate: int = 0,
                    lecture: list[int] = Query(default=[]), force: str = "") -> HTMLResponse:
        with Library(db_path) as library:
            context = course_view(library, _course_or_404(library, course_id), tag, app.state.cfg)
        context["gate"] = gate_request(gate, lecture, force)
        return templates.TemplateResponse(request, "course.html", context)

    @app.get("/courses/{course_id}/review", response_class=HTMLResponse)
    def review_page(request: Request, course_id: int, focus: int = 0) -> HTMLResponse:
        with Library(db_path) as library:
            context = review_view(library, _course_or_404(library, course_id), focus=bool(focus))
        return templates.TemplateResponse(request, "review.html", context)

    @app.get("/lectures/{lecture_id}", response_class=HTMLResponse)
    def lecture_page(request: Request, lecture_id: int, gate: int = 0, lecture: list[int] = Query(default=[]),
                     force: str = "") -> HTMLResponse:
        with Library(db_path) as library:
            context = lecture_view(library, _lecture_or_404(library, lecture_id))
        context["gate"] = gate_request(gate, lecture, force)
        return templates.TemplateResponse(request, "lecture.html", context)

    @app.get("/books/{lecture_id}")
    def old_book_link(lecture_id: int) -> RedirectResponse:
        """M1 的旧地址（一讲曾叫"书"），重定向到讲的阅读页。"""
        return RedirectResponse(f"/lectures/{lecture_id}", status_code=301)


def _register_analysis_pages(app: FastAPI, templates: Jinja2Templates, db_path: Path) -> None:
    """课程资料、考前报告和后台任务的页面。"""

    @app.get("/materials/{material_id}", response_class=HTMLResponse)
    def material_page(request: Request, material_id: int) -> HTMLResponse:
        with Library(db_path) as library:
            context = material_view(library, app.state.cfg.data_dir, _material_or_404(library, material_id))
        return templates.TemplateResponse(request, "material.html", context)

    @app.get("/reports/{report_id}", response_class=HTMLResponse)
    def report_page(request: Request, report_id: int) -> HTMLResponse:
        with Library(db_path) as library:
            context = report_view(library, _report_or_404(library, report_id).id)
        return templates.TemplateResponse(request, "report.html", context)

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(request: Request) -> HTMLResponse:
        with Library(db_path) as library:
            context = jobs_view(library)
        return templates.TemplateResponse(request, "jobs.html", context)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(request: Request, job_id: int) -> HTMLResponse:
        with Library(db_path) as library:
            context = job_view(library, _job_or_404(library, job_id))
        return templates.TemplateResponse(request, "job.html", context)


def _register_data(app: FastAPI, db_path: Path) -> None:
    """媒体文件、资料页面图片、导出文件和 JSON 数据。"""

    @app.get("/materials/{material_id}/pages/{page_no}.png")
    def material_page_image(material_id: int, page_no: int) -> FileResponse:
        with Library(db_path) as library:
            material = _material_or_404(library, material_id)
        path = page_image(app.state.cfg.data_dir, material, page_no)
        if path is None:
            raise HTTPException(404, "没有这一页的图片")
        return FileResponse(path, media_type="image/png")

    @app.get("/reports/{report_id}/markdown")
    def report_markdown(report_id: int) -> Response:
        with Library(db_path) as library:
            report = _report_or_404(library, report_id)
        disposition = "attachment; filename*=UTF-8" + "''" + quote(report_filename(report))
        return Response(render_report_markdown(report.title, report.result),
                        media_type="text/markdown; charset=utf-8", headers={"Content-Disposition": disposition})

    @app.get("/api/jobs")
    def jobs_json() -> dict[str, Any]:
        """顶栏的任务数和任务页的列表。"""
        with Library(db_path) as library:
            store = JobStore(library)
            return {"active": len(store.active_jobs()), "jobs": [_job_dict(job) for job in store.latest(30)]}

    @app.get("/api/jobs/{job_id}")
    def job_json(job_id: int) -> dict[str, Any]:
        with Library(db_path) as library:
            return _job_dict(_job_or_404(library, job_id))

    @app.get("/media/{lecture_id}/{part}")
    def media(lecture_id: int, part: int) -> FileResponse:
        with Library(db_path) as library:
            lecture = _lecture_or_404(library, lecture_id)
        found = next((p for p in lecture.parts if p.part == part), None)
        if found is None:
            raise HTTPException(404, "这一讲没有这一节")
        path = Path(found.media_path)
        if not path.is_file():
            raise HTTPException(404, "原始音视频文件不存在（可能被移动或删除）")
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")

    @app.get("/api/lectures/{lecture_id}")
    def lecture_json(lecture_id: int) -> dict[str, Any]:
        with Library(db_path) as library:
            return build_lecture(library, CourseMemory(library), _lecture_or_404(library, lecture_id))

    @app.get("/api/courses/{course_id}")
    def course_json(course_id: int) -> dict[str, Any]:
        with Library(db_path) as library:
            return build_course_book(library, CourseMemory(library), _course_or_404(library, course_id))


def _job_dict(job: Job) -> dict[str, Any]:
    return {"id": job.id, "label": job.label, "state": job.state, "state_label": JOB_STATES.get(job.state, job.state),
            "active": job.state in ACTIVE_STATES, "progress": job.progress, "error": job.error,
            "url": job.result.get("url", ""), "summary": job.result.get("summary", ""),
            "warnings": job.result.get("warnings", []), "created_at": job.created_at, "updated_at": job.updated_at}


def _lecture_or_404(library: Library, lecture_id: int) -> Lecture:
    try:
        return library.lecture_by_id(lecture_id)
    except LibraryError:
        raise HTTPException(404, "没有这一讲") from None


def _material_or_404(library: Library, material_id: int) -> Material:
    try:
        return MaterialStore(library).get(material_id)
    except LibraryError:
        raise HTTPException(404, "没有这份资料") from None


def _report_or_404(library: Library, report_id: int) -> StoredReport:
    try:
        return ReportStore(library).get(report_id)
    except LibraryError:
        raise HTTPException(404, "没有这份考前报告") from None


def _job_or_404(library: Library, job_id: int) -> Job:
    try:
        return JobStore(library).get(job_id)
    except LibraryError:
        raise HTTPException(404, "没有这个任务") from None


def _course_or_404(library: Library, course_id: int) -> Course:
    try:
        return library.course_by_id(course_id)
    except LibraryError:
        raise HTTPException(404, "没有这门课") from None
