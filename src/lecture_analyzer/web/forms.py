"""网页写操作的公共部分：读取并校验表单、取整数和本机路径字段、提交后台任务、重定向。

所有写操作都经过这里，安全措施：
- 防跨站请求伪造：每次启动生成随机令牌，页面表单带隐藏字段 csrf，这里用 compare_digest 校验，不对返回 403；
- 防 DNS rebinding：Host 检查由 app.py 的 TrustedHostMiddleware 负责；
- 请求体有大小和字段数限制，所有字段先校验再写入；重定向地址由服务端拼出，不使用表单传来的地址；
- 本机路径（添加讲、导入资料时选的文件）必须是存在的绝对路径。书架只监听本机、写操作要带令牌，
  路径是使用者在自己电脑上选的，视为可信输入（与命令行一致）。
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode

from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse

from ..jobs import JobStore
from ..library import Course, Library

MAX_FORM_BYTES = 4096
MAX_FORM_FIELDS = 20
LONG_FORM_BYTES = 64 * 1024  # 课程提示词、多个文件路径
MAX_NAME_CHARS = 200
MAX_PATHS = 100
MAX_PATH_CHARS = 1000


class Form(dict[str, str]):
    """表单字段：按名称取第一个值；同名的多个值（复选框）用 getlist。"""

    def __init__(self, fields: dict[str, list[str]]) -> None:
        super().__init__({key: values[0] for key, values in fields.items()})
        self.lists = fields

    def getlist(self, name: str) -> list[str]:
        return self.lists.get(name, [])


async def read_form(request: Request, max_bytes: int = MAX_FORM_BYTES, max_fields: int = MAX_FORM_FIELDS) -> Form:
    """读取 application/x-www-form-urlencoded 表单并校验令牌。"""
    declared = request.headers.get("content-length") or "0"
    if not declared.isdigit() or int(declared) > max_bytes:
        raise HTTPException(413, "表单内容过长")
    body = await request.body()
    if len(body) > max_bytes:
        raise HTTPException(413, "表单内容过长")
    try:
        fields = parse_qs(body.decode("utf-8"), keep_blank_values=True, max_num_fields=max_fields,
                          strict_parsing=False)
    except (UnicodeDecodeError, ValueError):
        raise HTTPException(400, "表单格式不正确") from None
    form = Form(fields)
    expected = request.app.state.csrf_token.encode()
    if not secrets.compare_digest(form.get("csrf", "").encode(), expected):
        raise HTTPException(403, "无法确认请求来自本书架页面，请刷新页面后重试")
    return form


def library_of(request: Request) -> Library:
    return Library(request.app.state.cfg.data_dir / "lecture.db")


def redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def int_field(form: dict[str, str], name: str, label: str, required: bool = True) -> int | None:
    value = form.get(name, "").strip()
    if not value and not required:
        return None
    if not value.isdigit() or int(value) < 1:
        raise HTTPException(400, f"{label}必须是正整数")
    return int(value)


def numbers(values: list[str], label: str) -> list[int]:
    if any(not value.isdigit() for value in values):
        raise HTTPException(400, f"{label}必须是正整数")
    return [int(value) for value in values]


def text_field(form: dict[str, str], name: str, label: str, limit: int = MAX_NAME_CHARS) -> str:
    value = form.get(name, "").strip()
    if len(value) > limit:
        raise HTTPException(400, f"{label}最多 {limit} 字")
    return value


def path_list(text: str, label: str, files_only: bool = False) -> list[Path]:
    """每行一个路径（资源管理器「复制文件地址」带的引号会去掉）；必须是存在的绝对路径，去重后保持顺序。"""
    paths: list[Path] = []
    for raw in text.splitlines():
        value = raw.strip().strip('"').strip()
        if not value:
            continue
        if len(value) > MAX_PATH_CHARS or "\x00" in value:
            raise HTTPException(400, f"{label}的路径过长或不合法")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise HTTPException(400, f"{label}要写完整路径（如 D:\\录课\\第1讲.mp4）：{value}")
        if not (path.is_file() or (not files_only and path.is_dir())):
            raise HTTPException(400, f"找不到{label}：{value}")
        paths.append(path.resolve())
    if not paths:
        raise HTTPException(400, f"请选择{label}")
    if len(paths) > MAX_PATHS:
        raise HTTPException(400, f"一次最多 {MAX_PATHS} 个{label}")
    return list(dict.fromkeys(paths))


def submit_job(request: Request, library: Library, course: Course | None, kind: str,
               params: dict[str, Any]) -> RedirectResponse:
    """提交后台任务并转到任务页；同样的任务还没结束时直接转到那个任务，不重复提交。"""
    job = enqueue(request, library, course, kind, params)
    return redirect(f"/jobs/{job}")


def enqueue(request: Request, library: Library, course: Course | None, kind: str, params: dict[str, Any]) -> int:
    """提交后台任务（或找到同样的、还没结束的任务），返回任务编号。"""
    store = JobStore(library)
    running = store.active(course, kind, params)
    if running is not None:
        return running.id
    job = store.add(course, kind, params)
    request.app.state.jobs.submit(job.id)
    return job.id


# ---------- 处理讲的参数、课件关卡、提示 ----------


def run_params(form: Form, library: Library, course: Course) -> dict[str, Any]:
    """处理讲的任务参数：lecture 为讲的编号（不选表示所有没完成的讲），force 为从哪一步起重做（all 表示全部）。"""
    from ..pipeline.run import STEPS

    ids = numbers(form.getlist("lecture"), "讲的编号")
    for lecture_id in ids:
        if library.lecture_by_id(lecture_id).course_id != course.id:
            raise HTTPException(400, "所选的讲不属于这门课")
    force = form.get("force", "").strip()
    if force and force not in (*STEPS, "all"):
        raise HTTPException(400, f"未知的步骤：{force}")
    return {"lectures": ids, "force": [force] if force else []}


def gate_url(page: str, params: dict[str, Any]) -> str:
    """课件关卡：课程还没选择是否提供课件时，回到页面并弹出选择框（带上原本要处理的讲和步骤）。"""
    query = urlencode({"gate": "1", "lecture": params["lectures"], "force": params["force"]}, doseq=True)
    return f"{page}?{query}"


def flash(request: Request, message: str, kind: str = "ok") -> None:
    """下一个页面顶部显示的提示（单人使用的本机程序，存在应用状态里即可）。"""
    request.app.state.flash = {"message": message, "kind": kind}


def take_flash(request: Request) -> dict[str, str] | None:
    message = getattr(request.app.state, "flash", None)
    request.app.state.flash = None
    return message
