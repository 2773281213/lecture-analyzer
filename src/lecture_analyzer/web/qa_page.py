"""问答页：一门课里的多段对话。可以新开对话、接着追问、限定某一讲；提问作为轻任务在后台回答
（不会被正在跑的转写挡住），页面显示进度，答完自动刷新；没答成的问题留在对话里，显示原因，可以重新提问。
回答里的出处编号渲染成链接（见 app.py 的 cite 过滤器）。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ..conversations import QAStore
from ..jobs import Job, JobStore
from ..knowledge import CourseMemory
from ..library import Course, Library
from ..pipeline.qa import MAX_QUESTION_CHARS
from .forms import LONG_FORM_BYTES, enqueue, int_field, library_of, read_form, redirect
from .views import spine_color

router = APIRouter()


def _ready_lectures(library: Library, course: Course) -> set[int]:
    """已整合进课程记忆、可以限定范围提问的讲。"""
    return {a.lecture_id for kp in CourseMemory(library).tree(course) for a in kp.appearances}


def _thread_jobs(library: Library, course: Course, thread: int, answered_at: str) -> tuple[list[Job], list[Job]]:
    """这段对话里还在回答的问题，以及最后一次回答之后失败或取消的问题。"""
    jobs = [job for job in JobStore(library).of_kind(course, "qa") if job.params.get("thread") == thread]
    pending = [job for job in jobs if not job.finished]
    failed = [job for job in jobs if job.state in ("failed", "cancelled") and job.updated_at > answered_at]
    return pending, failed


@router.get("/courses/{course_id}/qa", response_class=HTMLResponse)
def qa_page(request: Request, course_id: int, thread: int = 0, lecture: int = 0) -> HTMLResponse:
    """thread 为 0 时是新对话；lecture 为预先选好的提问范围（从讲的页面点「就这一讲提问」）。"""
    with library_of(request) as library:
        course = library.course_by_id(course_id)
        store = QAStore(library)
        turns = store.thread(course, thread) if thread else []
        pending, failed = _thread_jobs(library, course, thread, turns[-1].created_at if turns else "")
        last_scope = turns[-1].lecture_id if turns else None  # 追问时默认沿用上一问的范围
        ready = _ready_lectures(library, course)
        scope = lecture or last_scope or 0
        context = {
            "course": course, "spine": spine_color(course), "threads": store.threads(course), "thread": thread,
            "turns": turns, "pending": pending, "failed": failed, "lectures": library.lectures(course),
            "scope": scope if scope in ready else 0, "ready": ready,
            "has_memory": bool(ready), "max_chars": MAX_QUESTION_CHARS,
        }
    return request.app.state.templates.TemplateResponse(request, "qa.html", context)


@router.post("/courses/{course_id}/qa")
async def ask(request: Request, course_id: int) -> RedirectResponse:
    form = await read_form(request, LONG_FORM_BYTES)
    question = form.get("question", "").strip()
    if not question:
        raise HTTPException(400, "请输入问题")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"问题最多 {MAX_QUESTION_CHARS} 字")
    lecture_id = int_field(form, "lecture", "讲", required=False)
    with library_of(request) as library:
        course = library.course_by_id(course_id)
        if lecture_id and library.lecture_by_id(lecture_id).course_id != course.id:
            raise HTTPException(400, "所选的讲不属于这门课")
        if lecture_id and lecture_id not in _ready_lectures(library, course):
            raise HTTPException(400, "这一讲还没有整合进课程记忆，先处理这一讲再提问")
        thread = int_field(form, "thread", "对话编号", required=False) or _new_thread(library, course)
        enqueue(request, library, course, "qa", {"thread": thread, "question": question, "lecture_id": lecture_id})
    return redirect(f"/courses/{course_id}/qa?thread={thread}")


def _new_thread(library: Library, course: Course) -> int:
    """新对话的编号：已存的对话和提过问的对话（含还在回答、没答成的）都要避开。"""
    asked = [int(job.params.get("thread", 0)) + 1 for job in JobStore(library).of_kind(course, "qa")]
    return max([QAStore(library).next_thread(course), *asked])


@router.post("/courses/{course_id}/qa/{thread}/delete")
async def delete_thread(request: Request, course_id: int, thread: int) -> RedirectResponse:
    await read_form(request)
    with library_of(request) as library:
        QAStore(library).delete_thread(library.course_by_id(course_id), thread)
    return redirect(f"/courses/{course_id}/qa")
