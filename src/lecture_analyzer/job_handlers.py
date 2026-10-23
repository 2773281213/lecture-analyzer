"""各类后台任务的处理，在任务子进程里执行（见 worker.py；测试时直接调用 run_job）。

参数在提交前已经校验过（web/actions.py）；这里照样按编号取出课程、讲和资料，取不到时任务失败并说明原因。
每个处理函数返回任务结果：url 是"查看结果"的地址，warnings 是要提醒的事项。
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import AppConfig
from .jobs import Job, JobStore
from .library import Course, Lecture, Library, LibraryError

Report = Callable[[str], None]
Handler = Callable[[AppConfig, Library, Job, Report], dict[str, Any]]


def run_job(cfg: AppConfig, db_path: Path, job_id: int) -> None:
    """执行一个任务：结果、失败原因都写回任务表；任务已被取消时什么也不做。"""
    with Library(db_path) as library:
        store = JobStore(library)
        store.start(job_id)  # 执行器已经标为进行中时什么也不做
        job = store.get(job_id)
        if job.state != "running":
            return
        try:
            result = HANDLERS[job.kind](cfg, library, job, lambda message: store.report(job_id, message))
        except Exception as exc:  # 任务边界：记录失败原因，网页上显示；完整的出错位置写进任务日志
            traceback.print_exc()
            store.fail(job_id, f"{type(exc).__name__}: {exc}")
            return
        store.finish(job_id, result)


def _course(library: Library, job: Job) -> Course:
    if job.course_id is None:
        raise LibraryError("任务缺少课程")
    return library.course_by_id(job.course_id)


def _course_url(course: Course, anchor: str = "") -> str:
    return f"/courses/{course.id}{anchor}"


class _LectureReport:
    """处理多讲时，在进度前面加上正在处理的是第几讲。"""

    def __init__(self, report: Report) -> None:
        self.report = report
        self.prefix = ""

    def announce(self, lecture: Lecture) -> None:
        self.prefix = f"第 {lecture.seq} 讲 · "

    def __call__(self, message: str) -> None:
        self.report(self.prefix + message)


# ---------- 讲与课程记忆 ----------


def _run(cfg: AppConfig, library: Library, job: Job, report: Report) -> dict[str, Any]:
    """处理讲：lectures 为讲的编号（空表示这门课所有没完成的讲）；force 为要重跑的步骤（all 表示全部）。

    考试方案还没确认时停在整理笔记之前：任务照常结束（不算失败），结果转到课程页的考试方案。"""
    from .pipeline.exam_plan import AwaitingExamPlan
    from .pipeline.run import STEPS, RunOptions, process_lectures

    course = _course(library, job)
    params = job.params
    ids = [int(i) for i in params.get("lectures") or []]
    force = set(params.get("force") or [])
    options = RunOptions(force=frozenset(STEPS if "all" in force else force & set(STEPS)),
                         model=params.get("model") or None, asr=params.get("asr") or None)
    progress = _LectureReport(report)
    try:
        processed = process_lectures(cfg, library, course, [library.lecture_by_id(i) for i in ids] or None, options,
                                     progress, progress.announce)
    except AwaitingExamPlan as exc:
        return {"url": _course_url(course, "#exam-plan"), "summary": str(exc), "awaiting_exam_plan": True}
    url = f"/lectures/{ids[0]}" if len(ids) == 1 else _course_url(course)
    return {"url": url, "summary": f"处理了 {processed} 讲" if processed else "所有讲都已处理完成"}


def _rebuild(cfg: AppConfig, library: Library, job: Job, report: Report) -> dict[str, Any]:
    from .pipeline.run import rebuild_course

    course = _course(library, job)
    from_seq = job.params.get("from_seq")
    outcome = rebuild_course(cfg, library, course, int(from_seq) if from_seq else None,
                             job.params.get("model") or None, report)
    if outcome is None:
        return {"url": _course_url(course), "summary": "课程记忆已是最新，不需要重建"}
    return {"url": _course_url(course),
            "summary": f"重放 {outcome.replayed} 讲、重新整合 {len(outcome.integrated)} 讲，已重新导出"}


def _exam_plan(cfg: AppConfig, library: Library, job: Job, report: Report) -> dict[str, Any]:
    """分析考试方案（来源讲和考试说明在提交前已保存），结果为待确认的草稿。"""
    from .pipeline.exam_plan import analyze_exam_plan

    course = _course(library, job)
    lecture_id = job.params.get("lecture")
    analysis = analyze_exam_plan(cfg, library, course, int(lecture_id) if lecture_id else None, report=report)
    return {"url": _course_url(course, "#exam-plan"), "summary": f"考试方案待确认：{analysis.summary}"}


# ---------- 课程资料 ----------


def _import_material(cfg: AppConfig, library: Library, job: Job, report: Report) -> dict[str, Any]:
    """导入资料（文件、文件夹或压缩包），process 为真时接着逐页识别；
    pin_lecture 为讲的编号时，把这一讲固定对应到导入的课件（添加讲时附带的课件）。"""
    from .pipeline.ingest import import_materials, pin_lecture_pages
    from .pipeline.materials import process_materials

    course = _course(library, job)
    params = job.params
    result = import_materials(library, cfg.data_dir, course, [Path(p) for p in params["paths"]], params["kind"],
                              params.get("title", ""), report=report)
    if params.get("pin_lecture"):
        pin_lecture_pages(library, int(params["pin_lecture"]), result.materials[0])
    if params.get("process", True):
        process_materials(cfg, library, course, result.materials, report=report)
    single = len(result.materials) == 1
    return {"url": f"/materials/{result.materials[0].id}" if single else _course_url(course, "#materials"),
            "summary": f"导入了 {len(result.materials)} 份资料",
            "warnings": [f"跳过 {source}：{reason}" for source, reason in result.skipped]}


def _process_material(cfg: AppConfig, library: Library, job: Job, report: Report) -> dict[str, Any]:
    """继续处理资料：materials 为资料编号（空表示这门课所有没处理完的）。"""
    from .materials import MaterialStore
    from .pipeline.materials import process_materials

    course = _course(library, job)
    store = MaterialStore(library)
    ids = [int(i) for i in job.params.get("materials") or []]
    process_materials(cfg, library, course, [store.get(i) for i in ids] or None, report=report)
    return {"url": f"/materials/{ids[0]}" if len(ids) == 1 else _course_url(course, "#materials")}


# ---------- 本地模型 ----------


def _download_models(cfg: AppConfig, library: Library, job: Job, report: Report) -> dict[str, Any]:
    from .asr.models import download_models

    downloaded = download_models(cfg, job.params.get("spec") or None, report)
    return {"url": "/settings#asr", "summary": f"下载了 {len(downloaded)} 个模型" if downloaded else "模型都已下载"}


# ---------- 考前报告与总复习 ----------


def _report(cfg: AppConfig, library: Library, job: Job, report: Report) -> dict[str, Any]:
    from .pipeline.report import build_plan, generate_report, save_report

    course = _course(library, job)
    params = job.params
    plan = build_plan(cfg, library, course, params.get("lectures", []), params.get("tag", ""),
                      params.get("materials", []), params.get("models", []))
    result = generate_report(cfg, library, plan, report=report)
    report_id, _ = save_report(library, cfg.data_dir, plan, result)
    return {"url": f"/reports/{report_id}", "report_id": report_id, "warnings": result["warnings"]}


def _review(cfg: AppConfig, library: Library, job: Job, report: Report) -> dict[str, Any]:
    from .books import export_course
    from .pipeline.materials import ensure_mapped
    from .pipeline.review import generate_review, plan_review, review_context

    course = _course(library, job)
    ensure_mapped(cfg, library, course, report)
    planned = plan_review(library, course, bool(job.params.get("all")))
    if not planned:
        raise LibraryError("这门课还没有课程记忆：先处理至少一讲")
    needed = [p for p in planned if p.needed]
    if needed:
        report(f"将生成 {len(needed)} 章（共 {len(planned)} 章）")
        generate_review(review_context(cfg, library, report), course, planned)
    export_course(library, cfg.data_dir, course)
    return {"url": f"/courses/{course.id}/review", "chapters": len(needed)}


# ---------- 问答 ----------


def _qa(cfg: AppConfig, library: Library, job: Job, report: Report) -> dict[str, Any]:
    """回答一个问题并存进这段对话：thread 为对话编号，lecture_id 为限定的讲（空表示整门课）。"""
    from .conversations import QAStore
    from .pipeline.qa import answer_question

    course = _course(library, job)
    params = job.params
    lecture = library.lecture_by_id(int(params["lecture_id"])) if params.get("lecture_id") else None
    store = QAStore(library)
    thread = int(params["thread"])
    result = answer_question(cfg, library, course, params["question"], lecture, store.thread(course, thread), report)
    turn_id = store.add(course, lecture.id if lecture else None, thread, result["question"], result,
                        list(result["models"].values()))
    return {"url": f"/courses/{course.id}/qa?thread={thread}#qa-{turn_id}", "turn_id": turn_id}


HANDLERS: dict[str, Handler] = {
    "run": _run,
    "rebuild": _rebuild,
    "exam_plan": _exam_plan,
    "import_material": _import_material,
    "process_material": _process_material,
    "download_models": _download_models,
    "report": _report,
    "review": _review,
    "qa": _qa,
}
