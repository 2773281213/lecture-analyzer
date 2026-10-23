"""导出：一门课是一本书（data/books/<课程>/），每讲一个文件夹。

- book.json（schema_version 3）：整本书——课程设置、各讲目录、知识点体系、跨讲联系、课程资料；
- 第NN讲/lecture.json：一讲的笔记、考点信号、逐句转写、各节信息、对应的课件页、与课程知识点的对应；
- 第NN讲/笔记.md、课程总览.md、总复习.md（生成过复习讲义后）、考前报告-时间.md：便于直接阅读的 Markdown。
导出文件都由书库生成，每次上架整门课一起重写，保证与书库一致。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .exam_plans import ExamPlan, ExamPlanStore, plan_text
from .knowledge import CourseMemory, KnowledgePoint
from .library import EXAM_PLAN_LABELS, Course, Lecture, Library
from .materials import MaterialStore
from .pipeline.report_render import render_report_markdown
from .pipeline.review import ChapterView, render_review_markdown, review_book
from .pipeline.validate import pass_rate
from .render import render_notes, render_overview, signal_stats
from .reports import ReportStore, StoredReport

BOOK_SCHEMA_VERSION = 3
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def media_type(path: str) -> str:
    return "video" if Path(path).suffix.lower() in VIDEO_SUFFIXES else "audio"


def lecture_media_type(lecture: Lecture) -> str:
    """有一节是录像就按录像播放（录像播放器也能放音频）。"""
    return "video" if any(media_type(part.media_path) == "video" for part in lecture.parts) else "audio"


def safe_name(name: str) -> str:
    """去掉 Windows 文件名里不允许的字符。"""
    return _UNSAFE_CHARS.sub("_", name).strip(" .") or "未命名"


def course_dir(data_dir: Path, course: Course) -> Path:
    return data_dir / "books" / safe_name(course.name)


def lecture_dir(data_dir: Path, course: Course, lecture: Lecture) -> Path:
    return course_dir(data_dir, course) / f"第{lecture.seq:02d}讲"


def work_dir(data_dir: Path, lecture_id: int) -> Path:
    """一讲的中间文件（抽出的音频、转写分片缓存）。"""
    return data_dir / "work" / f"lecture-{lecture_id}"


def build_lecture(library: Library, memory: CourseMemory, lecture: Lecture) -> dict[str, Any]:
    course = library.course_by_id(lecture.course_id)
    notes = library.notes(lecture.id)
    segments = library.segments(lecture.id)
    checked = library.signals(lecture.id)
    model = library.notes_model(lecture.id)
    keys = memory.kp_keys(lecture.id)
    signal_keys = memory.signal_keys(lecture.id)
    return {
        "schema_version": BOOK_SCHEMA_VERSION,
        "id": lecture.id,
        "course": _course_info(course),
        "lecture": {
            "seq": lecture.seq,
            "title": lecture.title or (notes.title if notes else ""),
            "date": lecture.date,
            "kind": lecture.kind,
            "duration_ms": lecture.duration_ms,
            "tags": library.tags(lecture.id),
            "media_type": lecture_media_type(lecture),
            "parts": [
                {"part": p.part, "media_path": p.media_path, "media_type": media_type(p.media_path),
                 "duration_ms": p.duration_ms, "offset_ms": offset}
                for p, offset in zip(lecture.parts, lecture.part_offsets())
            ],
        },
        "notes": notes.model_dump() if notes else None,
        "signals": [
            {**c.signal.model_dump(), "segment_ids": c.segment_ids, "valid": c.valid, "match_method": c.method,
             "kp_key": key}
            for c, key in zip(checked, signal_keys)
        ],
        "transcript": [
            {"idx": i, "start_ms": s.start_ms, "end_ms": s.end_ms, "text": s.text, "lang": s.lang}
            for i, s in enumerate(segments, start=1)
        ],
        "slides": _slides_info(library, lecture),
        "memory": {
            "state": lecture.step_state("integrate"),
            "knowledge_points": [{"note_index": index, "kp_key": key} for index, key in sorted(keys.items())],
            "references": [_reference_info(ref) for ref in memory.references(lecture.id)],
        },
        "processing": {
            "status": lecture.status,
            "notes_model": {"profile": model[0], "model": model[1]} if model else None,
            "validation_pass_rate": pass_rate(checked) if checked else None,
        },
    }


def build_course_book(
    library: Library, memory: CourseMemory, course: Course, review: list[ChapterView] | None = None
) -> dict[str, Any]:
    tree = memory.tree(course)
    stats = signal_stats(memory.history_signals(course))
    tags = library.course_tags(course)
    review = review if review is not None else review_book(library, course)
    return {
        "schema_version": BOOK_SCHEMA_VERSION,
        "course": {**_course_info(course), "instructions": course.instructions},
        "exam_plan": _exam_plan_info(library, course),
        "lectures": [
            {"id": lecture.id, "seq": lecture.seq, "title": lecture.title, "date": lecture.date, "kind": lecture.kind,
             "duration_ms": lecture.duration_ms, "parts": len(lecture.parts), "tags": tags.get(lecture.id, []),
             "status": {step: info.get("state", "") for step, info in lecture.status.items()}}
            for lecture in library.lectures(course)
        ],
        "knowledge_points": [_kp_info(kp, stats.get(kp.key, {})) for kp in tree],
        "references": [_reference_info(ref) for ref in memory.course_references(course)],
        "materials": [
            {"id": m.id, "kind": m.kind, "title": m.title, "source": m.source, "page_count": m.page_count,
             "status": {step: info.get("state", "") for step, info in m.status.items()}}
            for m in MaterialStore(library).list(course)
        ],
        "reports": [
            {"id": r.id, "title": r.title, "created_at": r.created_at, "models": r.models, "file": report_filename(r)}
            for r in ReportStore(library).list(course)
        ],
        "review": [
            {"chapter": chapter.chapter, "state": chapter.state, "generated_at": chapter.generated_at,
             "model": chapter.model, "overview_md": chapter.overview_md,
             "entries": [{"kp_key": e.key, "priority": e.priority, "summary_md": e.summary_md, "generated": e.generated,
                          "reviewed": e.reviewed} for e in chapter.entries]}
            for chapter in review
        ],
    }


def export_course(library: Library, data_dir: Path, course: Course) -> dict[int, Path]:
    """导出整门课，返回 {讲的编号: 文件夹}。"""
    memory = CourseMemory(library)
    folder = course_dir(data_dir, course)
    folder.mkdir(parents=True, exist_ok=True)
    folders = {lecture.id: export_lecture(library, memory, data_dir, course, lecture)
               for lecture in library.lectures(course)}
    review = review_book(library, course)
    book = build_course_book(library, memory, course, review)
    (folder / "book.json").write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
    tree = memory.tree(course)
    overview = render_overview(course, library.lectures(course), tree, signal_stats(memory.history_signals(course)))
    (folder / "课程总览.md").write_text(overview, encoding="utf-8")
    if any(chapter.state != "missing" for chapter in review):
        (folder / "总复习.md").write_text(render_review_markdown(course, review), encoding="utf-8")
    stored = ExamPlanStore(library).get(course)
    if stored is not None and stored.plan is not None:
        (folder / "考试方案.md").write_text(render_exam_plan_markdown(course, stored.plan), encoding="utf-8")
    return folders


def _exam_plan_info(library: Library, course: Course) -> dict[str, Any]:
    stored = ExamPlanStore(library).get(course)
    plan = stored.plan if stored else None
    return {"state": course.exam_plan, "plan": plan.model_dump() if plan else None,
            "confirmed_at": stored.confirmed_at if stored else ""}


def render_exam_plan_markdown(course: Course, plan: ExamPlan) -> str:
    state = EXAM_PLAN_LABELS.get(course.exam_plan, course.exam_plan)
    return f"# {course.name} · 考试方案（{state}）\n\n{plan_text(plan) or '（没有内容）'}\n"


def report_filename(report: StoredReport) -> str:
    stamp = report.created_at.replace("-", "").replace(":", "").replace("T", "-")[:13]
    return f"考前报告-{stamp}-{report.id}.md"


def remove_report(library: Library, data_dir: Path, course: Course, report: StoredReport) -> None:
    """删除一份考前报告：书库里的记录、导出的 Markdown，并更新 book.json 里的报告列表。"""
    ReportStore(library).remove(report.id)
    (course_dir(data_dir, course) / report_filename(report)).unlink(missing_ok=True)
    export_course(library, data_dir, course)


def export_report(data_dir: Path, course: Course, report: StoredReport) -> Path:
    folder = course_dir(data_dir, course)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / report_filename(report)
    path.write_text(render_report_markdown(report.title, report.result), encoding="utf-8")
    return path


def export_lecture(library: Library, memory: CourseMemory, data_dir: Path, course: Course, lecture: Lecture) -> Path:
    folder = lecture_dir(data_dir, course, lecture)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "book.json").unlink(missing_ok=True)  # M1 导出的旧文件，已改名为 lecture.json
    data = build_lecture(library, memory, lecture)
    (folder / "lecture.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    notes = library.notes(lecture.id)
    if notes is not None:
        heading = f"{course.name} 第{lecture.seq}讲：{data['lecture']['title']}"
        markdown = render_notes(
            notes, library.segments(lecture.id), library.signals(lecture.id), heading=heading,
            references=memory.references(lecture.id),
            page_labels=MaterialStore(library).page_labels(library.note_pages(lecture.id)),
        )
        (folder / "笔记.md").write_text(markdown, encoding="utf-8")
    return folder


def _slides_info(library: Library, lecture: Lecture) -> dict[str, Any]:
    """笔记里 P1、P2… 对应的课件页，以及这一讲当前匹配到的页（重新整理笔记前两者可能不同）。"""
    store = MaterialStore(library)
    refs = library.note_pages(lecture.id)
    return {
        "notes_pages": [
            {"p": index, "material_id": ref.material_id, "page_no": ref.page_no, "label": label}
            for index, (ref, label) in enumerate(zip(refs, store.page_labels(refs)), start=1)
        ],
        "matched": [
            {"material_id": page.ref.material_id, "page_no": page.ref.page_no, "source": page.source}
            for page in store.lecture_pages(lecture.id)
        ],
    }


def _course_info(course: Course) -> dict[str, Any]:
    settings = course.settings
    return {"id": course.id, "name": course.name, "subject": settings.subject, "teaching": settings.teaching,
            "interpreted_to": settings.interpreted_to, "note_language": settings.note_language,
            "slides": course.slides}


def _kp_info(kp: KnowledgePoint, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": kp.key, "name": kp.name, "chapter": kp.chapter, "aliases": list(kp.aliases),
        "appearances": [{"seq": a.seq, "role": a.role, "minutes": a.minutes, "note_index": a.note_index}
                        for a in kp.appearances],
        "signals": stats,
    }


def _reference_info(ref) -> dict[str, Any]:
    return {"note_index": ref.note_index, "kp_key": ref.key, "kp_name": ref.kp_name, "target_seq": ref.target_seq,
            "relation": ref.relation, "explanation": ref.explanation, "segment_ids": ref.segment_ids}
