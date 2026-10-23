"""网页各页面要展示的数据（书架、课程目录与知识点体系、讲的阅读页、资料、考前报告、后台任务）。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from ..books import lecture_media_type, media_type
from ..config import AppConfig
from ..jobs import Job, JobStore
from ..knowledge import CourseMemory, HistorySignal, KnowledgePoint, LectureCache
from ..library import Course, Lecture, Library, PageRef
from ..materials import ITEM_KINDS, PAGE_KINDS, Material, MaterialStore
from ..pipeline.ingest import page_image
from ..pipeline.materials import ITEM_TYPES, MaterialHit, material_hits, stale_materials
from ..pipeline.organize import next_free_seq
from ..pipeline.review import kp_priority, review_book
from ..pipeline.run import STEP_LABELS, STEPS
from ..pipeline.slides import describe_pages
from ..pipeline.report_render import PRIORITY_LEGEND
from ..reports import ReportStore
from ..render import RELATION_LABELS, ROLE_LABELS, SIGNAL_LABELS, first_time, group_by_chapter, signal_stats
from ..schemas import LectureNotes
from ..textnorm import normalize_zh
from .exam_plan_page import plan_context

# 书脊颜色：按课程编号轮换，和样式表里的配色一致
SPINE_COLORS = ("#2f6f62", "#9a4a37", "#3d5a80", "#7a5c2e", "#5b4b8a", "#4f6d3a")
SORTS = {"custom": "自定义", "recent": "最近更新", "name": "名称"}


def spine_color(course: Course) -> str:
    return SPINE_COLORS[(course.id - 1) % len(SPINE_COLORS)]


def lecture_state(lecture: Lecture) -> tuple[str, str]:
    """一讲的整体状态：（样式名，中文说明）。"""
    for step in STEPS:
        if lecture.step_state(step) == "failed":
            return "failed", f"{STEP_LABELS[step]}失败"
    if lecture.step_state("integrate") == "stale":
        return "stale", "课程记忆待更新"
    if lecture.step_state("book") == "done":
        return "done", "已上架"
    if lecture.status:
        return "running", "处理中"
    return "pending", "待处理"


def progress(rows: Sequence[Any]) -> dict[str, int]:
    """复习进度：rows 里每项都有 reviewed 和 priority（课程页的知识点行、复习页的条目都适用）。"""
    return {
        "kp_total": len(rows),
        "kp_reviewed": sum(row.reviewed for row in rows),
        "key_total": sum(row.priority == 3 for row in rows),
        "key_reviewed": sum(row.priority == 3 and row.reviewed for row in rows),
    }


# ---------- 书架 ----------


@dataclass
class ShelfBook:
    course: Course
    spine: str
    lectures: int
    duration_ms: int
    last_activity: str
    states: list[tuple[str, str, int]]  # （样式名，中文说明，讲数）
    kp_total: int
    kp_reviewed: int


def shelf_books(library: Library, sort: str) -> list[ShelfBook]:
    memory = CourseMemory(library)
    books = [_shelf_book(library, memory, course) for course in library.courses()]
    if sort == "recent":
        books.sort(key=lambda b: b.last_activity, reverse=True)
    elif sort == "name":
        books.sort(key=lambda b: b.course.name)
    return books


def _shelf_book(library: Library, memory: CourseMemory, course: Course) -> ShelfBook:
    lectures = library.lectures(course)
    states: dict[tuple[str, str], int] = {}
    for lecture in lectures:
        state = lecture_state(lecture)
        states[state] = states.get(state, 0) + 1
    tree = memory.tree(course)
    reviewed = library.reviewed_names(course.id)
    return ShelfBook(
        course=course,
        spine=spine_color(course),
        lectures=len(lectures),
        duration_ms=sum(lecture.duration_ms or 0 for lecture in lectures),
        last_activity=max((lecture.last_activity for lecture in lectures), default=""),
        states=[(key, label, count) for (key, label), count in states.items()],
        kp_total=len(tree),
        kp_reviewed=sum(is_reviewed(kp, reviewed) for kp in tree),
    )


def is_reviewed(kp: KnowledgePoint, reviewed: set[str]) -> bool:
    return bool(kp.norm_names & reviewed)


# ---------- 课程（书）页 ----------


@dataclass
class KPRow:
    kp: KnowledgePoint
    norm_name: str  # 记复习进度用的名称
    appearances: list[dict[str, Any]]  # 讲次、角色、跳转链接
    signals: list[tuple[str, int]]  # （信号类型中文名，条数）
    reviewed: bool
    priority: int  # 重要程度，见 review.kp_priority


def course_view(library: Library, course: Course, tag: str | None, cfg: AppConfig) -> dict[str, Any]:
    tags = library.course_tags(course)
    lectures = library.lectures(course)
    chapters = _kp_chapters(library, course)
    materials = material_rows(library, course)
    return {
        "materials": materials,
        "pending_materials": any(unfinished(row.material) for row in materials),
        "reports": ReportStore(library).list(course)[:20],
        "jobs": JobStore(library).recent(course),
        "active_jobs": JobStore(library).active_jobs(course),
        "next_seq": next_free_seq(library, course),
        "models": ready_profiles(cfg),
        "default_model": cfg.steps.get("report", ""),
        "all_lectures": lectures,
        "course": course,
        "spine": spine_color(course),
        "lectures": [(lecture, lecture_state(lecture), tags.get(lecture.id, [])) for lecture in lectures
                     if tag is None or tag in tags.get(lecture.id, [])],
        "all_tags": sorted({t for values in tags.values() for t in values}),
        "active_tag": tag,
        "chapters": chapters,
        "total_ms": sum(lecture.duration_ms or 0 for lecture in lectures),
        **progress([row for _, rows in chapters for row in rows]),
        **plan_context(library, course),
    }


def _kp_chapters(library: Library, course: Course) -> list[tuple[str, list[KPRow]]]:
    """按章节分组的知识点体系。"""
    memory = CourseMemory(library)
    cache = LectureCache(library)
    history = memory.history_signals(course)
    stats = signal_stats(history)
    reviewed = library.reviewed_names(course.id)
    hits = material_hits(library, course)
    return [
        (chapter, [_kp_row(kp, stats.get(kp.key, {}), history, reviewed, cache, hits.get(kp.key, []))
                   for kp in points])
        for chapter, points in group_by_chapter(memory.tree(course))
    ]


def _kp_row(
    kp: KnowledgePoint, stats: dict[str, Any], history: Sequence[HistorySignal], reviewed: set[str], data: LectureCache,
    hits: Sequence[MaterialHit] = (),
) -> KPRow:
    appearances = [
        {"seq": a.seq, "role": ROLE_LABELS.get(a.role, a.role), "lecture_id": a.lecture_id,
         "t": data.kp_time(a.lecture_id, a.note_index), "minutes": a.minutes}
        for a in kp.appearances
    ]
    signals = [(SIGNAL_LABELS.get(t, t), v["count"]) for t, v in stats.items()]
    priority = kp_priority([s for s in history if s.key == kp.key], len({a.seq for a in kp.appearances}), hits)
    return KPRow(kp, normalize_zh(kp.name), appearances, signals, is_reviewed(kp, reviewed), priority)


# ---------- 总复习页 ----------


def review_view(library: Library, course: Course, focus: bool) -> dict[str, Any]:
    """focus 为真时只看重点（三星）知识点。"""
    chapters = review_book(library, course)
    summary = {
        "generated": any(c.state != "missing" for c in chapters),
        "stale": [c.chapter for c in chapters if c.state == "stale"],
        **progress([entry for chapter in chapters for entry in chapter.entries]),
    }
    if focus:
        chapters = [replace(c, entries=[e for e in c.entries if e.priority == 3]) for c in chapters]
        chapters = [c for c in chapters if c.entries]
    return {"course": course, "spine": spine_color(course), "chapters": chapters, "focus": focus, **summary}


# ---------- 讲（章）的阅读页 ----------


def lecture_view(library: Library, lecture: Lecture) -> dict[str, Any]:
    course = library.course_by_id(lecture.course_id)
    memory = CourseMemory(library)
    data = LectureCache(library)
    segments = data.segments(lecture.id)
    notes = data.notes(lecture.id)
    offsets = lecture.part_offsets()
    return {
        "course": course,
        "lecture": lecture,
        "notes": notes,
        "segments": segments,
        "signals": library.signals(lecture.id),
        "media_kind": lecture_media_type(lecture),
        "parts": [
            {"src": f"/media/{lecture.id}/{part.part}", "offset": offset, "duration": part.duration_ms or 0,
             "video": media_type(part.media_path) == "video"}
            for part, offset in zip(lecture.parts, offsets)
        ],
        "total_ms": _continuous_total(lecture, offsets),
        "spine": spine_color(course),
        "time_of": lambda ids: first_time(ids, segments),
        "steps": [(STEP_LABELS[s], lecture.status.get(s, {})) for s in STEPS],
        "model": library.notes_model(lecture.id),
        "kp_links": _kp_links(memory, course, lecture, data),
        "references": _references(memory, lecture, notes, data),
        "memory_state": lecture.step_state("integrate"),
        "tags": library.tags(lecture.id),
        "other_courses": [c for c in library.courses() if c.id != course.id],
        "active_jobs": JobStore(library).active_jobs(course),
        "page_materials": MaterialStore(library).list(course, PAGE_KINDS),
        **_slide_info(library, course, lecture),
    }


def _slide_info(library: Library, course: Course, lecture: Lecture) -> dict[str, Any]:
    """本讲对应的课件页，以及笔记里 P1、P2… 的出处和页面图片地址。"""
    store = MaterialStore(library)
    current = store.lecture_pages(lecture.id)
    titles = {m.id: m.title for m in store.list(course)}
    refs = library.note_pages(lecture.id)
    links = {
        index: {"url": page_url(ref), "label": label}
        for index, (ref, label) in enumerate(zip(refs, store.page_labels(refs)), start=1)
    }
    return {
        "matched_pages": describe_pages([page.ref for page in current], titles) if current else "",
        "matched_manual": bool(current) and current[0].source == "manual",
        "page_links": links,
    }


def page_url(ref: PageRef) -> str:
    return f"/materials/{ref.material_id}/pages/{ref.page_no}.png"


# ---------- 课程资料 ----------


@dataclass
class MaterialRow:
    material: Material
    digested: int  # 已识别的页数
    items: int
    stale: bool  # 题目与知识点的对应待更新


def unfinished(material: Material) -> bool:
    """还没处理完的资料：页面没识别完，或者试题类资料还没拆题。"""
    return material.step_state("digest") != "done" or (material.kind in ITEM_KINDS and
                                                        material.step_state("items") != "done")


def material_rows(library: Library, course: Course) -> list[MaterialRow]:
    store = MaterialStore(library)
    stale = {m.id for m in stale_materials(library, course)}
    return [MaterialRow(m, len(store.pages(m.id)), len(store.items(m.id)), m.id in stale) for m in store.list(course)]


def material_view(library: Library, data_dir: Any, material: Material) -> dict[str, Any]:
    course = library.course_by_id(material.course_id)
    store = MaterialStore(library)
    names = {kp.key: kp.name for kp in CourseMemory(library).tree(course)}
    lectures = {lecture.id: lecture for lecture in library.lectures(course)}
    using = [lectures[i] for i in store.lectures_using(material.id) if i in lectures]
    return {
        "course": course, "material": material, "spine": spine_color(course),
        "pages": [(page, page_image(data_dir, material, page.page_no) is not None) for page in store.pages(material.id)],
        "items": [(item, [(key, names.get(key, "")) for key in item.kp_keys]) for item in store.items(material.id)],
        "stale": material.id in {m.id for m in stale_materials(library, course)},
        "unfinished": unfinished(material),
        "lectures": sorted(using, key=lambda lecture: lecture.seq),
        "ITEM_TYPES": ITEM_TYPES,
    }


def ready_profiles(cfg: AppConfig) -> list[str]:
    """填好了型号和密钥、可以直接调用的模型档案。"""
    return [name for name, profile in cfg.models.items()
            if profile.is_configured and (not profile.api_key_env or profile.api_key())]


# ---------- 考前报告与后台任务 ----------


def report_view(library: Library, report_id: int) -> dict[str, Any]:
    report = ReportStore(library).get(report_id)
    course = library.course_by_id(report.course_id)
    return {"course": course, "report": report, "result": report.result, "spine": spine_color(course),
            "legacy": report.result.get("version", 1) < 2, "legend": PRIORITY_LEGEND}


def job_view(library: Library, job: Job) -> dict[str, Any]:
    course = library.course_by_id(job.course_id) if job.course_id else None
    return {"job": job, "course": course, "spine": spine_color(course) if course else SPINE_COLORS[0]}


def jobs_view(library: Library) -> dict[str, Any]:
    """任务页：最近的任务，最新的在前；顶部的「全部暂停 / 全部继续 / 全部重试」按全部任务统计（不限于这 50 个）。"""
    store = JobStore(library)
    jobs = store.latest(50)
    return {"jobs": jobs, "course_names": {course.id: course.name for course in library.courses()},
            "has_active": any(not job.finished for job in jobs),
            "active_count": len(store.active_jobs()), "paused_count": len(store.paused_jobs()),
            "retry_count": len(store.retry_candidates())}


def _continuous_total(lecture: Lecture, offsets: Sequence[int]) -> int | None:
    """多节的讲在播放器上连成一条整讲进度条，返回整讲时长；只有一节（原生控件就是整讲），
    或还有节的时长未知（抽音频之前）时返回 None，保留原生控件逐节播放。"""
    if len(lecture.parts) < 2 or not all(part.duration_ms for part in lecture.parts):
        return None
    return offsets[-1] + (lecture.parts[-1].duration_ms or 0)


@dataclass
class KPLink:
    key: int
    name: str
    chapter: str
    role: str
    others: list[dict[str, Any]] = field(default_factory=list)  # 这个知识点在其他讲的出现


def _kp_links(memory: CourseMemory, course: Course, lecture: Lecture, data: LectureCache) -> dict[int, KPLink]:
    """本讲笔记第几个知识点 → 它在课程知识点体系里的位置。"""
    links: dict[int, KPLink] = {}
    for kp in memory.tree(course):
        mine = [a for a in kp.appearances if a.lecture_id == lecture.id]
        others = [
            {"seq": a.seq, "role": ROLE_LABELS.get(a.role, a.role), "lecture_id": a.lecture_id,
             "t": data.kp_time(a.lecture_id, a.note_index)}
            for a in kp.appearances if a.lecture_id != lecture.id
        ]
        for appearance in mine:
            links[appearance.note_index] = KPLink(kp.key, kp.name, kp.chapter,
                                                  ROLE_LABELS.get(appearance.role, appearance.role), others)
    return links


def _references(memory: CourseMemory, lecture: Lecture, notes: LectureNotes | None, data: LectureCache) -> list[dict]:
    result = []
    for ref in memory.references(lecture.id):
        here = notes.knowledge_points[ref.note_index - 1].name if notes and ref.note_index <= len(
            notes.knowledge_points) else ""
        result.append({
            "relation": RELATION_LABELS.get(ref.relation, ref.relation),
            "t": first_time(ref.segment_ids, data.segments(lecture.id)),
            "here": here,
            "target_seq": ref.target_seq,
            "target_lecture_id": ref.target_lecture_id,
            "target_t": data.kp_time(ref.target_lecture_id, ref.target_note_index),
            "target_name": ref.kp_name,
            "explanation": ref.explanation,
        })
    return result
