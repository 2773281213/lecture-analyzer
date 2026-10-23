"""整理书架：把放错课程的一讲移到正确的课程，删除讲或整门课（命令行和网页共用）。

删除只删书库里的数据、导出文件、中间文件和资料副本，原始音视频不动。
课程还有没结束的后台任务时不能移动或删除（任务进程正在读写这些数据）。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ..books import course_dir, export_course, lecture_dir, work_dir
from ..exam_plans import ExamPlanStore, plan_block
from ..jobs import ensure_idle
from ..knowledge import CourseMemory
from ..library import Course, Lecture, Library, LibraryError
from ..materials import MaterialStore
from .ingest import material_dir
from .integrate import restore_prefix


@dataclass
class MoveResult:
    lecture: Lecture
    rerun_from: str  # 下次运行时从哪一步起重跑
    stale: list[Lecture]  # 两门课里因此标记为待更新的讲


def rerun_step_for_move(source: Course, target: Course) -> str:
    """移到设置不同的课程后，从哪一步起需要重跑：授课方式不同重新转写；学科或提示词不同重新整理笔记；
    设置都相同只需重新整合课程记忆。"""
    a, b = source.settings, target.settings
    if (a.teaching, a.interpreted_to) != (b.teaching, b.interpreted_to):
        return "transcribe"
    if (a.subject, source.instructions) != (b.subject, target.instructions):
        return "extract"
    return "integrate"


def _plan_text(library: Library, course: Course) -> str:
    """整理笔记时用的考试方案文字（没有已确认的方案时为空）。"""
    return plan_block(ExamPlanStore(library).active(course), "extract")


def rerun_step_for_settings(old: Course, new: Course) -> str | None:
    """改了课程设置后，已处理的讲要从哪一步起重做（转写方案或授课方式变了重新转写；学科或提示词变了重新整理笔记）；
    不用重做时返回 None。"""
    if old.asr != new.asr:
        return "transcribe"
    step = rerun_step_for_move(old, new)
    return None if step == "integrate" else step


def next_free_seq(library: Library, course: Course) -> int:
    return max((lecture.seq for lecture in library.lectures(course)), default=0) + 1


def move_lecture(
    library: Library, data_dir: Path, lecture: Lecture, target: Course, seq: int | None = None
) -> MoveResult:
    """把一讲移到 target 课程的第 seq 讲（不给时沿用原讲次，冲突时排到最后）。

    两门课的课程记忆都回到移动位置之前的一致状态，之后整合过的讲标记为待更新；导出文件夹随之移动。
    """
    source = library.course_by_id(lecture.course_id)
    if source.id == target.id:
        raise LibraryError(f"第 {lecture.seq} 讲已经在「{target.name}」里了")
    ensure_idle(library, source, "移动讲")
    ensure_idle(library, target, "移动讲")
    if seq is None:
        taken = {other.seq for other in library.lectures(target)}
        seq = lecture.seq if lecture.seq not in taken else next_free_seq(library, target)
    old_folder = lecture_dir(data_dir, source, lecture)
    memory = CourseMemory(library)
    library.move_lecture(lecture.id, target, seq)
    memory.drop_lecture(lecture.id)  # 旧课程里的整合日志作废
    rerun_from = rerun_step_for_move(source, target)
    if rerun_from == "integrate" and _plan_text(library, source) != _plan_text(library, target):
        rerun_from = "extract"  # 两门课的考试方案不同：笔记要按新课程的方案重新整理
    library.reset_steps(lecture.id, [rerun_from, "integrate", "book"])
    stale = restore_prefix(library, memory, source, lecture.seq) + restore_prefix(library, memory, target, seq)
    moved = library.lecture_by_id(lecture.id)
    new_folder = lecture_dir(data_dir, target, moved)
    if old_folder.is_dir() and not new_folder.exists():
        new_folder.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_folder), str(new_folder))
    export_course(library, data_dir, source)
    export_course(library, data_dir, target)
    return MoveResult(moved, rerun_from, stale)


def delete_lecture(library: Library, data_dir: Path, lecture: Lecture) -> list[Lecture]:
    """删除一讲；课程记忆回到这一讲之前的一致状态，其后整合过的讲标记为待更新并返回。"""
    course = library.course_by_id(lecture.course_id)
    ensure_idle(library, course, "删除讲")
    folder = lecture_dir(data_dir, course, lecture)
    memory = CourseMemory(library)
    memory.drop_lecture(lecture.id)
    library.delete_lecture(lecture.id)
    stale = restore_prefix(library, memory, course, lecture.seq)
    shutil.rmtree(folder, ignore_errors=True)
    shutil.rmtree(work_dir(data_dir, lecture.id), ignore_errors=True)
    export_course(library, data_dir, course)
    return stale


def delete_course(library: Library, data_dir: Path, course: Course) -> None:
    """删除整门课：书库数据、导出的书、各讲中间文件和资料副本。"""
    ensure_idle(library, course, "删除课程")
    lectures = library.lectures(course)
    materials = MaterialStore(library).list(course)
    library.delete_course(course.id)
    for lecture in lectures:
        shutil.rmtree(work_dir(data_dir, lecture.id), ignore_errors=True)
    for material in materials:
        shutil.rmtree(material_dir(data_dir, material.id), ignore_errors=True)
    shutil.rmtree(course_dir(data_dir, course), ignore_errors=True)
