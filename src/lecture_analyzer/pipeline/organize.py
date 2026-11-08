"""整理书架：把放错课程的一讲移到正确的课程，删除讲或整门课（命令行和网页共用）。

删除只删书库里的数据、导出文件、中间文件和资料副本，原始音视频不动。
后台任务在处理这门课的其他讲时，也可以删除或移动一讲：只有正在处理这一讲、点名用到这一讲的任务，
以及正在整合课程记忆或导出整门课的那几分钟会挡住（见 locks.py、jobs.lecture_blockers）；
删除整门课仍要等这门课的任务都结束。
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..books import course_dir, export_course, lecture_dir, work_dir
from ..exam_plans import ExamPlanStore, plan_block
from ..jobs import (
    Job, cancel_orphans, ensure_idle, ensure_unblocked, lecture_blockers, material_blockers, reading_jobs,
)
from ..knowledge import CourseMemory
from ..library import Course, Lecture, Library, LibraryError
from ..locks import UI_WAIT, discard, hold, lecture_lock, material_lock, materials_lock, memory_lock
from ..materials import Material, MaterialStore
from .ingest import material_dir, remove_material
from .integrate import restore_prefix


@dataclass
class MoveResult:
    lecture: Lecture
    rerun_from: str  # 下次运行时从哪一步起重跑
    stale: list[Lecture]  # 两门课里因此标记为待更新的讲


@dataclass
class DeleteResult:
    stale: list[Lecture]  # 因此标记为课程记忆待更新的讲
    cancelled: list[Job]  # 只处理这一讲、还没开始或已暂停的任务，已随之取消


@contextmanager
def lecture_free(library: Library, data_dir: Path, lecture: Lecture, action: str) -> Iterator[None]:
    """拿着这一讲的讲锁执行：此刻没有任务在处理这一讲，也没有别的任务点名用到它；否则说明原因。"""
    busy = f"第 {lecture.seq} 讲正在后台处理，请等这一讲处理完，或在「后台任务」里暂停或取消后再{action}"
    with hold(lecture_lock(data_dir, lecture.id), wait=0, busy=busy):
        ensure_unblocked(lecture_blockers(library, lecture), f"第 {lecture.seq} 讲", action)
        yield


def memory_free(data_dir: Path, course: Course, action: str) -> AbstractContextManager[None]:
    """拿着这门课的课程记忆锁执行：后台任务正在整合课程记忆或导出整门课时，等几秒还不行就说明原因。"""
    return hold(memory_lock(data_dir, course.id), wait=UI_WAIT,
                busy=f"「{course.name}」正在后台整合课程记忆或导出，请过一会儿再{action}")


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
    with (lecture_free(library, data_dir, lecture, "移动"), memory_free(data_dir, source, "移动讲"),
          memory_free(data_dir, target, "移动讲")):
        ensure_unblocked(reading_jobs(library, target), f"「{target.name}」", "把讲移进来")
        return _move(library, data_dir, lecture, source, target, seq)


def _move(library: Library, data_dir: Path, lecture: Lecture, source: Course, target: Course,
          seq: int | None) -> MoveResult:
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


def delete_lecture(library: Library, data_dir: Path, lecture: Lecture) -> DeleteResult:
    """删除一讲；课程记忆回到这一讲之前的一致状态，其后整合过的讲标记为待更新。
    只处理这一讲、还没开始或已暂停的任务随之取消；处理多讲的任务到这一讲时跳过。"""
    course = library.course_by_id(lecture.course_id)
    with lecture_free(library, data_dir, lecture, "删除"), memory_free(data_dir, course, "删除讲"):
        cancelled = cancel_orphans(library, course, "run", "lectures", lecture.id)
        folder = lecture_dir(data_dir, course, lecture)
        memory = CourseMemory(library)
        memory.drop_lecture(lecture.id)
        library.delete_lecture(lecture.id)
        stale = restore_prefix(library, memory, course, lecture.seq)
        shutil.rmtree(folder, ignore_errors=True)
        shutil.rmtree(work_dir(data_dir, lecture.id), ignore_errors=True)
        export_course(library, data_dir, course)
    discard(lecture_lock(data_dir, lecture.id))
    return DeleteResult(stale, cancelled)


@dataclass
class MaterialsDeleted:
    affected: set[int]  # 用到这些资料的讲（编号）：下次处理时重新匹配课件
    cancelled: list[Job]  # 只处理其中一份、还没开始或已暂停的任务，已随之取消


def delete_materials(library: Library, data_dir: Path, course: Course, materials: list[Material]) -> MaterialsDeleted:
    """删除这门课的几份资料（书库里的副本、识别结果和题目；原文件不受影响）。
    后台任务在处理别的讲和资料时也可以删除；其中某份正在识别、拆题或对应知识点，这门课正在导入资料，
    或有任务点名用到它们时不行。"""
    with ExitStack() as stack:
        for material in materials:
            busy = (f"「{material.title}」正在后台处理（识别、拆题或对应知识点），请等它处理完，"
                    "或在「后台任务」里暂停或取消后再删除")
            stack.enter_context(hold(material_lock(data_dir, material.id), wait=0, busy=busy))
            ensure_unblocked(material_blockers(library, course, material.id), f"「{material.title}」", "删除")
        stack.enter_context(hold(materials_lock(data_dir, course.id), wait=UI_WAIT,
                                 busy=f"「{course.name}」正在后台导入资料，请等导入完再删除资料"))
        store = MaterialStore(library)
        result = MaterialsDeleted(set(), [])
        for material in materials:
            result.cancelled += cancel_orphans(library, course, "process_material", "materials", material.id)
            result.affected.update(store.lectures_using(material.id))
            remove_material(library, data_dir, material.id)
    for material in materials:
        discard(material_lock(data_dir, material.id))
    return result


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
