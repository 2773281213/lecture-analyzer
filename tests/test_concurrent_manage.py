"""后台任务没结束时照样能管理书架：删除、移动其他讲，追加节，删除其他资料；
只有正在处理、或点名用到那一讲（那份资料）的任务，以及正在整合课程记忆、导入资料的那一会儿才挡住。"""

import threading

import pytest

from lecture_analyzer.course import CourseSettings
from lecture_analyzer.job_handlers import run_job
from lecture_analyzer.jobs import JobStore
from lecture_analyzer.library import Library, LibraryError, PageRef
from lecture_analyzer.locks import hold, lecture_lock, material_lock, materials_lock, memory_lock
from lecture_analyzer.materials import MaterialStore
from lecture_analyzer.pipeline import organize
from lecture_analyzer.pipeline.organize import delete_lecture, delete_materials, move_lecture

from .test_jobs import wait_for
from .test_locks import held_elsewhere


def _db(env):
    return env.cfg.data_dir / "lecture.db"


def test_delete_lecture_while_other_lectures_have_jobs(m1_env):
    with Library(_db(m1_env)) as library:
        course = library.course_by_id(m1_env.course_id)
        third = library.lecture_by_id(m1_env.lecture_id)
        fourth = library.add_lecture(course, 4, [m1_env.media])
        store = JobStore(library)
        other = store.add(course, "run", {"lectures": [fourth.id], "force": []})
        whole = store.add(course, "run", {"lectures": [], "force": []})
        store.pause(whole.id)
        rebuild = store.add(course, "rebuild", {})
        only = store.add(course, "run", {"lectures": [third.id], "force": []})  # 只处理要删的这一讲
        both = store.add(course, "run", {"lectures": [third.id, fourth.id], "force": []})

        result = delete_lecture(library, m1_env.cfg.data_dir, third)

        assert [job.id for job in result.cancelled] == [only.id]
        states = {job.id: store.get(job.id).state for job in (other, whole, rebuild, only, both)}
        assert states == {other.id: "queued", whole.id: "paused", rebuild.id: "queued", only.id: "cancelled",
                          both.id: "queued"}
        assert [lecture.seq for lecture in library.lectures(course)] == [4]
    run_job(m1_env.cfg, _db(m1_env), both.id)  # 处理多讲的任务跳过已删除的讲
    with Library(_db(m1_env)) as library:
        job = JobStore(library).get(both.id)
        assert job.state == "done" and job.result["summary"] == "处理了 1 讲", job.error
        assert library.lecture_by_id(fourth.id).step_state("book") == "done"


def test_lecture_being_processed_cannot_be_deleted_or_moved(m1_env):
    with Library(_db(m1_env)) as library:
        lecture = library.lecture_by_id(m1_env.lecture_id)
        target = library.add_course("病毒学", CourseSettings())
        with held_elsewhere(lecture_lock(m1_env.cfg.data_dir, lecture.id)):
            with pytest.raises(LibraryError, match="第 3 讲正在后台处理.*再删除"):
                delete_lecture(library, m1_env.cfg.data_dir, lecture)
            with pytest.raises(LibraryError, match="第 3 讲正在后台处理.*再移动"):
                move_lecture(library, m1_env.cfg.data_dir, lecture, target)
        assert library.lecture_by_id(lecture.id).course_id == m1_env.course_id
        moved = move_lecture(library, m1_env.cfg.data_dir, lecture, target)  # 处理完就可以了
        assert moved.lecture.course_id == target.id


def test_jobs_that_name_the_lecture_block_deleting_it(m1_env):
    with Library(_db(m1_env)) as library:
        course = library.course_by_id(m1_env.course_id)
        lecture = library.lecture_by_id(m1_env.lecture_id)
        store = JobStore(library)
        blockers = [
            store.add(course, "exam_plan", {"lecture": lecture.id}),
            store.add(course, "report", {"lectures": [lecture.seq], "materials": [], "models": []}),  # 按讲次选讲
            store.add(course, "qa", {"thread": 1, "question": "?", "lecture_id": lecture.id}),
            store.add(course, "import_material", {"paths": ["x.pdf"], "kind": "slides", "pin_lecture": lecture.id}),
        ]
        whole_course_qa = store.add(course, "qa", {"thread": 2, "question": "?", "lecture_id": None})
        for job in blockers:
            with pytest.raises(LibraryError, match=f"第 3 讲正被后台任务用到（#{job.id} {job.label}"):
                delete_lecture(library, m1_env.cfg.data_dir, lecture)
            store.cancel(job.id)
        store.start(whole_course_qa.id)  # 问整门课的问答正在进行：在读这门课的笔记
        with pytest.raises(LibraryError, match=f"#{whole_course_qa.id} 问答"):
            delete_lecture(library, m1_env.cfg.data_dir, lecture)
        store.finish(whole_course_qa.id, {})
        delete_lecture(library, m1_env.cfg.data_dir, lecture)


def test_course_memory_busy_makes_delete_wait_then_refuse(m1_env, monkeypatch):
    monkeypatch.setattr(organize, "UI_WAIT", 0.2)
    with Library(_db(m1_env)) as library:
        lecture = library.lecture_by_id(m1_env.lecture_id)
        with held_elsewhere(memory_lock(m1_env.cfg.data_dir, m1_env.course_id)):  # 任务正在整合课程记忆
            with pytest.raises(LibraryError, match="正在后台整合课程记忆或导出"):
                delete_lecture(library, m1_env.cfg.data_dir, lecture)
        assert library.lecture_by_id(lecture.id)
        delete_lecture(library, m1_env.cfg.data_dir, lecture)


def test_queued_job_waits_for_the_lecture_and_skips_it_once_deleted(m1_env):
    """任务开始处理某一讲时，这一讲正被删除：任务等删除完成，然后跳过它，而不是读写已删除的数据。"""
    with Library(_db(m1_env)) as library:
        course = library.course_by_id(m1_env.course_id)
        job = JobStore(library).add(course, "run", {"lectures": [m1_env.lecture_id], "force": []})
    path = lecture_lock(m1_env.cfg.data_dir, m1_env.lecture_id)
    with hold(path), Library(_db(m1_env)) as library:  # 删除操作拿着讲锁
        worker = threading.Thread(target=run_job, args=(m1_env.cfg, _db(m1_env), job.id))
        worker.start()
        store = JobStore(library)
        wait_for(lambda: "等第 3 讲的其他操作" in store.get(job.id).progress)
        delete_lecture(library, m1_env.cfg.data_dir, library.lecture_by_id(m1_env.lecture_id))
    worker.join(30)
    with Library(_db(m1_env)) as library:
        finished = JobStore(library).get(job.id)
    assert finished.state == "done" and finished.result["summary"] == "要处理的讲已被删除", finished.error


def test_delete_materials_while_other_materials_are_processed(m1_env, monkeypatch):
    monkeypatch.setattr(organize, "UI_WAIT", 0.2)
    data_dir = m1_env.cfg.data_dir
    with Library(_db(m1_env)) as library:
        course = library.course_by_id(m1_env.course_id)
        store = MaterialStore(library)
        first, second, third = (store.add(course, "slides", f"第{n}章", f"第{n}章.pdf") for n in (1, 2, 3))
        jobs = JobStore(library)
        only_first = jobs.add(course, "process_material", {"materials": [first.id]})
        all_materials = jobs.add(course, "process_material", {"materials": []})
        with held_elsewhere(material_lock(data_dir, second.id)):  # 后台正在识别第二份
            result = delete_materials(library, data_dir, course, [first])
            assert [job.id for job in result.cancelled] == [only_first.id]
            assert jobs.get(all_materials.id).state == "queued"
            with pytest.raises(LibraryError, match="「第2章」正在后台处理"):
                delete_materials(library, data_dir, course, [third, second])
        assert [m.id for m in store.list(course)] == [second.id, third.id]  # 拒绝时一份都不删
        with held_elsewhere(materials_lock(data_dir, course.id)):  # 正在导入资料
            with pytest.raises(LibraryError, match="正在后台导入资料"):
                delete_materials(library, data_dir, course, [third])
        delete_materials(library, data_dir, course, [second, third])
        assert store.list(course) == []


def test_jobs_skip_materials_deleted_meanwhile(m1_env):
    with Library(_db(m1_env)) as library:
        course = library.course_by_id(m1_env.course_id)
        store = MaterialStore(library)
        kept, gone = store.add(course, "slides", "留下", "a.pdf"), store.add(course, "slides", "删掉", "b.pdf")
        job = JobStore(library).add(course, "process_material", {"materials": [gone.id]})
        store.remove(gone.id)
        # 匹配课件期间被删除的资料：它的页不写入，其余照常
        store.set_lecture_pages(m1_env.lecture_id, [PageRef(gone.id, 1), PageRef(kept.id, 1)], "auto")
        assert [page.ref for page in store.lecture_pages(m1_env.lecture_id)] == [PageRef(kept.id, 1)]
    run_job(m1_env.cfg, _db(m1_env), job.id)
    with Library(_db(m1_env)) as library:
        finished = JobStore(library).get(job.id)
    assert finished.state == "done" and finished.result["summary"] == "要处理的资料已被删除", finished.error
