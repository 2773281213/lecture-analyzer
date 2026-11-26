"""第二轮数据一致性与并发加固的回归测试：移动讲清课件页、费用不串账、status 原子更新、删对话取消任务、跨进程锁。"""

from __future__ import annotations

from lecture_analyzer.course import CourseSettings
from lecture_analyzer.library import Library, PageRef
from lecture_analyzer.llm import ModelProfile, Usage
from lecture_analyzer.materials import MaterialStore
from lecture_analyzer.pipeline.organize import move_lecture
from lecture_analyzer.schemas import PageDigest


def _db(env):
    return env.cfg.data_dir / "lecture.db"


# ---------- T2：移动讲清掉原课程的课件页；删课/删讲后费用不随编号复用串账 ----------


def test_move_lecture_drops_pages_of_the_old_course(m1_env):
    with Library(_db(m1_env)) as library:
        source = library.update_course(m1_env.course_id, slides="provided")
        store = MaterialStore(library)
        material = store.add(source, "slides", "药理课件", "x.pdf")
        store.set_file(material.id, "materials/x.pdf", 2)
        store.save_pages(material.id, [PageDigest(page_no=1, title="t", content_md="c", terms=[])])
        store.set_lecture_pages(m1_env.lecture_id, [PageRef(material.id, 1)], "manual")
        for step in ("media", "transcribe", "slides", "extract", "validate", "integrate", "book"):
            library.set_step(m1_env.lecture_id, step, "done")
        target = library.add_course("病毒学", CourseSettings(subject="药理学"))
        target = library.update_course(target.id, slides="none", exam_plan="skipped")
        result = move_lecture(library, m1_env.cfg.data_dir, library.lecture_by_id(m1_env.lecture_id), target)
        moved = library.lecture_by_id(m1_env.lecture_id)
        assert store.lecture_pages(moved.id) == []
        assert moved.step_state("slides") == "" and moved.step_state("extract") == "done"
        assert result.rerun_from == "slides"  # 原来用了课件页：从匹配课件起重跑
        assert store.remove(material.id) == []  # 删原课程的课件不再牵连移走的讲


def test_costs_of_deleted_courses_and_lectures_do_not_move_to_new_ones(tmp_path):
    profile = ModelProfile(name="p", adapter="openai_compat", model="m", price_in=1.0, price_out=1.0)
    with Library(tmp_path / "lecture.db") as library:
        a = library.add_course("课A", CourseSettings())
        lecture = library.add_lecture(a, 1, [tmp_path / "a.wav"])
        library.log_call(lecture.id, "extract", profile, Usage(1000, 100))  # 讲级费用：course_id 为空
        library.log_call(None, "qa", profile, Usage(2000, 200), course_id=a.id)  # 课级费用
        library.delete_course(a.id)
        b = library.add_course("课B", CourseSettings())
        lb = library.add_lecture(b, 1, [tmp_path / "b.wav"])
        assert (b.id, lb.id) == (a.id, lecture.id)  # 编号确实被复用
        assert library.cost_summary(b) == []
        assert {row["step"] for row in library.cost_summary()} == {"extract", "qa"}  # 总账保留

        second = library.add_lecture(b, 2, [tmp_path / "b2.wav"])
        library.log_call(second.id, "extract", profile, Usage(5, 5))
        library.delete_lecture(second.id)
        assert [row["input_tokens"] for row in library.cost_summary(b)] == [5]  # 删讲后这笔费用仍算在课程上
        c = library.add_course("课C", CourseSettings())
        lc = library.add_lecture(c, 1, [tmp_path / "c.wav"])
        assert lc.id == second.id and library.cost_summary(c) == []


def test_legacy_orphan_costs_are_detached_when_ids_are_reused(tmp_path):
    profile = ModelProfile(name="p", adapter="openai_compat", model="m")
    with Library(tmp_path / "lecture.db") as library:
        # 模拟旧版本删课后留下的记录：编号还指着已经不存在的课程和讲
        library.conn.execute(
            "INSERT INTO llm_call (lecture_id, course_id, step, model_profile, model, input_tokens,"
            " output_tokens, cache_read_tokens, cost_usd, created_at)"
            " VALUES (1, 1, 'qa', 'p', 'm', 1, 1, 0, NULL, 'x')")
        library.conn.commit()
        course = library.add_course("新课", CourseSettings())
        library.add_lecture(course, 1, [tmp_path / "a.wav"])
        assert course.id == 1 and library.cost_summary(course) == []
        assert len(library.cost_summary()) == 1
