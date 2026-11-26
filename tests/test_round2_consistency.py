"""第二轮数据一致性与并发加固的回归测试：移动讲清课件页、费用不串账、status 原子更新、删对话取消任务、跨进程锁。"""

from __future__ import annotations

import threading
import time

import pytest
from typer.testing import CliRunner

from lecture_analyzer.cli import app as cli_app
from lecture_analyzer.conversations import QAStore
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.job_handlers import run_job
from lecture_analyzer.jobs import JobStore
from lecture_analyzer.library import Library, LibraryError, PageRef
from lecture_analyzer.llm import ModelProfile, Usage
from lecture_analyzer.locks import hold, lecture_lock, material_lock, materials_lock, memory_lock
from lecture_analyzer.materials import MaterialStore
from lecture_analyzer.pipeline import qa as qa_module
from lecture_analyzer.pipeline.ingest import material_dir
from lecture_analyzer.pipeline.organize import delete_course, move_lecture
from lecture_analyzer.qa_attachments import mark_used, save_attachment
from lecture_analyzer.schemas import PageDigest

from .test_locks import held_elsewhere
from .test_web import _post
from .test_web_analysis import _client


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


# ---------- T3c：讲和资料的 status 读-改-写是原子的 ----------


def test_concurrent_status_updates_do_not_lose_steps(tmp_path):
    db = tmp_path / "lecture.db"
    with Library(db) as library:
        course = library.add_course("c", CourseSettings())
        lecture = library.add_lecture(course, 1, [tmp_path / "a.wav"])
        store = MaterialStore(library)
        material = store.add(course, "other", "m", "x")
    barrier = threading.Barrier(2)

    def write(tag: str) -> None:
        with Library(db) as library:
            barrier.wait()
            for i in range(50):
                library.set_step(lecture.id, f"{tag}{i}", "done")
                MaterialStore(library).set_step(material.id, f"{tag}{i}", "done")

    threads = [threading.Thread(target=write, args=(tag,)) for tag in "ab"]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with Library(db) as library:
        assert len(library.lecture_by_id(lecture.id).status) == 100
        assert len(MaterialStore(library).get(material.id).status) == 100
        with pytest.raises(LibraryError, match="没有编号为 999 的讲"):
            library.set_step(999, "x", "done")
        assert not library.conn.in_transaction


def test_set_step_still_commits_pending_writes(tmp_path):
    db = tmp_path / "lecture.db"
    with Library(db) as library:
        course = library.add_course("c", CourseSettings())
        lecture = library.add_lecture(course, 1, [tmp_path / "a.wav"])
        store = MaterialStore(library)
        material = store.add(course, "other", "m", "x")
        # 调用方之前有未提交的写：set_step 仍应一并提交（保持原来的提交语义）
        library.conn.execute("UPDATE course SET instructions = ? WHERE id = ?", ("x", course.id))
        library.set_step(lecture.id, "x", "done")
        assert not library.conn.in_transaction
    with Library(db) as reader:
        assert reader.course_by_id(course.id).instructions == "x"
        assert reader.lecture_by_id(lecture.id).step_state("x") == "done"

    with Library(db) as library:
        library.conn.execute("UPDATE course SET instructions = ? WHERE id = ?", ("y", course.id))
        MaterialStore(library).set_step(material.id, "x", "done")
        assert not library.conn.in_transaction
    with Library(db) as reader:
        assert reader.course_by_id(course.id).instructions == "y"
        assert MaterialStore(reader).get(material.id).step_state("x") == "done"


# ---------- T3a：删除对话时取消没答完的问题，任务取消后不再写回 ----------


def test_deleting_a_thread_cancels_its_questions_and_removes_their_attachments(m1_env):
    client = _client(m1_env)
    data_dir = m1_env.cfg.data_dir
    with Library(_db(m1_env)) as library:
        course = library.course_by_id(m1_env.course_id)
        queued_file = save_attachment(data_dir, course.id, "q.txt", b"q")
        failed_file = save_attachment(data_dir, course.id, "f.txt", b"f")
        for item in (queued_file, failed_file):
            mark_used(data_dir, item)
        QAStore(library).add(course, None, 1, "第一问", {"question": "第一问", "answer_md": "答"}, [])
        store = JobStore(library)
        queued = store.add(course, "qa", {"thread": 1, "question": "第二问", "lecture_id": None,
                                          "attachments": [queued_file.summary()]})
        failed = store.add(course, "qa", {"thread": 1, "question": "第三问", "lecture_id": None,
                                          "attachments": [failed_file.summary()]})
        store.start(failed.id)
        store.fail(failed.id, "LLMError")
        other = store.add(course, "qa", {"thread": 2, "question": "别的对话", "lecture_id": None})
    assert _post(client, f"/courses/{m1_env.course_id}/qa/1/delete").status_code == 303
    with Library(_db(m1_env)) as library:
        store = JobStore(library)
        assert store.get(queued.id).state == "cancelled" and store.get(other.id).state == "queued"
    assert not queued_file.path.parent.exists() and not failed_file.path.parent.exists()


def test_answer_finished_after_cancel_is_not_written_back(m1_env, monkeypatch):
    db = _db(m1_env)
    with Library(db) as library:
        course = library.course_by_id(m1_env.course_id)
        job = JobStore(library).add(course, "qa", {"thread": 1, "question": "问", "lecture_id": None})

    def answer(cfg, library, course, question, lecture, history, report, attachments=()):
        JobStore(library).cancel(job.id)  # 回答期间对话被删除、任务被取消
        return {"question": question, "answer_md": "迟到的回答", "models": {"qa": "fake"}}

    monkeypatch.setattr(qa_module, "answer_question", answer)
    run_job(m1_env.cfg, db, job.id)
    with Library(db) as library:
        course = library.course_by_id(m1_env.course_id)
        assert QAStore(library).thread(course, 1) == []
        assert JobStore(library).get(job.id).state == "cancelled"


# ---------- T3b：改课件页和删整门课要与命令行处理互斥 ----------


def test_set_pages_web_and_cli_respect_lecture_lock(m1_env):
    client = _client(m1_env)
    with Library(_db(m1_env)) as library:
        course = library.update_course(m1_env.course_id, slides="provided")
        store = MaterialStore(library)
        material = store.add(course, "slides", "课件", "x.pdf")
        store.set_file(material.id, "materials/x.pdf", 3)
        library.set_step(m1_env.lecture_id, "slides", "done")
    config = ["--config", str(m1_env.cfg.source)]
    with held_elsewhere(lecture_lock(m1_env.cfg.data_dir, m1_env.lecture_id)):
        refused = _post(client, f"/lectures/{m1_env.lecture_id}/pages", material=str(material.id), pages="1-2")
        assert refused.status_code == 400 and "正在后台处理" in refused.text
        cli = CliRunner().invoke(cli_app, ["lecture", "pages", "药理学", "3", "--auto", *config])
        assert cli.exit_code == 1 and "正在后台处理" in cli.output
        with Library(_db(m1_env)) as library:
            assert MaterialStore(library).lecture_pages(m1_env.lecture_id) == []
            assert library.lecture_by_id(m1_env.lecture_id).step_state("slides") == "done"
    assert _post(client, f"/lectures/{m1_env.lecture_id}/pages", material=str(material.id), pages="1-2").status_code == 303


@pytest.mark.parametrize("which", ["lecture", "memory"])
def test_delete_course_waits_for_command_line_work(m1_env, which):
    data_dir = m1_env.cfg.data_dir
    path = lecture_lock(data_dir, m1_env.lecture_id) if which == "lecture" else memory_lock(data_dir, m1_env.course_id)
    with Library(_db(m1_env)) as library:
        course = library.course_by_id(m1_env.course_id)
        with held_elsewhere(path):
            with pytest.raises(LibraryError, match="再删除课程"):
                delete_course(library, data_dir, course)
        assert library.course_by_id(course.id)
        delete_course(library, data_dir, course)
    assert not lecture_lock(data_dir, m1_env.lecture_id).exists()
    assert not memory_lock(data_dir, m1_env.course_id).exists()


def test_delete_course_refuses_when_content_changed_while_waiting(tmp_path):
    """等待资料导入锁期间新导入的资料会被级联删掉记录，但磁盘目录和锁文件会漏删；发现变动就拒绝本次删除。"""
    data_dir = tmp_path / "data"
    db = data_dir / "lecture.db"
    with Library(db) as library:
        course_id = library.add_course("药理学", CourseSettings()).id
    created: list[int] = []
    entered = threading.Event()

    def simulate_import() -> None:
        with hold(materials_lock(data_dir, course_id)):
            entered.set()
            time.sleep(0.4)
            with Library(db) as library:
                material = MaterialStore(library).add(
                    library.course_by_id(course_id), "other", "新导入资料", "new.pdf")
                folder = material_dir(data_dir, material.id)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "new.pdf").write_bytes(b"%PDF-1.4")
                created.append(material.id)

    thread = threading.Thread(target=simulate_import)
    thread.start()
    entered.wait(timeout=5)
    with Library(db) as library:
        with pytest.raises(LibraryError, match="刚有变动"):
            delete_course(library, data_dir, library.course_by_id(course_id))
    thread.join()
    # 第一次删除被拒绝：课程和新导入的资料目录都还在
    with Library(db) as library:
        assert library.course_by_id(course_id)
        assert material_dir(data_dir, created[0]).exists()
        delete_course(library, data_dir, library.course_by_id(course_id))
    # 第二次删除成功：新导入的资料目录和它的锁文件都被清理
    assert not material_dir(data_dir, created[0]).exists()
    assert not material_lock(data_dir, created[0]).exists()
