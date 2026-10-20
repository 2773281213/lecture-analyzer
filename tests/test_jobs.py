"""后台任务：任务表的状态变化、子进程执行器（两条队列、取消、崩溃、关闭），以及各类任务的处理。"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lecture_analyzer.config import load_config
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.job_handlers import run_job
from lecture_analyzer.job_runner import JobRunner, log_path
from lecture_analyzer.jobs import Job, JobStore, ensure_idle
from lecture_analyzer.library import Library, LibraryError
from lecture_analyzer.pipeline import ingest
from lecture_analyzer.pipeline.organize import delete_course

from .conftest import LECTURE_NOTES, FakeProvider, integration_json
from .test_materials import _answer, _digest_json, _pdf

STUB = Path(__file__).with_name("job_stub.py")


def wait_for(check, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("等待超时")


# ---------- 任务表 ----------


def test_job_states_only_move_forward(tmp_path):
    with Library(tmp_path / "lecture.db") as library:
        course = library.add_course("病毒学", CourseSettings())
        store = JobStore(library)
        heavy = store.add(course, "run", {"lectures": [1], "force": []})
        light = store.add(course, "report", {"models": ["claude"]})
        models = store.add(None, "download_models", {"spec": "funasr:paraformer"})
        assert store.active(course, "run", {"force": [], "lectures": [1]}).id == heavy.id  # 参数顺序不影响
        assert store.active(course, "run", {"lectures": [2], "force": []}) is None
        assert store.active(None, "download_models", {"spec": "funasr:paraformer"}).id == models.id
        assert store.next_queued("heavy").id == heavy.id and store.next_queued("light").id == light.id
        assert store.start(heavy.id) and not store.start(heavy.id)
        assert store.next_queued("heavy").id == models.id
        assert store.cancel(heavy.id) and not store.cancel(heavy.id)
        store.report(heavy.id, "取消后的进度")
        assert not store.finish(heavy.id, {"url": "/x"})  # 取消之后子进程写回的进度和结果都不生效
        job = store.get(heavy.id)
        assert (job.state, job.progress, job.result) == ("cancelled", "开始…", {})  # 保留取消前的进度
        assert [j.id for j in store.active_jobs(course)] == [light.id]
        with pytest.raises(LibraryError, match="还有没结束的后台任务"):
            ensure_idle(library, course, "删除课程")
        with pytest.raises(LibraryError, match="还有没结束的后台任务"):
            delete_course(library, tmp_path, course)
        with pytest.raises(LibraryError, match="未知的任务类型"):
            store.add(course, "everything", {})
        assert store.fail_interrupted() == 2 and store.active_jobs() == []
        assert "被关闭" in store.get(light.id).error
        ensure_idle(library, course, "删除课程")


# ---------- 子进程执行器 ----------


@pytest.fixture
def runner_env(example_config: Path):
    cfg = load_config(example_config)
    db = cfg.data_dir / "lecture.db"
    with Library(db) as library:
        course = library.add_course("病毒学", CourseSettings())
    runner = JobRunner(cfg, db, command=lambda job_id: [sys.executable, str(STUB), str(db), str(job_id)])
    runner.start()
    yield SimpleNamespace(cfg=cfg, db=db, course_id=course.id, runner=runner)
    runner.shutdown()


def _submit(env, kind: str, **params) -> int:
    with Library(env.db) as library:
        job = JobStore(library).add(library.course_by_id(env.course_id), kind, params)
    env.runner.submit(job.id)
    return job.id


def _job(env, job_id: int) -> Job:
    with Library(env.db) as library:
        return JobStore(library).get(job_id)


def _finished(env, job_id: int) -> Job | None:
    job = _job(env, job_id)
    return job if job.finished else None


def test_light_lane_is_not_blocked_and_heavy_lane_keeps_order(runner_env):
    first = _submit(runner_env, "run", sleep=6)
    second = _submit(runner_env, "rebuild")  # 同一条队列，排在 first 后面
    light = _submit(runner_env, "review")
    wait_for(lambda: _job(runner_env, light).state == "done")
    assert _job(runner_env, first).state == "running" and _job(runner_env, second).state == "queued"
    wait_for(lambda: _job(runner_env, second).state == "done")
    done = _job(runner_env, first)
    assert done.state == "done" and done.result == {"url": "/done"}
    assert not log_path(runner_env.cfg.data_dir, first).exists()  # 成功的任务不留日志


def test_cancel_running_job_kills_its_process(runner_env):
    job_id = _submit(runner_env, "run", sleep=60)
    wait_for(lambda: _job(runner_env, job_id).progress == "子进程已启动")
    process = runner_env.runner._processes[job_id]
    started = time.monotonic()
    assert runner_env.runner.cancel(job_id)
    assert process.poll() is not None and time.monotonic() - started < 15
    assert _job(runner_env, job_id).state == "cancelled"
    wait_for(lambda: not runner_env.runner.running())
    assert not runner_env.runner.cancel(job_id)  # 已经结束


def test_cancelled_queued_job_never_starts(runner_env):
    blocker = _submit(runner_env, "run", sleep=2)
    queued = _submit(runner_env, "rebuild", outcome="crash")
    assert runner_env.runner.cancel(queued)
    wait_for(lambda: _finished(runner_env, blocker))
    time.sleep(1)  # 给队列线程时间取下一个任务
    job = _job(runner_env, queued)
    assert (job.state, job.progress, job.error) == ("cancelled", "", "")


def test_crashed_worker_is_marked_failed_with_log_tail(runner_env):
    job_id = _submit(runner_env, "review", outcome="crash")
    job = wait_for(lambda: _finished(runner_env, job_id))
    assert job.state == "failed" and "退出码 3" in job.error and "模拟崩溃" in job.error
    assert log_path(runner_env.cfg.data_dir, job_id).is_file()  # 失败的任务保留日志
    failed = _submit(runner_env, "review", outcome="fail")
    assert wait_for(lambda: _finished(runner_env, failed)).error == "模拟失败"


def test_shutdown_stops_running_jobs(runner_env):
    job_id = _submit(runner_env, "run", sleep=60)
    wait_for(lambda: _job(runner_env, job_id).progress == "子进程已启动")
    runner_env.runner.shutdown()
    job = _job(runner_env, job_id)
    assert job.state == "failed" and "被关闭" in job.error


def test_real_worker_process(example_config: Path):
    """真实的 python -m lecture_analyzer.worker：一个能完成的任务和一个被课件关卡挡住的任务。"""
    cfg = load_config(example_config)
    db = cfg.data_dir / "lecture.db"
    runner = JobRunner(cfg, db)
    runner.start()  # 先成为执行器（启动时会把表里没结束的任务当成上次留下的），再提交任务
    with Library(db) as library:
        store = JobStore(library)
        ready = library.add_course("病毒学", CourseSettings())
        library.update_course(ready.id, slides="none")
        blocked = library.add_course("药理学", CourseSettings())
        jobs = [store.add(ready, "run", {}), store.add(blocked, "run", {})]
    runner.submit(jobs[0].id)
    try:
        env = SimpleNamespace(db=db)
        done, failed = (wait_for(lambda job=job: _finished(env, job.id), timeout=90) for job in jobs)
    finally:
        runner.shutdown()
    assert done.state == "done" and done.result["summary"] == "所有讲都已处理完成"
    assert failed.state == "failed" and "是否提供课件" in failed.error


# ---------- 各类任务的处理（在当前进程里执行，假模型才生效） ----------


def _run_inline(env, kind: str, params: dict, course_id: int | None = None) -> Job:
    db = env.cfg.data_dir / "lecture.db"
    with Library(db) as library:
        course = library.course_by_id(course_id or env.course_id)
        job = JobStore(library).add(course, kind, params)
    run_job(env.cfg, db, job.id)
    with Library(db) as library:
        return JobStore(library).get(job.id)


def test_run_and_rebuild_jobs(m1_env, monkeypatch):
    seen: list[str] = []
    original = JobStore.report
    monkeypatch.setattr(JobStore, "report", lambda self, job_id, message: (seen.append(message),
                                                                            original(self, job_id, message)))
    job = _run_inline(m1_env, "run", {"lectures": [m1_env.lecture_id]})
    assert job.state == "done", job.error
    assert job.result == {"url": f"/lectures/{m1_env.lecture_id}", "summary": "处理了 1 讲"}
    assert "第 3 讲 · 开始「转写」…" in seen and "第 3 讲 · 整合课程记忆：第 3 讲…" in seen
    assert _run_inline(m1_env, "run", {}).result["summary"] == "所有讲都已处理完成"

    assert _run_inline(m1_env, "rebuild", {}).result["summary"] == "课程记忆已是最新，不需要重建"
    m1_env.routes["IntegrationResult"] = [integration_json(LECTURE_NOTES, chapter="药理学总论")]
    rebuilt = _run_inline(m1_env, "rebuild", {"from_seq": 3})
    assert rebuilt.result["summary"] == "重放 0 讲、重新整合 1 讲，已重新导出"
    forced = _run_inline(m1_env, "run", {"lectures": [m1_env.lecture_id], "force": ["validate"]})
    assert forced.state == "done" and m1_env.asr.calls == 1  # 只重跑校验及其后的步骤，不重新转写


def test_material_and_model_jobs(m1_env, monkeypatch):
    pdf = _pdf(m1_env.tmp_path / "第二章.pdf", 2)
    imported = _run_inline(m1_env, "import_material", {"paths": [str(pdf)], "kind": "slides", "process": False})
    assert imported.state == "done", imported.error
    assert imported.result["url"].startswith("/materials/") and imported.result["summary"] == "导入了 1 份资料"
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        assert library.course_by_id(m1_env.course_id).slides == "provided"  # 导入课件即视为提供课件
    missing = _run_inline(m1_env, "import_material", {"paths": [str(m1_env.tmp_path / "没有.pdf")], "kind": "slides"})
    assert missing.state == "failed" and "没有找到可以导入的文档" in missing.error
    def fake_provider(profile):  # 逐页识别
        provider = FakeProvider([], profile.model_copy(update={"model": "fake-model"}))
        provider._complete = lambda parts, **kw: _answer(parts, {(1, 2): _digest_json([1, 2])})
        return provider

    monkeypatch.setattr(ingest, "create_provider", fake_provider)
    processed = _run_inline(m1_env, "process_material", {"materials": []})  # 继续处理所有没处理完的资料
    assert processed.state == "done", processed.error
    assert processed.result["url"] == f"/courses/{m1_env.course_id}#materials"

    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        job = JobStore(library).add(None, "download_models", {"spec": "llm:gemini"})
    run_job(m1_env.cfg, m1_env.cfg.data_dir / "lecture.db", job.id)
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        downloaded = JobStore(library).get(job.id)
    assert downloaded.state == "done" and downloaded.result["summary"] == "模型都已下载"


def test_one_runner_per_library(runner_env):
    """同一个书库只有一个执行器：后打开的只提交任务，由先打开的执行；在后打开的那里取消，执行器结束进程。"""
    second = JobRunner(runner_env.cfg, runner_env.db,
                       command=lambda job_id: [sys.executable, str(STUB), str(runner_env.db), str(job_id)])
    second.start()
    try:
        assert runner_env.runner.owner and not second.owner and second.running() == []
        with Library(runner_env.db) as library:
            job = JobStore(library).add(library.course_by_id(runner_env.course_id), "run", {"sleep": 60})
        second.submit(job.id)  # 只唤醒自己（没有队列线程），由第一个执行器从任务表里取走
        wait_for(lambda: _job(runner_env, job.id).progress == "子进程已启动")
        process = runner_env.runner._processes[job.id]
        assert second.cancel(job.id)  # 只改任务表
        wait_for(lambda: process.poll() is not None, timeout=15)
        assert _job(runner_env, job.id).state == "cancelled"
    finally:
        second.shutdown()
    runner_env.runner.shutdown()
    third = JobRunner(runner_env.cfg, runner_env.db)
    third.start()  # 先打开的关掉之后，锁就释放了
    assert third.owner
    third.shutdown()
