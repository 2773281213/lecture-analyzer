import json

from fastapi.testclient import TestClient

from lecture_analyzer.job_handlers import run_job
from lecture_analyzer.jobs import JobStore
from lecture_analyzer.library import Library
from lecture_analyzer.materials import MaterialStore
from lecture_analyzer.pipeline import report as report_module
from lecture_analyzer.pipeline import review as review_module
from lecture_analyzer.reports import ReportStore
from lecture_analyzer.pipeline.ingest import import_materials
from lecture_analyzer.web import create_app

from .conftest import RoutingProvider
from .test_materials import _pdf
from .test_report import _prepare
from .test_web import BASE, _post


class InlineRunner:
    """测试里同步执行后台任务（不启动子进程，假模型才生效）。"""

    def __init__(self, cfg, db_path):
        self.cfg, self.db_path = cfg, db_path

    def start(self):
        pass

    def submit(self, job_id):
        run_job(self.cfg, self.db_path, job_id)

    def cancel(self, job_id):
        with Library(self.db_path) as library:
            return JobStore(library).cancel(job_id)

    def shutdown(self):
        pass


def _client(env) -> TestClient:
    app = create_app(env.cfg)
    app.state.jobs = InlineRunner(env.cfg, env.cfg.data_dir / "lecture.db")
    return TestClient(app, base_url=BASE, follow_redirects=False)


def _fake_report_models(monkeypatch):
    ranking = {"ranking": [{"kp": "K1", "reason": "老师说期末一定会考", "question_types": ["简答题"],
                            "evidence": ["Q1", "M1"]}]}
    review = {"overview_md": "总论", "entries": [{"kp": "K1", "summary_md": "先背三段剂量", "sources": [],
                                                 "emphasis": [], "examples": []}]}
    routes = {"KPRanking": [json.dumps(ranking, ensure_ascii=False)],
              "ReviewChapter": [json.dumps(review, ensure_ascii=False)]}
    fake = lambda profile: RoutingProvider(routes, profile.model_copy(update={"model": "fake-model"}))  # noqa: E731
    monkeypatch.setattr(report_module, "create_provider", fake)
    monkeypatch.setattr(review_module, "create_provider", fake)


def test_report_job_from_course_page(m1_env, monkeypatch):
    _prepare(m1_env, monkeypatch)
    _fake_report_models(monkeypatch)
    client = _client(m1_env)
    page = client.get(f"/courses/{m1_env.course_id}")
    assert "考前分析" in page.text and "模拟题（样例）" in page.text and 'name="lecture" value="3"' in page.text

    response = client.post(f"/courses/{m1_env.course_id}/report",
                           data={"csrf": client.app.state.csrf_token, "lecture": "3", "model": "claude"})
    assert response.status_code == 303 and response.headers["location"].startswith("/jobs/")
    job_id = int(response.headers["location"].rsplit("/", 1)[1])
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["state"] == "done" and status["url"].startswith("/reports/")
    assert "查看结果" in client.get(f"/jobs/{job_id}").text

    report = client.get(status["url"])
    assert report.status_code == 200
    for text in ("1. 阿司匹林的剂量分段", "先背三段剂量", "可能题型：简答题", "在课程里查看出处", "打印 / 另存为 PDF"):
        assert text in report.text
    for text in ("这个期末一定会考", "共识分", "参考分"):  # 不显示推断过程
        assert text not in report.text
    download = client.get(status["url"] + "/markdown")
    assert download.status_code == 200 and "attachment" in download.headers["content-disposition"]
    assert "## 1. 阿司匹林的剂量分段" in download.text
    course_page = client.get(f"/courses/{m1_env.course_id}").text
    assert status["url"] in course_page and "最近的后台任务" in course_page


def test_legacy_report_and_delete(m1_env):
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        old = {"course": "药理学", "generated_at": "2026-10-15T18:22:17", "models": [], "warnings": [],
               "scope": {"lectures": [], "materials": []}, "priorities": [{"rank": 1, "name": "阿司匹林的剂量分段"}]}
        report_id = ReportStore(library).add(course, "考前报告 2026-10-15 18:22", {}, old, ["claude"])
    client = _client(m1_env)
    page = client.get(f"/reports/{report_id}").text
    assert "这是旧版报告" in page and "阿司匹林的剂量分段" in page  # M3 的旧报告只显示考点名称
    assert "这是旧版报告" in client.get(f"/reports/{report_id}/markdown").text
    deleted = _post(client, f"/reports/{report_id}/delete")
    assert deleted.status_code == 303 and deleted.headers["location"] == f"/courses/{m1_env.course_id}#analysis"
    assert "已删除「考前报告 2026-10-15 18:22」" in client.get(deleted.headers["location"]).text
    assert client.get(f"/reports/{report_id}").status_code == 404


def test_report_form_is_validated_before_queueing(m1_env, monkeypatch):
    _prepare(m1_env, monkeypatch)
    client = _client(m1_env)
    bad_model = _post(client, f"/courses/{m1_env.course_id}/report", model="不存在的档案")
    assert bad_model.status_code == 400 and "不存在的档案" in bad_model.text
    bad_lecture = _post(client, f"/courses/{m1_env.course_id}/report", lecture="9")
    assert bad_lecture.status_code == 400 and "没有第 9 讲" in bad_lecture.text
    assert client.post(f"/courses/{m1_env.course_id}/report", data={"lecture": "3"}).status_code == 403  # 没有令牌
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        assert JobStore(library).recent(library.course_by_id(m1_env.course_id)) == []


def test_failed_job_shows_error_and_review_job(m1_env, monkeypatch):
    _prepare(m1_env, monkeypatch)
    monkeypatch.setattr(report_module, "create_provider", lambda profile: (_ for _ in ()).throw(RuntimeError("断网")))
    client = _client(m1_env)
    failed = _post(client, f"/courses/{m1_env.course_id}/report")
    status = client.get(f"/api/jobs/{failed.headers['location'].rsplit('/', 1)[1]}").json()
    assert status["state"] == "failed" and "所有模型都没能给出排名" in status["error"]

    review = {"overview_md": "本章概览", "entries": [{"kp": "K1", "summary_md": "要点", "sources": [], "emphasis": [],
                                                    "examples": []}]}
    monkeypatch.setattr(review_module, "create_provider", lambda profile: RoutingProvider(
        {"ReviewChapter": [json.dumps(review, ensure_ascii=False)]}, profile.model_copy(update={"model": "fake"})))
    done = _post(client, f"/courses/{m1_env.course_id}/review/generate")
    status = client.get(f"/api/jobs/{done.headers['location'].rsplit('/', 1)[1]}").json()
    assert status["state"] == "done" and status["url"] == f"/courses/{m1_env.course_id}/review"
    assert "本章概览" in client.get(status["url"]).text


def test_interrupted_jobs_are_failed_on_startup(m1_env):
    from lecture_analyzer.job_runner import JobRunner

    db = m1_env.cfg.data_dir / "lecture.db"
    with Library(db) as library:
        job = JobStore(library).add(library.course_by_id(m1_env.course_id), "review", {})
    with TestClient(create_app(m1_env.cfg), base_url=BASE):  # 启动时成为执行器，先清理上次没执行完的任务
        pass
    with Library(db) as library:
        stored = JobStore(library).get(job.id)
    assert stored.state == "failed" and "被关闭" in stored.error
    assert isinstance(create_app(m1_env.cfg).state.jobs, JobRunner)


def test_material_page_and_page_images(m1_env):
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        material = import_materials(library, m1_env.cfg.data_dir, course, [_pdf(m1_env.tmp_path / "第二章.pdf", 2)],
                                    "slides", report=lambda _: None).materials[0]
        MaterialStore(library).set_lecture_pages(m1_env.lecture_id, [], "auto")
    client = _client(m1_env)
    page = client.get(f"/materials/{material.id}")
    assert page.status_code == 200 and "第二章" in page.text and "继续处理" in page.text  # 还没识别页面
    image = client.get(f"/materials/{material.id}/pages/1.png")
    assert image.status_code == 200 and image.content.startswith(b"\x89PNG")
    assert client.get(f"/materials/{material.id}/pages/3.png").status_code == 404
    assert client.get("/materials/999").status_code == 404
    assert "第二章" in client.get(f"/courses/{m1_env.course_id}").text


def test_jobs_api_and_cancel(m1_env):
    client = _client(m1_env)  # 先启动：启动时会把没结束的任务标为中断
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        job = JobStore(library).add(library.course_by_id(m1_env.course_id), "rebuild", {})
    listing = client.get("/api/jobs").json()
    assert listing["active"] == 1 and listing["jobs"][0]["label"] == "重建课程记忆" and listing["jobs"][0]["active"]
    assert client.post(f"/jobs/{job.id}/cancel", data={}).status_code == 403  # 没有令牌
    cancelled = _post(client, f"/jobs/{job.id}/cancel")
    assert cancelled.status_code == 303 and cancelled.headers["location"] == f"/jobs/{job.id}"
    status = client.get(f"/api/jobs/{job.id}").json()
    assert (status["state"], status["state_label"], status["active"]) == ("cancelled", "已取消", False)
    assert client.get("/api/jobs").json()["active"] == 0
    assert _post(client, "/jobs/999/cancel").status_code == 400


def test_job_pause_resume_retry_buttons(m1_env):
    """单个任务的暂停 / 继续 / 重试，以及任务页顶部的全部暂停 / 全部继续 / 全部重试。
    执行器不启动（只改任务表、不执行），看得清每一步的状态变化。"""
    from lecture_analyzer.job_runner import JobRunner

    db = m1_env.cfg.data_dir / "lecture.db"
    app = create_app(m1_env.cfg)
    app.state.jobs = JobRunner(m1_env.cfg, db, command=lambda job_id: ["unused"])
    client = TestClient(app, base_url=BASE, follow_redirects=False)
    with Library(db) as library:
        course = library.course_by_id(m1_env.course_id)
        store = JobStore(library)
        first = store.add(course, "rebuild", {})
        second = store.add(course, "process_material", {"materials": [1]})
        failed = store.add(course, "review", {"all": True})
        store.fail(failed.id, "模型请求失败")

    page = client.get("/jobs").text
    assert "全部暂停（2）" in page and "全部重试（1）" in page and "全部继续" not in page
    assert f'action="/jobs/{first.id}/pause"' in page and f'action="/jobs/{failed.id}/retry"' in page

    paused = _post(client, f"/jobs/{first.id}/pause", back="jobs")
    assert paused.status_code == 303 and paused.headers["location"] == "/jobs"
    assert client.get(f"/api/jobs/{first.id}").json()["state_label"] == "已暂停"
    job_page = client.get(f"/jobs/{first.id}").text
    assert f'action="/jobs/{first.id}/resume"' in job_page and "取消任务" in job_page
    assert _post(client, f"/jobs/{first.id}/resume").headers["location"] == f"/jobs/{first.id}"
    assert client.get(f"/api/jobs/{first.id}").json()["state"] == "queued"

    assert _post(client, "/jobs/pause-all").headers["location"] == "/jobs"
    page = client.get("/jobs").text
    assert "已暂停 2 个任务" in page and "全部继续（2）" in page and "全部暂停" not in page
    _post(client, "/jobs/resume-all")
    assert client.get("/api/jobs").json()["active"] == 2

    retried = _post(client, f"/jobs/{failed.id}/retry")
    with Library(db) as library:
        new = JobStore(library).latest(1)[0]
    assert retried.headers["location"] == f"/jobs/{new.id}"
    assert (new.kind, new.params, new.state) == ("review", {"all": True}, "queued")
    assert _post(client, f"/jobs/{failed.id}/retry").headers["location"] == f"/jobs/{new.id}"  # 同样的任务已在排队
    with Library(db) as library:
        assert JobStore(library).latest(1)[0].id == new.id  # 没有重复提交
    _post(client, "/jobs/retry-all")
    assert "没有需要重试的任务" in client.get("/jobs").text
    assert _post(client, "/jobs/999/pause").status_code == 400


def test_report_shows_must_points_and_folds_the_rest(m1_env):
    """考前报告只展开 ★★★ 必考考点，其余考点和不考的知识点折叠在后面。"""
    def point(rank, key, name, priority):
        return {"rank": rank, "key": key, "name": name, "priority": priority, "types": ["简答题"],
                "summary_md": f"{name}的要点"}

    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        result = {"version": 2, "course": "药理学", "generated_at": "2026-10-22T23:30:00", "models": [], "warnings": [],
                  "scope": {"lectures": [], "materials": []},
                  "points": [point(1, 1, "阿司匹林的剂量分段", 3), point(2, 2, "肝药酶诱导", 2), point(3, 3, "首过效应", 1)],
                  "excluded": [{"name": "药物经济学"}]}
        report_id = ReportStore(library).add(course, "考前报告 2026-10-22 23:30", {}, result, ["claude"])
    page = _client(m1_env).get(f"/reports/{report_id}").text
    folded = page.index('<details class="rest-points">')
    assert "必考（★★★，1 个）" in page and page.index("1. 阿司匹林的剂量分段") < folded
    assert folded < page.index("2. 肝药酶诱导") and folded < page.index("3. 首过效应") < page.index("药物经济学")
    assert "其余 2 个考点（★★ 及以下）和 1 个不考的知识点" in page

    with Library(m1_env.cfg.data_dir / "lecture.db") as library:  # 没有必考考点时给出说明
        result["points"] = [point(1, 2, "肝药酶诱导", 2)]
        result["excluded"] = []
        report_id = ReportStore(library).add(library.course_by_id(m1_env.course_id), "考前报告 2", {}, result, ["claude"])
    page = _client(m1_env).get(f"/reports/{report_id}").text
    assert "没有找到 ★★★ 必考考点" in page and "其余 1 个考点" in page
