"""网页里的管理操作：新建和修改课程、删除课程、添加讲、课件关卡、管理一讲、导入和删除资料、任务页。"""

from pathlib import Path

from fastapi.testclient import TestClient

from lecture_analyzer.jobs import JobStore
from lecture_analyzer.library import Library
from lecture_analyzer.materials import MaterialStore
from lecture_analyzer.web import create_app

from .test_materials import _pdf
from .test_pipeline_run import _run
from .test_web import BASE, _post
from .test_web_analysis import _client


class RecordingRunner:
    """只记下提交的任务，不执行（检查提交了哪些任务、顺序对不对）。"""

    def __init__(self) -> None:
        self.submitted: list[int] = []

    def start(self) -> None:
        pass

    def submit(self, job_id: int) -> None:
        self.submitted.append(job_id)

    def cancel(self, job_id: int) -> bool:
        return False

    def shutdown(self) -> None:
        pass


def _recording_client(env) -> TestClient:
    app = create_app(env.cfg)
    app.state.jobs = RecordingRunner()
    return TestClient(app, base_url=BASE, follow_redirects=False)


def _db(env) -> Path:
    return env.cfg.data_dir / "lecture.db"


def _jobs(env) -> list:
    with Library(_db(env)) as library:
        return list(reversed(JobStore(library).latest()))


def test_create_edit_and_delete_course(m1_env):
    client = _client(m1_env)
    form = client.get("/courses/new").text
    assert "新建课程" in form and 'name="teaching" value="ko"' in form and 'name="teaching" value="en"' in form
    assert "一律用中文" in form and 'name="note_language"' not in form  # 笔记语言固定中文，不再是选项
    created = _post(client, "/courses", name="病毒学", subject="病毒学", teaching=["ko"], interpreted_to="zh",
                    asr="", slides="none", instructions="老师常考名词解释")
    assert created.status_code == 303
    page = client.get(created.headers["location"]).text
    assert "老师用韩语授课，现场有人翻译成中文" in page and "老师常考名词解释" in page
    mixed = _post(client, "/courses", name="生物学", teaching=["ko", "en"], teaching_other="fr", asr="")
    assert mixed.status_code == 303 and "韩语、英语、法语授课" in client.get(mixed.headers["location"]).text
    bad = _post(client, "/courses", name="微生物", teaching_other="韩语", asr="")
    assert bad.status_code == 400 and "语言代码格式不对" in bad.text
    none = _post(client, "/courses", name="微生物", asr="")
    assert none.status_code == 400 and "至少要有一种授课语言" in none.text
    duplicate = _post(client, "/courses", name="病毒学", teaching=["zh"], asr="")
    assert duplicate.status_code == 400 and "已存在" in duplicate.text
    assert _post(client, "/courses", name="生理学", teaching=["zh"], asr="qwen9").status_code == 400
    english = _post(client, "/courses", name="英语课", teaching=["en"], asr="funasr:paraformer")
    assert english.status_code == 400 and "本地 Paraformer 不支持英语" in english.text

    _run(m1_env)  # 药理学第 3 讲已处理：改学科后要从「整理笔记」起重做
    course_id = m1_env.course_id
    saved = _post(client, f"/courses/{course_id}/settings", subject="临床药理学", teaching=["zh"], interpreted_to="",
                  asr="", instructions="")
    assert saved.headers["location"] == f"/courses/{course_id}/settings?saved=extract"
    settings_page = client.get(saved.headers["location"]).text
    assert "从「整理笔记」起重做" in settings_page and 'name="force" value="extract"' in settings_page
    assert 'name="teaching" value="zh" checked' in settings_page
    same = _post(client, f"/courses/{course_id}/settings", subject="临床药理学", teaching=["zh"], interpreted_to="",
                 asr="", instructions="")
    assert same.headers["location"].endswith("saved=none")

    wrong = _post(client, f"/courses/{course_id}/delete", confirm="药理")
    assert wrong.status_code == 400 and "完整的课程名" in wrong.text
    deleted = _post(client, f"/courses/{course_id}/delete", confirm="药理学")
    assert deleted.status_code == 303 and deleted.headers["location"] == "/"
    shelf = client.get("/").text
    assert "已删除课程「药理学」" in shelf and "/courses/{}".format(course_id) not in shelf
    assert m1_env.media.is_file()  # 原始音视频不动


def test_add_lecture_and_slides_gate(m1_env, tmp_path):
    with Library(_db(m1_env)) as library:
        library.update_course(m1_env.course_id, slides="unset")
    media = [tmp_path / "第4讲-1.mp4", tmp_path / "第4讲-2.mp4"]
    for path in media:
        path.write_bytes(b"fake")
    client = _recording_client(m1_env)
    form = {"paths": f'"{media[0]}"\n{media[1]}\n', "seq": "", "kind": "review", "title": "", "date": "2026-10-16",
            "run": "1"}
    added = _post(client, f"/courses/{m1_env.course_id}/lectures", **form)
    location = added.headers["location"]
    assert added.status_code == 303 and "?gate=1&lecture=" in location  # 还没选择是否提供课件：先弹出选择框
    with Library(_db(m1_env)) as library:
        lecture = library.lecture(library.course_by_id(m1_env.course_id), 4)  # 不填讲次时接在最后
    assert [Path(p.media_path) for p in lecture.parts] == media and lecture.kind == "review"
    gate_page = client.get(location).text
    assert 'id="slides-gate"' in gate_page and f'name="lecture" value="{lecture.id}"' in gate_page
    assert _jobs(m1_env) == []  # 选择之前不提交任务

    chosen = _post(client, f"/courses/{m1_env.course_id}/slides", then_run="1", lecture=str(lecture.id), force="")
    jobs = _jobs(m1_env)
    assert chosen.headers["location"] == f"/jobs/{jobs[0].id}"
    assert jobs[0].kind == "run" and jobs[0].params == {"lectures": [lecture.id], "force": []}
    with Library(_db(m1_env)) as library:
        assert library.course_by_id(m1_env.course_id).slides == "none"

    relative = _post(client, f"/courses/{m1_env.course_id}/lectures", paths="第5讲.mp4", seq="5")
    assert relative.status_code == 400 and "完整路径" in relative.text
    missing = _post(client, f"/courses/{m1_env.course_id}/lectures", paths=str(tmp_path / "没有.mp4"), seq="5")
    assert missing.status_code == 400 and "找不到" in missing.text
    taken = _post(client, f"/courses/{m1_env.course_id}/lectures", paths=str(media[0]), seq="4")
    assert taken.status_code == 400 and "已有第 4 讲" in taken.text


def test_import_slides_from_gate_then_run(m1_env, tmp_path):
    with Library(_db(m1_env)) as library:
        library.update_course(m1_env.course_id, slides="unset")
    pdf = _pdf(tmp_path / "第三章.pdf", 2)
    client = _recording_client(m1_env)
    response = _post(client, f"/courses/{m1_env.course_id}/materials", paths=str(pdf), kind="slides", process="1",
                     then_run="1", lecture=str(m1_env.lecture_id), force="extract")
    assert response.headers["location"] == "/jobs"
    imported, run = _jobs(m1_env)
    assert imported.kind == "import_material" and imported.params["paths"] == [str(pdf.resolve())]
    assert run.kind == "run" and run.params == {"lectures": [m1_env.lecture_id], "force": ["extract"]}
    assert client.app.state.jobs.submitted == [imported.id, run.id]  # 同一条队列，先导入再处理
    jobs_page = client.get("/jobs").text
    assert "导入资料" in jobs_page and "处理讲" in jobs_page and "取消" in jobs_page
    assert _post(client, f"/courses/{m1_env.course_id}/materials", paths=str(pdf), kind="unknown").status_code == 400


def test_run_buttons_and_lecture_management(m1_env, tmp_path):
    client = _client(m1_env)
    lecture_id = m1_env.lecture_id
    started = _post(client, f"/lectures/{lecture_id}/run", force="")
    assert started.status_code == 303
    job = _jobs(m1_env)[-1]
    assert job.state == "done" and job.result["url"] == f"/lectures/{lecture_id}"
    page = client.get(f"/lectures/{lecture_id}").text
    assert "就这一讲提问" in page and "管理这一讲" in page and 'value="validate"' in page
    bad_step = _post(client, f"/lectures/{lecture_id}/run", force="everything")
    assert bad_step.status_code == 400 and "未知的步骤" in bad_step.text
    # 课程页也能从某一步起重做全部讲（M4 修订：病毒学 4 讲按新规则从「整理笔记」起重做）
    course_page = client.get(f"/courses/{m1_env.course_id}").text
    assert "处理全部讲" in course_page and 'name="force"' in course_page
    redo = _post(client, f"/courses/{m1_env.course_id}/run", force="extract")
    assert redo.status_code == 303
    job = _jobs(m1_env)[-1]
    assert job.params == {"lectures": [], "force": ["extract"]} and job.state == "done"
    assert job.result["summary"] == "处理了 1 讲"  # 已处理过的讲也重做了

    edited = _post(client, f"/lectures/{lecture_id}/edit", title="量效关系（修订）", kind="normal", date="2026-10-01")
    assert edited.status_code == 303
    bad_date = _post(client, f"/lectures/{lecture_id}/edit", title="x", kind="normal", date="10/01")
    assert bad_date.status_code == 400
    extra = tmp_path / "补录.wav"
    extra.write_bytes(b"fake")
    _post(client, f"/lectures/{lecture_id}/parts", paths=str(extra))
    with Library(_db(m1_env)) as library:
        lecture = library.lecture_by_id(lecture_id)
        assert lecture.title == "量效关系（修订）" and lecture.date == "2026-10-01" and len(lecture.parts) == 2
        assert lecture.step_state("media") == ""  # 追加一节后要重新抽取音频
        store = MaterialStore(library)
        material = store.add(library.course_by_id(m1_env.course_id), "slides", "第二章", "第二章.pdf")
        store.set_file(material.id, f"materials/{material.id}/x.pdf", 10)
    pinned = _post(client, f"/lectures/{lecture_id}/pages", material=str(material.id), pages="3-5")
    assert pinned.status_code == 303 and "已修改对应的课件页" in client.get(f"/lectures/{lecture_id}").text
    with Library(_db(m1_env)) as library:
        assert [p.ref.page_no for p in MaterialStore(library).lecture_pages(lecture_id)] == [3, 4, 5]
    out_of_range = _post(client, f"/lectures/{lecture_id}/pages", material=str(material.id), pages="9-12")
    assert out_of_range.status_code == 400 and "超出范围" in out_of_range.text
    _post(client, f"/lectures/{lecture_id}/pages", mode="auto")
    with Library(_db(m1_env)) as library:
        assert MaterialStore(library).lecture_pages(lecture_id) == []

    not_confirmed = _post(client, f"/lectures/{lecture_id}/delete")
    assert not_confirmed.status_code == 400
    deleted = _post(client, f"/lectures/{lecture_id}/delete", confirm="1")
    assert deleted.headers["location"] == f"/courses/{m1_env.course_id}"
    assert "已删除第 3 讲" in client.get(deleted.headers["location"]).text
    assert client.get(f"/lectures/{lecture_id}").status_code == 404


def test_delete_material_and_busy_course(m1_env, tmp_path):
    client = _client(m1_env)
    pdf = _pdf(tmp_path / "模拟题.pdf", 1)
    imported = _post(client, f"/courses/{m1_env.course_id}/materials", paths=str(pdf), kind="mock_exam", title="期中模拟")
    job = _jobs(m1_env)[-1]
    assert imported.headers["location"] == f"/jobs/{job.id}" and job.state == "done", job.error
    with Library(_db(m1_env)) as library:
        material = MaterialStore(library).list(library.course_by_id(m1_env.course_id))[0]
        assert material.title == "期中模拟" and material.step_state("digest") == ""  # 没勾选立即识别
        busy = JobStore(library).add(library.course_by_id(m1_env.course_id), "rebuild", {})
    course_page = client.get(f"/courses/{m1_env.course_id}").text
    assert "继续处理没处理完的资料" in course_page and f'data-job-id="{busy.id}"' in course_page
    refused = _post(client, f"/materials/{material.id}/delete")
    assert refused.status_code == 400 and "还有没结束的后台任务" in refused.text
    with Library(_db(m1_env)) as library:
        JobStore(library).cancel(busy.id)
    removed = _post(client, f"/materials/{material.id}/delete")
    assert removed.headers["location"] == f"/courses/{m1_env.course_id}#materials"
    assert "已删除资料「期中模拟」" in client.get(removed.headers["location"]).text
    assert pdf.is_file()  # 原文件不受影响


def test_path_fields_have_unique_ids(m1_env):
    """原生选文件按钮按 id 找要填的框：同一页上的几个路径框 id 不能重复。"""
    import re

    with Library(_db(m1_env)) as library:
        library.update_course(m1_env.course_id, slides="unset")
    client = _client(m1_env)
    for url in (f"/courses/{m1_env.course_id}?gate=1&lecture={m1_env.lecture_id}", f"/lectures/{m1_env.lecture_id}"):
        ids = re.findall(r'<textarea id="(field-[^"]+)"', client.get(url).text)
        targets = re.findall(r'data-target="(field-[^"]+)"', client.get(url).text)
        assert ids and len(ids) == len(set(ids)) and set(targets) <= set(ids), (url, ids)
