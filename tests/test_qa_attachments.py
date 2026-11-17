"""问答附件：上传、读取、送进模型，以及随对话删除。"""

import json

from lecture_analyzer.qa_attachments import (
    MAX_FILES, AttachmentError, get_attachment, read_attachments, resolve_attachments, save_attachment,
)
from lecture_analyzer.library import Library
from lecture_analyzer.pipeline import qa as qa_module

from .conftest import RoutingProvider
from .test_pipeline_run import _run
from .test_web import _post
from .test_web_analysis import _client


def _fake_models(monkeypatch, answer: str = "附件里要求每天一次 [F1]，课程里也讲过剂量分段 [N1]"):
    routes = {"QARetrieval": [json.dumps({"knowledge_points": ["K1"], "lectures": []})],
              "QAAnswer": [json.dumps({"found": True, "answer_md": answer, "supplement_md": ""},
                                      ensure_ascii=False)]}
    created: list[RoutingProvider] = []

    def fake(profile):
        provider = RoutingProvider(routes, profile.model_copy(update={"model": "fake"}))
        created.append(provider)
        return provider

    monkeypatch.setattr(qa_module, "create_provider", fake)
    return created


def test_text_attachment_is_read_and_sent_to_models(m1_env, monkeypatch, tmp_path):
    _run(m1_env)
    data = "题目：阿司匹林应该每天吃几次？\n答案：每天一次。".encode()
    attachment = save_attachment(m1_env.cfg.data_dir, m1_env.course_id, "作业.txt", data)
    assert attachment.name == "作业.txt" and attachment.size_text
    assert get_attachment(m1_env.cfg.data_dir, m1_env.course_id, attachment.id).path.read_bytes() == data

    work = tmp_path / "work"
    contents = read_attachments([attachment], work)
    assert contents[0].has_text and "每天一次" in contents[0].text and not contents[0].images

    created = _fake_models(monkeypatch)
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        result = qa_module.answer_question(
            m1_env.cfg, library, course, "附件里的题目怎么做？", report=lambda _: None, attachments=[attachment],
        )
    assert result["attachments"] == [{**attachment.summary(), "pages": 1}]
    assert "F1" in result["evidence"]
    inputs = [call["parts"][0].text for provider in created for call in provider.calls]
    assert any("[F1] 附件「作业.txt」" in text and "每天一次" in text for text in inputs)
    assert all(sum(isinstance(part, qa_module.ImagePart) for part in call["parts"]) == 0
               for provider in created for call in provider.calls)


def test_upload_and_ask_with_attachment_from_web(m1_env, monkeypatch):
    _run(m1_env)
    _fake_models(monkeypatch)
    client = _client(m1_env)
    token = client.app.state.csrf_token
    url = f"/courses/{m1_env.course_id}/qa/attachments"
    uploaded = client.post(url + "?name=" + "题目.txt", content="题目：剂量怎么分？".encode(),
                           headers={"X-CSRF": token, "Content-Type": "application/octet-stream"})
    assert uploaded.status_code == 200
    attachment = uploaded.json()["attachment"]
    assert attachment["name"] == "题目.txt" and attachment["is_image"] is False

    fetched = client.get(f"{url}/{attachment['id']}")
    assert fetched.status_code == 200 and fetched.content == "题目：剂量怎么分？".encode()

    asked = _post(client, f"/courses/{m1_env.course_id}/qa", question="附件里的题目怎么做？",
                  lecture="", attachment=attachment["id"])
    page = client.get(asked.headers["location"]).text
    assert "题目.txt" in page and "附件" in page and "F1" in page
    assert f'href="{url}/{attachment["id"]}"' in page
    assert "data-attach-drop" in page and "把文件拖到这里" in page and "只用于这一次提问" in page

    _post(client, f"/courses/{m1_env.course_id}/qa/1/delete")
    assert client.get(f"{url}/{attachment['id']}").status_code == 404


def test_attachment_limits_and_unsupported_type(m1_env):
    data_dir = m1_env.cfg.data_dir
    try:
        save_attachment(data_dir, m1_env.course_id, "bad.exe", b"x")
    except AttachmentError as exc:
        assert "不支持" in str(exc)
    else:
        raise AssertionError("应该拒绝不支持的附件格式")

    saved = [save_attachment(data_dir, m1_env.course_id, f"文件{i}.txt", "内容".encode())
             for i in range(MAX_FILES + 1)]
    try:
        resolve_attachments(data_dir, m1_env.course_id, [item.id for item in saved])
    except AttachmentError as exc:
        assert "最多" in str(exc)
    else:
        raise AssertionError("应该限制附件数量")
