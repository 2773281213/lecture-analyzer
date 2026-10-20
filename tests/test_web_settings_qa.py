"""设置页（改配置保留注释、密钥存进凭据管理器的测试替身、测试连接、下载模型）、问答页和出处链接。"""

import json

import pytest
from markupsafe import Markup

from lecture_analyzer import config_edit
from lecture_analyzer.asr import models as asr_models
from lecture_analyzer.config import load_config
from lecture_analyzer.conversations import QAStore
from lecture_analyzer.library import Library
from lecture_analyzer.llm import LLMError
from lecture_analyzer.pipeline import qa as qa_module
from lecture_analyzer.web import settings_page
from lecture_analyzer.web.rendering import cite_links, render_markdown

from .conftest import RoutingProvider
from .test_pipeline_run import _run
from .test_web import _post
from .test_web_analysis import _client

ENV_NAMES = ("CLAUDE_MODEL", "GEMINI_MODEL", "GEMINI_BASE_URL", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
             "ANTHROPIC_BASE_URL", "DEEPSEEK_API_KEY")


@pytest.fixture
def clean_env(monkeypatch):
    """设置页会改当前进程的环境变量：测试前记下，测试后恢复。"""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_settings_page_edits_keep_comments_and_never_show_keys(m1_env, clean_env, memory_keyring, monkeypatch):
    client = _client(m1_env)
    env_file = m1_env.cfg.root / ".env"
    env_file.write_text("# 我的设置\nGEMINI_API_KEY=old-secret-value\nGEMINI_BASE_URL=\n", encoding="utf-8")
    page = client.get("/settings?welcome=1").text
    assert "第一次使用" in page and "GEMINI_API_KEY" in page and "移到 Windows 凭据管理器" in page

    _post(client, "/settings/models/gemini", model="gemini-3.8-flash", base_url="https://relay.example")
    text = env_file.read_text(encoding="utf-8")
    assert "# 我的设置" in text and "GEMINI_MODEL=gemini-3.8-flash" in text and "GEMINI_BASE_URL=https://relay.example" in text
    assert client.app.state.cfg.profile("gemini").model == "gemini-3.8-flash"
    _post(client, "/settings/models/deepseek", model="deepseek-chat", base_url="")  # 没有 model_env：写进 config.toml
    config_text = (m1_env.cfg.root / "config.toml").read_text(encoding="utf-8")
    assert 'model = "deepseek-chat"' in config_text and "# 考前报告" in config_text
    bad_url = _post(client, "/settings/models/gemini", model="x", base_url="relay.example")
    assert "接口地址要以 http:// 或 https:// 开头" in client.get(bad_url.headers["location"]).text

    _post(client, "/settings/models/gemini/key", key="  new-secret-value  ")
    assert memory_keyring.store == {("lecture-analyzer", "GEMINI_API_KEY"): "new-secret-value"}
    assert "GEMINI_API_KEY=\n" in env_file.read_text(encoding="utf-8")  # .env 里的旧值清空
    page = client.get("/settings").text
    assert "已设置（凭据管理器）" in page and "new-secret-value" not in page and "old-secret-value" not in page
    _post(client, "/settings/models/gemini/key/delete")
    assert memory_keyring.store == {}

    monkeypatch.setattr(settings_page, "ping", lambda profile: (True, "0.5s，输入 3 / 输出 1 token"))
    tested = _post(client, "/settings/models/gemini/test")
    assert "连接成功：0.5s" in client.get(tested.headers["location"]).text

    _post(client, "/settings/steps", qa="gemini", page_digest="claude")
    _post(client, "/settings/asr", backend="llm", llm_profile="gemini", funasr_preset="sensevoice", qwen3_size="0.6b",
          device="cpu", model_dir="models")
    _post(client, "/settings/general", digest_workers="8")
    cfg = load_config(m1_env.cfg.root / "config.toml")
    assert (cfg.steps["qa"], cfg.steps["page_digest"], cfg.asr.backend, cfg.asr.device) == ("gemini", "claude", "llm", "cpu")
    assert (cfg.materials.digest_workers, cfg.asr.model_dir) == (8, "models")
    assert "报告用什么语言" not in client.get("/settings").text  # 考前报告一律中文，不再是设置项
    config_text = (m1_env.cfg.root / "config.toml").read_text(encoding="utf-8")
    assert "# 各步骤的推理强度" in config_text and "# 每一步用哪个模型档案" in config_text  # 注释都在
    refused = _post(client, "/settings/asr", llm_profile="deepseek")
    assert "不支持音频输入" in client.get(refused.headers["location"]).text
    assert load_config(m1_env.cfg.root / "config.toml").asr.llm_profile == "gemini"  # 取值不对时不改文件


def test_move_env_keys_and_download_job(m1_env, clean_env, memory_keyring, monkeypatch):
    client = _client(m1_env)
    (m1_env.cfg.root / ".env").write_text("ANTHROPIC_API_KEY=sk-test\nANTHROPIC_BASE_URL=https://relay.example\n",
                                          encoding="utf-8")
    _post(client, "/settings/keys/move")
    assert memory_keyring.store == {("lecture-analyzer", "ANTHROPIC_API_KEY"): "sk-test"}
    assert (m1_env.cfg.root / ".env").read_text(encoding="utf-8") == "ANTHROPIC_API_KEY=\nANTHROPIC_BASE_URL=https://relay.example\n"

    monkeypatch.setattr(asr_models, "download_models", lambda cfg, spec, report: [f"{spec} 的模型"])
    started = _post(client, "/settings/models-download", spec="funasr:paraformer")
    status = client.get("/api/jobs/" + started.headers["location"].rsplit("/", 1)[1]).json()
    assert status["state"] == "done" and status["summary"] == "下载了 1 个模型"
    assert _post(client, "/settings/models-download", spec="whisper").status_code == 400


def test_config_edit_validation(example_config, clean_env):
    cfg = load_config(example_config)
    with pytest.raises(ValueError, match="未知的步骤"):
        config_edit.set_steps(cfg, {"everything": "claude"})
    with pytest.raises(ValueError, match="vision"):
        config_edit.set_steps(cfg, {"page_digest": "deepseek"})  # 识别页面需要能看图
    with pytest.raises(ValueError, match="控制字符"):
        config_edit.set_profile_model(cfg, "claude", "bad\nvalue")
    with pytest.raises(ValueError, match="并发数要在"):
        config_edit.set_general(cfg, 99)
    config_edit.set_env(cfg, "GEMINI_MODEL", "名字 带空格")
    assert 'GEMINI_MODEL="名字 带空格"' in (example_config.parent / ".env").read_text(encoding="utf-8")
    assert load_config(example_config).profile("gemini").model == "名字 带空格"


def test_qa_page_asks_and_links_sources(m1_env, monkeypatch):
    _run(m1_env)
    routes = {"QARetrieval": [json.dumps({"knowledge_points": ["K1"], "lectures": []})],
              "QAAnswer": [json.dumps({"found": True, "answer_md": "小于 75 mg 抗血凝 [N1]，见课件 [Q1]",
                                       "supplement_md": ""}, ensure_ascii=False)]}
    monkeypatch.setattr(qa_module, "create_provider",
                        lambda profile: RoutingProvider(routes, profile.model_copy(update={"model": "fake"})))
    client = _client(m1_env)
    empty = client.get(f"/courses/{m1_env.course_id}/qa").text
    assert "你的问题" in empty and "还没有对话" in empty
    asked = _post(client, f"/courses/{m1_env.course_id}/qa", question="阿司匹林的剂量怎么分？", lecture="")
    assert asked.headers["location"] == f"/courses/{m1_env.course_id}/qa?thread=1"
    page = client.get(asked.headers["location"]).text
    assert f'<a class="cite" href="/lectures/{m1_env.lecture_id}?t=2000"' in page  # 点出处跳到录音的时间点
    assert "出处（2）" in page and "这个期末一定会考" in page and "接着问" in page
    _post(client, f"/courses/{m1_env.course_id}/qa", question="那大剂量呢？", thread="1",
          lecture=str(m1_env.lecture_id))
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        turns = QAStore(library).thread(library.course_by_id(m1_env.course_id), 1)
    assert [t.lecture_id for t in turns] == [None, m1_env.lecture_id]
    assert "限第 3 讲" in client.get(asked.headers["location"]).text
    assert _post(client, f"/courses/{m1_env.course_id}/qa", question=" ").status_code == 400
    _post(client, f"/courses/{m1_env.course_id}/qa/1/delete")
    assert "还没有对话" in client.get(f"/courses/{m1_env.course_id}/qa").text


def test_failed_question_stays_in_thread_and_can_be_asked_again(m1_env, monkeypatch):
    # M4 验收实测：问答任务失败（中转站容量不足）后，问题在对话页上直接消失
    _run(m1_env)
    answer = json.dumps({"found": True, "answer_md": "小于 75 mg 抗血凝 [N1]", "supplement_md": ""}, ensure_ascii=False)
    routes = {"QARetrieval": [LLMError("模型档案「gemini」暂时不可用（连接中断或服务端繁忙）"),
                              json.dumps({"knowledge_points": ["K1"], "lectures": []})],
              "QAAnswer": [answer]}
    monkeypatch.setattr(qa_module, "create_provider",
                        lambda profile: RoutingProvider(routes, profile.model_copy(update={"model": "fake"})))
    client = _client(m1_env)
    course_qa = f"/courses/{m1_env.course_id}/qa"
    asked = _post(client, course_qa, question="阿司匹林的剂量怎么分？", lecture=str(m1_env.lecture_id))
    page = client.get(asked.headers["location"]).text
    assert "没有答成" in page and "暂时不可用" in page and "重新提问" in page and "接着问" in page
    assert 'name="question" value="阿司匹林的剂量怎么分？"' in page and f'name="lecture" value="{m1_env.lecture_id}"' in page

    other = _post(client, course_qa, question="另一个问题")
    assert other.headers["location"] == f"{course_qa}?thread=2"  # 新对话不复用没答成的问题占用的编号

    _post(client, course_qa, question="阿司匹林的剂量怎么分？", thread="1", lecture=str(m1_env.lecture_id))
    page = client.get(asked.headers["location"]).text
    assert "抗血凝" in page and "重新提问" not in page  # 答成之后，之前的失败不再显示


def test_qa_scope_only_offers_integrated_lectures(m1_env, tmp_path):
    # M4 验收：第 4 讲还在转写时，问答的范围里已经能选它，选了只会得到一个失败的任务
    client = _client(m1_env)
    course_qa = f"/courses/{m1_env.course_id}/qa"
    assert _post(client, course_qa, question="剂量怎么分？", lecture=str(m1_env.lecture_id)).status_code == 400
    _run(m1_env)
    media = tmp_path / "第4讲.mp4"
    media.write_bytes(b"fake")
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        later = library.add_lecture(library.course_by_id(m1_env.course_id), 4, [media])
    page = client.get(f"{course_qa}?lecture={later.id}").text
    assert f'<option value="{later.id}" disabled>第 4 讲（还没处理）</option>' in page
    assert f'<option value="{later.id}" selected' not in page  # 不预先选中还没处理的讲
    assert "就这一讲提问" not in client.get(f"/lectures/{later.id}").text
    assert "就这一讲提问" in client.get(f"/lectures/{m1_env.lecture_id}").text
    # 中文标题按字截断（M4 验收：「第1章 病毒绪论：……」被截成「第1章...」）
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        library.update_lecture(m1_env.lecture_id, title="第3章 药效学：量效关系、受体学说、影响药效的因素与不良反应")
    assert "「第3章 药效学：量效关系、受体学说..." in client.get(course_qa).text


def test_cite_links_only_touch_text():
    evidence = {"P1": {"id": "P1", "material_id": 3, "page_no": 7, "label": "「第二章」第 7 页", "title": ""},
                "T1": {"id": "T1", "lecture_id": 5, "seq": 2, "t": 61000, "text": "原文"}}
    html = render_markdown('见 [P1][T1] 和 [X9]，![图](a.png "标题 [P1]")')
    linked = str(cite_links(html, evidence))
    assert '<a class="cite page-link" href="/materials/3/pages/7.png"' in linked
    assert '<a class="cite" href="/lectures/5?t=61000"' in linked and "[X9]" in linked
    assert 'title="标题 [P1]"' in linked  # 属性里的编号不动
    assert isinstance(cite_links(Markup("[T1]"), evidence), Markup)
