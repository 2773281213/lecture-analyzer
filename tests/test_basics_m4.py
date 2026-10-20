"""M4 的基础部分：删除讲和课程、凭据管理器、书库位置、问答记录、设备与模型目录。"""

import json
import os
import time
from pathlib import Path

import pytest

from lecture_analyzer import apphome, credentials
from lecture_analyzer.asr import resolve_device, use_model_dir
from lecture_analyzer.config import ConfigError, find_config_path, load_config
from lecture_analyzer.conversations import QAStore
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.knowledge import CourseMemory
from lecture_analyzer.library import Library, LibraryError
from lecture_analyzer.llm.base import ModelProfile
from lecture_analyzer.materials import MaterialStore
from lecture_analyzer.pipeline.ingest import material_dir
from lecture_analyzer.pipeline.organize import delete_course, delete_lecture

from .test_pipeline_run import _run

# ---------- 删除 ----------


def test_delete_lecture_restores_memory_and_files(m1_env):
    _run(m1_env)
    work = m1_env.cfg.data_dir / "work" / f"lecture-{m1_env.lecture_id}"
    folder = m1_env.cfg.data_dir / "books" / "药理学" / "第03讲"
    assert work.is_dir() and folder.is_dir()
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        second = library.add_lecture(course, 4, [m1_env.media])
        library.set_step(second.id, "validate", "done")
        library.set_step(second.id, "integrate", "done")
        stale = delete_lecture(library, m1_env.cfg.data_dir, library.lecture_by_id(m1_env.lecture_id))
        assert [lecture.seq for lecture in stale] == [4]  # 后面整合过的讲标记为待更新
        assert CourseMemory(library).tree(course) == []  # 课程记忆回到第 3 讲之前（空）
        assert [lecture.seq for lecture in library.lectures(course)] == [4]
        with pytest.raises(LibraryError):
            library.lecture_by_id(m1_env.lecture_id)
    assert not work.exists() and not folder.exists() and m1_env.media.is_file()  # 原始音视频不动


def test_delete_course_removes_everything_but_media(m1_env, tmp_path):
    _run(m1_env)
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        store = MaterialStore(library)
        material = store.add(course, "mock_exam", "模拟题", "x.pdf")
        material_dir(m1_env.cfg.data_dir, material.id).mkdir(parents=True)
        QAStore(library).add(course, None, 1, "问题", {"answer_md": "回答"}, ["claude"])
        delete_course(library, m1_env.cfg.data_dir, course)
        assert library.courses() == []
        counts = {table: library.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
                  for table in ("lecture", "segment", "knowledge_point", "material", "qa", "doc_page")}
        assert set(counts.values()) == {0}, counts
    data = m1_env.cfg.data_dir
    assert not (data / "books" / "药理学").exists() and not material_dir(data, material.id).exists()
    assert not (data / "work" / f"lecture-{m1_env.lecture_id}").exists() and m1_env.media.is_file()


# ---------- 凭据管理器 ----------


def test_credentials_prefer_keyring_then_env(monkeypatch, memory_keyring):
    monkeypatch.setenv("LEC_TEST_KEY", "env-value")
    assert credentials.get_secret("LEC_TEST_KEY") == "env-value"
    assert credentials.secret_source("LEC_TEST_KEY") == "env"
    credentials.set_secret("LEC_TEST_KEY", "  stored-value  ")
    assert credentials.get_secret("LEC_TEST_KEY") == "stored-value" and credentials.secret_source("LEC_TEST_KEY") == "credential"
    assert memory_keyring.store == {("lecture-analyzer", "LEC_TEST_KEY"): "stored-value"}
    profile = ModelProfile(name="x", adapter="openai_compat", model="m", api_key_env="LEC_TEST_KEY")
    assert profile.api_key() == "stored-value"
    credentials.delete_secret("LEC_TEST_KEY")
    credentials.delete_secret("LEC_TEST_KEY")  # 没有也不报错
    assert credentials.get_secret("LEC_TEST_KEY") == "env-value"
    for bad in ("lower_case", "A", "WITH-DASH", ""):
        with pytest.raises(credentials.CredentialError):
            credentials.set_secret(bad, "x")
    with pytest.raises(credentials.CredentialError, match="不能为空"):
        credentials.set_secret("LEC_TEST_KEY", "  ")


# ---------- 书库位置 ----------


def test_create_home_and_saved_home(tmp_path, monkeypatch):
    home = tmp_path / "课程书架"
    config = apphome.create_home(home)
    assert (home / ".env").is_file() and (home / "data").is_dir()
    cfg = load_config(config)
    assert cfg.asr.model_dir == "models" and cfg.model_dir == (home / "models").resolve()
    assert cfg.asr.device == "auto"
    apphome.create_home(home)  # 已有的文件不覆盖
    assert load_config(config).asr.model_dir == "models"

    monkeypatch.chdir(tmp_path)  # 当前目录没有 config.toml
    with pytest.raises(ConfigError, match="首次设置"):
        find_config_path()
    apphome.remember_home(home)
    assert apphome.saved_home() == home.resolve() and find_config_path() == config.resolve()
    settings = json.loads(apphome.settings_file().read_text(encoding="utf-8"))
    assert settings["home"] == str(home.resolve())


def test_saved_home_ignores_missing_folder(tmp_path):
    apphome.write_settings({"home": str(tmp_path / "不存在")})
    assert apphome.saved_home() is None
    apphome.settings_file().write_text("不是 JSON", encoding="utf-8")
    assert apphome.read_settings() == {}


# ---------- 问答记录 ----------


def test_qa_threads(tmp_path):
    with Library(tmp_path / "lecture.db") as library:
        course = library.add_course("病毒学", CourseSettings())
        lecture = library.add_lecture(course, 1, [tmp_path / "a.mp4"])
        store = QAStore(library)
        assert store.next_thread(course) == 1
        first = store.add(course, None, 1, "什么是朊病毒？", {"answer_md": "没有核酸的蛋白质"}, ["claude"])
        store.add(course, lecture.id, 1, "第 1 讲怎么讲的？", {"answer_md": "……"}, ["claude"])
        store.add(course, None, 2, "Baltimore 分类？", {"answer_md": "七群"}, ["gemini"])
        assert [t.question for t in store.thread(course, 1)] == ["什么是朊病毒？", "第 1 讲怎么讲的？"]
        assert store.get(first).result["answer_md"] == "没有核酸的蛋白质"
        assert [(s.thread, s.turns, s.first_question) for s in store.threads(course)][1] == (1, 2, "什么是朊病毒？")
        assert store.next_thread(course) == 3
        library.delete_lecture(lecture.id)
        assert store.thread(course, 1)[1].lecture_id is None  # 删讲后问答保留
        store.delete_thread(course, 1)
        assert [s.thread for s in store.threads(course)] == [2]


# ---------- 设备与模型目录 ----------


def test_device_and_model_dir(example_config: Path, monkeypatch):
    assert resolve_device("cpu") == "cpu" and resolve_device("cuda:0") == "cuda:0"
    assert resolve_device("auto") in ("cuda:0", "cpu")
    for name in ("MODELSCOPE_CACHE", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)  # 先记下原值，测试结束后恢复
    cfg = load_config(example_config)
    use_model_dir(cfg)  # 没配置时不改环境
    assert "MODELSCOPE_CACHE" not in os.environ
    cfg.asr.model_dir = "models"
    use_model_dir(cfg)
    models = example_config.parent.resolve() / "models"
    assert os.environ["MODELSCOPE_CACHE"] == str(models) and os.environ["HF_HOME"] == str(models / "huggingface")


# ---------- 本地模型下载 ----------


def test_download_models_reports_progress(example_config: Path, monkeypatch):
    from lecture_analyzer.asr import models

    cfg = load_config(example_config)
    cfg.asr.model_dir = "models"
    for name in ("MODELSCOPE_CACHE", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)
    root = models.cache_root(cfg)
    assert root == example_config.parent.resolve() / "models"
    assert models.required_repos(cfg, "llm:gemini") == []
    repos = models.required_repos(cfg, "funasr:paraformer")
    assert repos == [models.FUNASR_REPOS[name] for name in ("paraformer-zh", "fsmn-vad", "ct-punc")]
    assert models.required_repos(cfg, "qwen3:0.6b") == ["Qwen/Qwen3-ASR-0.6B-hf", models.FUNASR_REPOS["fsmn-vad"]]

    downloaded: list[str] = []

    def fake_snapshot(repo: str) -> None:  # 模拟下载：先写半成品，再写完整文件
        assert os.environ["MODELSCOPE_CACHE"] == str(root)
        snapshot = models.repo_dir(root, repo) / "snapshots" / "master"
        snapshot.mkdir(parents=True)
        (snapshot / "model.pt.incomplete").write_bytes(b"0" * 2_000_000)
        time.sleep(0.3)
        (snapshot / "model.pt.incomplete").rename(snapshot / "model.pt")
        (snapshot / "configuration.json").write_text("{}", encoding="utf-8")
        downloaded.append(repo)

    monkeypatch.setattr(models, "_snapshot", fake_snapshot)
    monkeypatch.setattr(models, "PROGRESS_SECONDS", 0.1)
    vad = models.repo_dir(root, models.FUNASR_REPOS["fsmn-vad"]) / "snapshots" / "master"
    vad.mkdir(parents=True)
    (vad / "configuration.json").write_text("{}", encoding="utf-8")  # 已经下载过的不再下载
    messages: list[str] = []
    assert models.download_models(cfg, "funasr:paraformer", messages.append) == [repos[0], repos[2]]
    assert downloaded == [repos[0], repos[2]]
    assert any("下载本地模型 1/2" in m and "已下载 2 / 约 953 MB（0%）" in m for m in messages)
    assert models.model_status(cfg, "funasr:paraformer") == [(repo, True) for repo in repos]
    assert models.download_models(cfg, "funasr:paraformer", messages.append) == []

    def broken(repo: str) -> None:
        raise ConnectionError("网络断开")

    monkeypatch.setattr(models, "_snapshot", broken)
    with pytest.raises(models.ASRError, match="下载模型 Qwen/Qwen3-ASR-0.6B-hf 失败：网络断开"):
        models.download_models(cfg, "qwen3:0.6b", messages.append)
