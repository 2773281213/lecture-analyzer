import os
from pathlib import Path

import pytest

from lecture_analyzer.config import ConfigError, find_config_path, load_config


def test_example_config_loads(example_config: Path):
    cfg = load_config(example_config)
    assert set(cfg.models) >= {"claude", "gemini", "gpt", "deepseek", "qwen", "kimi", "glm"}
    claude = cfg.profile("claude")
    assert claude.model == "claude-opus-5-5" and claude.vision and claude.structured == "native"
    assert claude.options["refusal_fallback"] is True
    assert cfg.profile("gpt").options["max_tokens_param"] == "max_completion_tokens"
    assert cfg.data_dir == (example_config.parent / "data").resolve()
    assert cfg.effort_for_step("report") == "high"


def test_check_reports_unconfigured_step_model(example_config: Path):
    problems = load_config(example_config).check()
    assert any("page_digest" in p and "gemini" in p for p in problems)


def test_profile_for_step_enforces_capabilities(example_config: Path):
    cfg = load_config(example_config)
    assert cfg.profile_for_step("page_digest", override="claude").name == "claude"
    with pytest.raises(ConfigError, match="vision"):
        cfg.profile_for_step("page_digest", override="deepseek")
    with pytest.raises(ConfigError, match="没有名为"):
        cfg.profile_for_step("extract", override="不存在")


def test_missing_config_raises(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEC_CONFIG", raising=False)
    with pytest.raises(ConfigError, match="config.example.toml"):
        find_config_path()


def test_config_found_via_env(example_config: Path, monkeypatch, tmp_path: Path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    monkeypatch.setenv("LEC_CONFIG", str(example_config))
    assert find_config_path() == example_config.resolve()


def test_invalid_profile_reported(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text('[models.bad]\nadapter = "nope"\nmodel = "x"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="有误"):
        load_config(path)


def test_dotenv_values_take_precedence_but_blanks_do_not_clobber(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LEC_TEST_BASE", "https://api.anthropic.com")  # 例如运行环境里已有的官方地址
    monkeypatch.setenv("LEC_TEST_KEEP", "existing-value")
    (tmp_path / ".env").write_text(
        "LEC_TEST_BASE=https://relay.example/\nLEC_TEST_KEEP=\n# 注释行\n", encoding="utf-8"
    )
    path = tmp_path / "config.toml"
    path.write_text(
        '[models.m]\nadapter = "anthropic"\nmodel = "x"\nbase_url_env = "LEC_TEST_BASE"\n', encoding="utf-8"
    )
    cfg = load_config(path)
    assert cfg.profile("m").resolved_base_url() == "https://relay.example/"
    assert os.environ["LEC_TEST_KEEP"] == "existing-value"


def test_model_from_env_overrides_config(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("LEC_TEST_MODEL", raising=False)
    monkeypatch.delenv("LEC_TEST_EMPTY_MODEL", raising=False)
    (tmp_path / ".env").write_text("LEC_TEST_MODEL=relay-model-x\nLEC_TEST_EMPTY_MODEL=\n", encoding="utf-8")
    path = tmp_path / "config.toml"
    path.write_text(
        '[models.a]\nadapter = "anthropic"\nmodel = "default-a"\nmodel_env = "LEC_TEST_MODEL"\n'
        '[models.b]\nadapter = "gemini"\nmodel = "default-b"\nmodel_env = "LEC_TEST_EMPTY_MODEL"\n',
        encoding="utf-8",
    )
    try:
        cfg = load_config(path)
        assert cfg.profile("a").model == "relay-model-x"  # .env 填了型号：覆盖
        assert cfg.profile("b").model == "default-b"  # .env 留空：沿用 config.toml
    finally:
        os.environ.pop("LEC_TEST_MODEL", None)


def test_dotenv_next_to_config_is_loaded(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("LEC_TEST_DOTENV_KEY", raising=False)
    (tmp_path / ".env").write_text("LEC_TEST_DOTENV_KEY=secret-value\n", encoding="utf-8")
    path = tmp_path / "config.toml"
    path.write_text(
        '[models.m]\nadapter = "openai_compat"\nmodel = "x"\napi_key_env = "LEC_TEST_DOTENV_KEY"\n',
        encoding="utf-8",
    )
    try:
        assert load_config(path).profile("m").api_key() == "secret-value"
    finally:
        os.environ.pop("LEC_TEST_DOTENV_KEY", None)
