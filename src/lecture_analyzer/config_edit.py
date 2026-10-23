"""在设置页里改设置（命令行用户也可以直接改文件）：

- config.toml 用 tomlkit 修改，保留注释和顺序；新内容先写进同一目录的临时文件试读一遍，没问题才替换原文件；
- 型号、接口地址写进 .env（档案设置了 model_env / base_url_env 时，.env 的值优先于 config.toml）；
  改 .env 只动对应的那一行，其余行和注释原样保留，并同步当前进程的环境变量；
- 密钥存进 Windows 凭据管理器（credentials.py），.env 里同名的旧值清空，免得两处不一致。

只开放设置页上的这些项，取值都先校验再写入。
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import tomlkit
from dotenv import dotenv_values

from .asr import ASR_BACKENDS
from .config import AppConfig, ConfigError, load_config
from .credentials import check_name, set_secret

STEP_LABELS = {
    "page_digest": "课件和资料逐页识别（需要能看图）", "match": "课件匹配到各讲", "exam_plan": "分析考试方案",
    "extract": "整理笔记", "integrate": "整合课程记忆", "review": "总复习讲义", "material": "资料拆题", "report": "考前报告",
    "qa_retrieve": "问答：查找相关知识点", "qa": "问答：回答",
}
_DEVICE = re.compile(r"^(auto|cpu|cuda(:\d{1,2})?)$")
_UNSAFE = re.compile(r"[\x00-\x1f\x7f]")
_NEEDS_QUOTES = re.compile(r"[\s#'\"\\]")
MAX_VALUE_CHARS = 500
MAX_DIGEST_WORKERS = 16


def clean_value(value: str) -> str:
    text = value.strip()
    if len(text) > MAX_VALUE_CHARS or _UNSAFE.search(text):
        raise ConfigError("取值过长或含有控制字符")
    return text


# ---------- config.toml ----------


def config_file(cfg: AppConfig) -> Path:
    if cfg.source is None:
        raise ConfigError("不知道配置文件的位置，无法修改")
    return cfg.source


def update_config(cfg: AppConfig, changes: dict[tuple[str, ...], Any]) -> AppConfig:
    """按 {(表, …, 键): 值} 修改 config.toml（值为 None 表示删掉这一项），返回重新读取的配置。"""
    path = config_file(cfg)
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    for keys, value in changes.items():
        table: Any = doc
        for key in keys[:-1]:
            if key not in table:
                table[key] = tomlkit.table()
            table = table[key]
        if value is None:
            table.pop(keys[-1], None)
        else:
            table[keys[-1]] = value
    fd, temp = tempfile.mkstemp(prefix=".config-", suffix=".toml", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            file.write(tomlkit.dumps(doc))
        load_config(Path(temp))  # 试读：配置有误时抛出 ConfigError，原文件不动
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return load_config(path)


def set_steps(cfg: AppConfig, steps: dict[str, str]) -> AppConfig:
    """各步骤用哪个模型档案；档案必须存在并具备这一步需要的能力。"""
    for step, name in steps.items():
        if step not in STEP_LABELS:
            raise ConfigError(f"未知的步骤：{step}")
        cfg.profile_for_step(step, name)
    return update_config(cfg, {("steps", step): name for step, name in steps.items()})


def set_asr(cfg: AppConfig, values: dict[str, str]) -> AppConfig:
    """转写方案、本地模型的预设和型号、设备、模型位置、大模型转写用的档案。"""
    from .asr.funasr_backend import PRESETS
    from .asr.models import QWEN3_REPOS

    choices = {"backend": ASR_BACKENDS, "funasr_preset": PRESETS, "qwen3_size": QWEN3_REPOS}
    changes: dict[tuple[str, ...], Any] = {}
    for key, raw in values.items():
        value = clean_value(raw)
        if key in choices and value not in choices[key]:
            raise ConfigError(f"[asr] {key} 的取值不对：{value}")
        if key == "device" and not _DEVICE.match(value):
            raise ConfigError("设备只能是 auto、cpu 或 cuda:0 这样的写法")
        if key == "llm_profile" and not cfg.profile(value).audio:
            raise ConfigError(f"模型档案「{value}」不支持音频输入，不能用来转写")
        if key not in (*choices, "device", "llm_profile", "model_dir"):
            raise ConfigError(f"不能在这里修改 [asr] {key}")
        changes[("asr", key)] = value
    return update_config(cfg, changes)


def set_general(cfg: AppConfig, digest_workers: int) -> AppConfig:
    if not 1 <= digest_workers <= MAX_DIGEST_WORKERS:
        raise ConfigError(f"逐页识别的并发数要在 1–{MAX_DIGEST_WORKERS} 之间")
    return update_config(cfg, {("materials", "digest_workers"): digest_workers})


def set_profile_model(cfg: AppConfig, name: str, model: str) -> AppConfig:
    """型号：档案设置了 model_env 时写进 .env（优先于 config.toml），否则写进 config.toml。"""
    profile = cfg.profile(name)
    value = clean_value(model)
    if profile.model_env:
        set_env(cfg, profile.model_env, value)
        return load_config(config_file(cfg))
    return update_config(cfg, {("models", name, "model"): value})


def set_profile_base_url(cfg: AppConfig, name: str, url: str) -> AppConfig:
    """接口地址：留空表示直连官方接口。档案设置了 base_url_env 时写进 .env，否则写进 config.toml。"""
    profile = cfg.profile(name)
    value = clean_value(url)
    if value and not re.match(r"^https?://[^\s/]+", value):
        raise ConfigError("接口地址要以 http:// 或 https:// 开头")
    if profile.base_url_env:
        set_env(cfg, profile.base_url_env, value)
        return load_config(config_file(cfg))
    return update_config(cfg, {("models", name, "base_url"): value or None})


# ---------- .env 与密钥 ----------


def env_file(cfg: AppConfig) -> Path:
    return cfg.root / ".env"


def set_env(cfg: AppConfig, name: str, value: str) -> None:
    """修改 .env 里的一项（没有就追加；值为空表示清空），其余行原样保留，并同步当前进程的环境变量。"""
    check_name(name)
    value = clean_value(value)
    path = env_file(cfg)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=")
    matches = [index for index, line in enumerate(lines) if pattern.match(line)]
    line = f"{name}={_quoted(value)}"
    if matches:  # 改第一行，重复的其余几行去掉
        first = matches[0]
        lines = [line if index == first else text for index, text in enumerate(lines) if index not in matches[1:]]
    else:
        lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)


def _quoted(value: str) -> str:
    """含空格、#、引号时按 python-dotenv 的规则加双引号，否则原样写。"""
    if not _NEEDS_QUOTES.search(value):
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def key_names(cfg: AppConfig) -> list[str]:
    return sorted({profile.api_key_env for profile in cfg.models.values() if profile.api_key_env})


def env_keys(cfg: AppConfig) -> list[str]:
    """.env 里填了值的密钥（只返回变量名）。"""
    values = dotenv_values(env_file(cfg)) if env_file(cfg).is_file() else {}
    return [name for name in key_names(cfg) if (values.get(name) or "").strip()]


def save_key(cfg: AppConfig, name: str, value: str) -> None:
    """密钥存进凭据管理器；.env 里同名的旧值清空。"""
    if name not in key_names(cfg):
        raise ConfigError(f"没有哪个模型档案使用密钥 {name}")
    set_secret(name, value)
    if name in env_keys(cfg):
        set_env(cfg, name, "")


def move_env_keys(cfg: AppConfig) -> list[str]:
    """把 .env 里各档案的密钥移到凭据管理器，返回移了哪些（变量名）。由使用者在设置页点击执行。"""
    values = dotenv_values(env_file(cfg))
    moved = []
    for name in env_keys(cfg):
        set_secret(name, str(values[name]))
        set_env(cfg, name, "")
        moved.append(name)
    return moved
