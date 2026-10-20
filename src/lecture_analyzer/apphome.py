"""书库文件夹的位置（桌面程序和安装版的命令行用）。

书库文件夹里放 config.toml、.env 和 data/；选定的位置记在 %APPDATA%\\LectureAnalyzer\\settings.json。
开发环境照旧：命令行先按 --config、LEC_CONFIG、当前目录的 config.toml 查找，找不到才用这里记下的书库。
"""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path
from typing import Any

import tomlkit

APP_NAME = "LectureAnalyzer"
CONFIG_NAME = "config.toml"
ENV_HEADER = "# 接口地址和型号（由「课程书架」的设置页维护）；密钥保存在 Windows 凭据管理器，不写在这里\n"


def settings_file() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / APP_NAME / "settings.json"


def read_settings() -> dict[str, Any]:
    try:
        data = json.loads(settings_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_settings(data: dict[str, Any]) -> None:
    path = settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def is_library_folder(path: Path) -> bool:
    return (path / CONFIG_NAME).is_file()


def saved_home() -> Path | None:
    """记下的书库文件夹；没有记录或文件夹已不存在时返回 None。"""
    value = read_settings().get("home")
    path = Path(value) if isinstance(value, str) and value else None
    return path if path is not None and is_library_folder(path) else None


def default_home() -> Path:
    return Path.home() / "Documents" / "课程书架"


def remember_home(path: Path) -> Path:
    path = path.resolve()
    write_settings({**read_settings(), "home": str(path)})
    return path


def create_home(path: Path) -> Path:
    """新建书库文件夹：示例配置（本地模型下载到书库里的 models）、.env、data/。已有的文件不覆盖。返回配置文件。"""
    path.mkdir(parents=True, exist_ok=True)
    config = path / CONFIG_NAME
    if not config.exists():
        document = tomlkit.parse(example_config_text())
        document["asr"]["model_dir"] = "models"  # type: ignore[index]
        config.write_text(tomlkit.dumps(document), encoding="utf-8")
    env = path / ".env"
    if not env.exists():
        env.write_text(ENV_HEADER, encoding="utf-8")
    (path / "data").mkdir(exist_ok=True)
    return config


def example_config_text() -> str:
    """示例配置：安装版从包里读（打包时放进 wheel），开发环境从仓库根目录读。"""
    packaged = resources.files("lecture_analyzer").joinpath("config.example.toml")
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    return (Path(__file__).resolve().parents[2] / "config.example.toml").read_text(encoding="utf-8")
