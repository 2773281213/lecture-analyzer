"""命令行各模块共用的工具：控制台输出、读取配置、打开书库、把使用者能处理的错误显示成说明。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markup import escape

from .archive import ArchiveError
from .asr.base import ASRError
from .config import AppConfig, ConfigError, load_config
from .course import SettingsError
from .documents import DocumentError
from .library import Library, LibraryError
from .llm import LLMError
from .media import MediaError
from .pipeline.memory import BudgetError

console = Console()

# 这些错误都是使用者能看懂、能处理的，直接显示说明，不打印堆栈
FRIENDLY_ERRORS = (
    ConfigError, LibraryError, SettingsError, LLMError, ASRError, MediaError, DocumentError, BudgetError,
    ArchiveError,
)

ConfigOption = Annotated[
    Optional[Path], typer.Option("--config", "-c", help="配置文件路径，默认读取当前目录的 config.toml")
]


def load_cfg(config: Path | None) -> AppConfig:
    try:
        return load_config(config)
    except ConfigError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc


@contextmanager
def friendly_errors() -> Iterator[None]:
    try:
        yield
    except FRIENDLY_ERRORS as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc


def open_library(cfg: AppConfig) -> Library:
    return Library(cfg.data_dir / "lecture.db")


def progress(message: str) -> None:
    console.print(f"[dim]{datetime.now():%H:%M:%S}[/dim] {escape(message)}")
