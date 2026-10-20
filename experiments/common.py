"""M0 实验脚本的公共工具。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lecture_analyzer.course import CourseSettings, build_settings

from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console()

OUTPUT_ROOT = Path("experiments/output")


def add_course_args(parser: argparse.ArgumentParser) -> None:
    """课程设置参数：学科、授课语言、现场翻译、笔记语言（M1 起改为存在课程里）。"""
    parser.add_argument("--subject", default="医学", help="学科，如 医学、药理学")
    parser.add_argument("--teaching",
                        help="授课语言代码，几种混用时用逗号分隔，如 zh、ko、ko,en；旧写法 ko+zh = 韩语授课 + 现场中文翻译")
    parser.add_argument("--interpreted-to", default="", help="现场翻译成的语言代码，如 zh")
    parser.add_argument("--notes", default="zh", help="笔记语言代码，如 zh、ko、en")


def course_settings(args: argparse.Namespace) -> CourseSettings | None:
    """没指定授课语言时返回 None，转写时自动识别语种。"""
    if args.teaching is None:
        return None
    return build_settings(subject=args.subject, teaching=args.teaching, interpreted_to=args.interpreted_to,
                          note_language=args.notes)


def read_lines(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def print_rows(title: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    table = Table(title=title)
    for column in rows[0]:
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(*(_cell(value) for value in row.values()))
    console.print(table)


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if value < 1 else f"{value:.2f}"
    return "-" if value is None else escape(str(value))  # 转义方括号，避免被当成 rich 样式
