"""命令行：考试方案（`lec plan …`）——先弄清这门课怎么考，再分析知识点。

分析的结果是待确认的草稿；确认后才进入整理笔记、总复习、考前报告和问答。在命令行里确认时，方案内容变了
且已有讲整理过笔记，会提示用 `lec run 课程 --force extract` 按方案重新整理（网页上确认会自动提交）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.markup import escape

from .cli_common import ConfigOption, console, friendly_errors, load_cfg, open_library, progress
from .exam_plans import SOURCE_LABELS, ExamPlanStore, plan_text
from .library import EXAM_PLAN_LABELS

plan_app = typer.Typer(help="考试方案：考核方式、题型分值、答题要求、考试范围（先于知识点分析）", no_args_is_help=True)


@plan_app.command("show")
def show(
    course: Annotated[str, typer.Argument(help="课程名")],
    config: ConfigOption = None,
) -> None:
    """显示考试方案和每一条的出处。"""
    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        target = library.course(course)
        stored = ExamPlanStore(library).get(target)
    console.print(f"[bold]「{escape(target.name)}」考试方案：{EXAM_PLAN_LABELS[target.exam_plan]}[/bold]")
    if stored is None or stored.plan is None:
        console.print(f"还没有考试方案：`lec plan analyze {course}`（默认从第 1 讲分析），或在书架的课程页填写")
        return
    console.print(escape(plan_text(stored.plan) or "（没有内容）"))
    for group in (stored.plan.question_types, stored.plan.answer_rules, stored.plan.scope, stored.plan.components):
        for item in group:
            ev = item.evidence
            where = SOURCE_LABELS[ev.source] + (f" 第{ev.seq}讲" if ev.seq else "")
            check = "" if ev.verified else "（原话未核对上）"
            label = getattr(item, "name", "") or getattr(item, "text", "")
            if ev.quote:
                console.print(f"[dim]· {escape(label)} ← {where}：「{escape(ev.quote)}」{check}[/dim]")


@plan_app.command("analyze")
def analyze(
    course: Annotated[str, typer.Argument(help="课程名")],
    lecture: Annotated[Optional[int], typer.Option("--lecture", help="从第几讲分析（默认讲次最小的一讲）")] = None,
    text_file: Annotated[Optional[Path], typer.Option("--text-file", exists=True, dir_okay=False,
                                                      help="老师的考试说明（UTF-8 文本），最优先采用")] = None,
    config: ConfigOption = None,
) -> None:
    """分析考试方案（调用一次模型），结果为待确认的草稿。"""
    from .pipeline.exam_plan import analyze_exam_plan

    cfg = load_cfg(config)
    text = None
    if text_file is not None:
        try:
            text = text_file.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            console.print(f"[red]考试说明需要是 UTF-8 编码：{escape(str(text_file))}[/red]")
            raise typer.Exit(1) from exc
    with friendly_errors(), open_library(cfg) as library:
        target = library.course(course)
        lecture_id = library.lecture(target, lecture).id if lecture is not None else None
        analysis = analyze_exam_plan(cfg, library, target, lecture_id, text, report=progress)
    console.print(f"[green]已分析[/green]：{escape(analysis.summary)}")
    console.print(escape(plan_text(analysis.plan) or "（没有内容）"))
    console.print(f"核对后确认：`lec plan confirm {course}`（要修改请在书架的课程页里改）")


@plan_app.command("confirm")
def confirm(
    course: Annotated[str, typer.Argument(help="课程名")],
    config: ConfigOption = None,
) -> None:
    """确认考试方案：之后整理笔记、总复习、考前报告和问答都以它为依据。"""
    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        target, changed = ExamPlanStore(library).confirm(library.course(course))
        noted = sum(library.notes(x.id) is not None for x in library.lectures(target))
    console.print(f"[green]已确认[/green]「{escape(target.name)}」的考试方案")
    if changed and noted:
        console.print(f"已有 {noted} 讲按旧方案整理过笔记：运行 `lec run {course} --force extract` 按新方案重新整理")
    else:
        console.print(f"运行 `lec run {course}` 接着处理")


@plan_app.command("skip")
def skip(
    course: Annotated[str, typer.Argument(help="课程名")],
    config: ConfigOption = None,
) -> None:
    """跳过考试方案：后续分析不使用考试方案（可以随时再分析、确认）。"""
    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        target = ExamPlanStore(library).skip(library.course(course))
    console.print(f"已跳过「{escape(target.name)}」的考试方案：运行 `lec run {course}` 接着处理")
