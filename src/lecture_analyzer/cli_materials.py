"""命令行：课程资料（导入、列表、处理、删除）、讲对应的课件页、考前报告。

由 cli.py 注册：资料命令是 `lec material …`，另外两个是 `lec lecture pages` 和 `lec report`。
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.markup import escape
from rich.table import Table

from .cli_common import ConfigOption, console, friendly_errors, load_cfg, open_library, progress
from .library import SLIDES_LABELS, LibraryError
from .materials import MATERIAL_KINDS, MaterialStore

material_app = typer.Typer(help="管理课程资料：课件、教材、往年真题、模拟题、复习提纲……", no_args_is_help=True)


# ---------------- 课程资料 ----------------


class MaterialKind(str, Enum):
    slides = "slides"
    textbook = "textbook"
    past_exam = "past_exam"
    mock_exam = "mock_exam"
    exercise = "exercise"
    review_outline = "review_outline"
    notice = "notice"
    other = "other"


KIND_HELP = "资料类型：" + "；".join(f"{key} {label}" for key, label in MATERIAL_KINDS.items())


@material_app.command("add")
def material_add(
    course: Annotated[str, typer.Argument(help="课程名")],
    paths: Annotated[list[Path], typer.Argument(exists=True, help="文件、文件夹或压缩包（zip / rar / 7z），可以给多个")],
    kind: Annotated[MaterialKind, typer.Option(help=KIND_HELP)],
    title: Annotated[str, typer.Option(help="标题（只导入一份资料时有效）；不填用文件名")] = "",
    process: Annotated[bool, typer.Option("--process/--no-process", help="导入后立即逐页识别（调用模型）")] = True,
    config: ConfigOption = None,
) -> None:
    """导入课程资料。每个文档是一份资料，同一文件夹里的图片合成一份；课件会在运行时自动匹配到各讲。"""
    from .pipeline.ingest import import_materials
    from .pipeline.materials import process_materials

    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        target = library.course(course)
        result = import_materials(library, cfg.data_dir, target, [p.resolve() for p in paths], kind.value, title,
                                  report=progress)
        for source, reason in result.skipped:
            console.print(f"[yellow]跳过 {escape(source)}：{escape(reason)}[/yellow]")
        if process:
            process_materials(cfg, library, target, result.materials, report=progress)
    console.print(f"[green]完成[/green]：导入 {len(result.materials)} 份资料；`lec material list {escape(course)}` 查看")


@material_app.command("list")
def material_list(course: Annotated[str, typer.Argument(help="课程名")], config: ConfigOption = None) -> None:
    """列出课程资料和处理进度。"""
    from .pipeline.materials import stale_materials

    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        target = library.course(course)
        store = MaterialStore(library)
        rows = [(m, len(store.pages(m.id)), len(store.items(m.id))) for m in store.list(target)]
        stale = {m.id for m in stale_materials(library, target)}
    table = Table(title=f"「{target.name}」的资料（课件：{SLIDES_LABELS[target.slides]}）")
    for column in ("编号", "类型", "标题", "页数", "识别", "题目", "来源"):
        table.add_column(column, overflow="fold")
    for material, digested, items in rows:
        state = "完成" if material.step_state("digest") == "done" else f"{digested}/{material.page_count} 页"
        if material.step_state("items") != "done":
            item_state = "-"
        else:
            item_state = f"{items} 条" + ("（知识点对应待更新）" if material.id in stale else "")
        table.add_row(str(material.id), material.kind_label, escape(material.title), str(material.page_count), state,
                      item_state, escape(material.source))
    console.print(table)


@material_app.command("process")
def material_process(
    course: Annotated[str, typer.Argument(help="课程名")], config: ConfigOption = None
) -> None:
    """继续处理没处理完的资料（逐页识别中断后，只识别还没识别的页）。"""
    from .pipeline.materials import process_materials

    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        process_materials(cfg, library, library.course(course), report=progress)
    console.print("[green]完成[/green]")


@material_app.command("remove")
def material_remove(
    course: Annotated[str, typer.Argument(help="课程名")],
    material_id: Annotated[int, typer.Argument(help="资料编号（`lec material list` 里的编号）")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="不再确认")] = False,
    config: ConfigOption = None,
) -> None:
    """删除一份资料（书库里的副本、识别结果和题目；原文件不受影响）。"""
    from .pipeline.ingest import remove_material

    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        target = library.course(course)
        material = MaterialStore(library).get(material_id)
        if material.course_id != target.id:
            raise LibraryError(f"编号 {material_id} 的资料不属于「{target.name}」")
        if not yes and not typer.confirm(f"删除「{material.title}」（{material.page_count} 页）及其识别结果？"):
            raise typer.Exit(1)
        affected = MaterialStore(library).lectures_using(material_id)
        remove_material(library, cfg.data_dir, material_id)
    console.print(f"已删除「{escape(material.title)}」")
    if affected:
        console.print(f"[yellow]{len(affected)} 讲用到了这份资料，运行 `lec run {escape(course)}` 重新匹配课件[/yellow]")


# ---------------- 讲对应的课件页 ----------------


def lecture_pages(
    course: Annotated[str, typer.Argument(help="课程名")],
    seq: Annotated[int, typer.Argument(help="第几讲")],
    pages: Annotated[
        Optional[str],
        typer.Option(help="手动指定对应的课件页，如 \"3:1-30,4:1-5\"（资料编号:起止页，资料编号见 `lec material list`；"
                          "只写编号表示整份资料）"),
    ] = None,
    auto: Annotated[bool, typer.Option("--auto", help="取消手动指定，下次运行时重新自动匹配")] = False,
    config: ConfigOption = None,
) -> None:
    """查看或修改一讲对应的课件页。改动后运行 `lec run 课程 讲次` 重新整理笔记。"""
    from .pipeline.slides import describe_pages, parse_page_spec

    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        target = library.course(course)
        lecture = library.lecture(target, seq)
        store = MaterialStore(library)
        materials = store.list(target)
        if pages is not None:
            try:
                refs = parse_page_spec(pages, materials)
            except ValueError as exc:
                raise LibraryError(str(exc)) from None
            store.set_lecture_pages(lecture.id, refs, "manual")
            library.reset_steps(lecture.id, ["slides"])
        elif auto:
            store.set_lecture_pages(lecture.id, [], "auto")
            library.reset_steps(lecture.id, ["slides"])
        current = store.lecture_pages(lecture.id)
    titles = {m.id: m.title for m in materials}
    how = "手动指定" if current and current[0].source == "manual" else "自动匹配"
    summary = describe_pages([page.ref for page in current], titles) if current else "还没有对应的课件页"
    console.print(f"第 {seq} 讲（{how}）：{escape(summary)}")
    if pages is not None or auto:
        console.print(f"运行 `lec run {escape(course)} {seq}` 按新的课件页重新整理笔记")


# ---------------- 考前报告 ----------------


def report(
    course: Annotated[str, typer.Argument(help="课程名")],
    lectures: Annotated[Optional[str], typer.Option(help="只分析这些讲，如 1,2,5；不填用全部讲")] = None,
    tag: Annotated[Optional[str], typer.Option(help="只分析带这个标签的讲")] = None,
    materials: Annotated[Optional[str], typer.Option(help="只用这些资料（编号见 `lec material list`），如 3,4；不填用全部")] = None,
    models: Annotated[
        Optional[str], typer.Option(help="用哪些模型档案排名，逗号分隔；不填用 [steps] report")
    ] = None,
    config: ConfigOption = None,
) -> None:
    """生成考前报告：各模型分别排名后取共识，按可能考到的程度列出知识点和要点（中文，取自总复习）。"""
    from .pipeline.report import build_plan, generate_report, parse_numbers, save_report

    cfg = load_cfg(config)
    names = [name.strip() for name in (models or "").replace("，", ",").split(",") if name.strip()]
    with friendly_errors(), open_library(cfg) as library:
        target = library.course(course)
        plan = build_plan(cfg, library, target, parse_numbers(lectures, "讲次"), tag or "",
                          parse_numbers(materials, "资料编号"), names)
        console.print(f"将调用模型排名 {plan.calls} 次（{'、'.join(plan.profiles)} 各一次），再更新相关章节的总复习")
        result = generate_report(cfg, library, plan, report=progress)
        report_id, path = save_report(library, cfg.data_dir, plan, result)
    for warning in result["warnings"]:
        console.print(f"[yellow]注意：{escape(warning)}[/yellow]")
    console.print(f"[green]完成[/green]：考前报告 #{report_id} 已导出到 {escape(str(path))}，也可以在书架上查看")
