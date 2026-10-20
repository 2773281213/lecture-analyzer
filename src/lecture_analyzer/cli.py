"""命令行入口：lec。"""

from __future__ import annotations

import webbrowser
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.markup import escape
from rich.table import Table

from .cli_common import ConfigOption, console, friendly_errors, load_cfg, open_library, progress
from .cli_materials import lecture_pages, material_app, report
from .asr import language_problem
from .config import AppConfig, ConfigError
from .course import CourseSettings, build_settings
from .library import SLIDES_LABELS, LibraryError
from .llm import LLMError, ModelProfile, create_provider
from .timecode import format_ms

app = typer.Typer(help="课程录音分析与考点预测工具", no_args_is_help=True, add_completion=False)
course_app = typer.Typer(help="管理课程（一门课是一本书）", no_args_is_help=True)
lecture_app = typer.Typer(help="管理讲（书里的章；一讲可以由多节录音录像组成）", no_args_is_help=True)
app.add_typer(course_app, name="course")
app.add_typer(lecture_app, name="lecture")
app.add_typer(material_app, name="material")
lecture_app.command("pages")(lecture_pages)
app.command("report")(report)


@app.callback()
def main() -> None:
    """课程录音分析与考点预测工具。"""


@app.command()
def models(
    config: ConfigOption = None,
    ping: Annotated[bool, typer.Option("--ping", help="给每个已就绪的模型发一个最小请求，检查密钥和端点")] = False,
    remote: Annotated[Optional[str], typer.Option("--remote", help="列出该档案对应服务端的可用型号")] = None,
) -> None:
    """列出模型档案、能力和状态。"""
    cfg = load_cfg(config)
    if remote:
        _print_remote_models(cfg, remote)
        return

    table = Table(title="模型档案")
    for column in ("档案", "接入方式", "型号", "能力", "接口地址", "状态"):
        table.add_column(column, overflow="fold")  # 终端较窄时换行显示，不截断
    for name, profile in cfg.models.items():
        status = _profile_status(profile)
        if ping and status == "就绪":
            status = _ping(profile)
        endpoint = escape(profile.resolved_base_url() or "官方")
        table.add_row(name, profile.adapter, profile.model or "-", _capabilities(profile), endpoint, status)
    console.print(table)

    steps = Table(title="步骤分配")
    steps.add_column("步骤")
    steps.add_column("模型档案")
    steps.add_column("effort")
    for step, name in cfg.steps.items():
        steps.add_row(step, name, cfg.effort_for_step(step) or "-")
    console.print(steps)

    for problem in cfg.check():
        console.print(f"[yellow]注意：{escape(problem)}[/yellow]")


def _profile_status(profile: ModelProfile) -> str:
    if not profile.is_configured:
        return "未配置（model 为空）"
    if profile.api_key_env and not profile.api_key():
        return f"缺少密钥 {profile.api_key_env}"
    return "就绪"


def _capabilities(profile: ModelProfile) -> str:
    caps = [profile.structured]
    if profile.vision:
        caps.append("看图")
    if profile.audio:
        caps.append("听音频")
    context = profile.max_context
    caps.append(f"{context / 1_000_000:g}M 上下文" if context >= 1_000_000 else f"{context // 1000}K 上下文")
    return "、".join(caps)


def _ping(profile: ModelProfile) -> str:
    from .llm.probe import ping

    ok, text = ping(profile)
    return f"[green]可用[/green]（{escape(text)}）" if ok else f"[red]失败：{escape(text)}[/red]"


def _print_remote_models(cfg: AppConfig, name: str) -> None:
    try:
        profile = cfg.profile(name)
        # 列型号不需要 model 字段，用占位型号创建适配器
        probe = profile.model_copy(update={"model": profile.model or "-"})
        names = create_provider(probe).list_remote_models()
    except (ConfigError, LLMError, NotImplementedError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    except Exception as exc:  # noqa: BLE001 —— 同上，网络或鉴权错误
        console.print(f"[red]查询失败：{_short(exc)}[/red]")
        raise typer.Exit(1) from exc
    for model_name in sorted(names):
        console.print(model_name, markup=False)


def _short(exc: Exception) -> str:
    """异常信息截短并转义，避免其中的方括号被 rich 当成样式标记。"""
    from .llm.probe import short_error

    return escape(short_error(exc))


# ---------------- 课程（书） ----------------


class LectureKind(str, Enum):
    normal = "normal"
    exercise = "exercise"
    review = "review"


InstructionsFile = Annotated[
    Optional[Path],
    typer.Option("--instructions-file", exists=True, dir_okay=False,
                 help="课程提示词文件（UTF-8 文本），整理笔记、整合和总复习时附加到提示词里"),
]
TEACHING_HELP = "授课语言代码（ISO 639），老师混用几种语言时用逗号分隔，如 zh、en、ko,en；旧写法 ko+zh 等于 --teaching ko --interpreted-to zh"
INTERPRETED_HELP = "现场翻译成的语言代码，如 zh；没有现场翻译时不填"
NO_SLIDES_HELP = "声明这门课不提供课件（分析前必须先导入课件或作此声明）"


@course_app.command("add")
def course_add(
    name: Annotated[str, typer.Argument(help="课程名，例如 药理学")],
    subject: Annotated[str, typer.Option(help="学科，写进提示词，例如 药理学、病毒学")] = "医学",
    teaching: Annotated[str, typer.Option(help=TEACHING_HELP)] = "zh",
    interpreted_to: Annotated[str, typer.Option("--interpreted-to", help=INTERPRETED_HELP)] = "",
    no_slides: Annotated[bool, typer.Option("--no-slides", help=NO_SLIDES_HELP)] = False,
    asr: Annotated[Optional[str], typer.Option(help="这门课专用的转写方案，如 llm:gemini；不填用 config.toml 的 [asr]")] = None,
    instructions_file: InstructionsFile = None,
    config: ConfigOption = None,
) -> None:
    """添加课程（一本书）。笔记和分析结果一律用中文。"""
    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        settings = build_settings(subject=subject, teaching=teaching, interpreted_to=interpreted_to)
        _check_asr(cfg, asr, settings)
        course = library.add_course(name, settings, asr, _read_instructions(instructions_file))
        if no_slides:
            library.update_course(course.id, slides="none")
    console.print(f"已添加课程「{escape(name)}」：{settings.teaching_description}")
    if not no_slides:
        console.print(f"开始分析前请先导入课件（`lec material add {escape(name)} 文件或压缩包 --kind slides`），"
                      f"或声明不提供（`lec course set {escape(name)} --no-slides`）")


@course_app.command("set")
def course_set(
    name: Annotated[str, typer.Argument(help="课程名")],
    instructions_file: InstructionsFile = None,
    clear_instructions: Annotated[bool, typer.Option("--clear-instructions", help="清空课程提示词")] = False,
    subject: Annotated[Optional[str], typer.Option(help="学科")] = None,
    teaching: Annotated[Optional[str], typer.Option(help=TEACHING_HELP)] = None,
    interpreted_to: Annotated[
        Optional[str], typer.Option("--interpreted-to", help="现场翻译成的语言代码；填 none 表示没有现场翻译")
    ] = None,
    no_slides: Annotated[bool, typer.Option("--no-slides", help=NO_SLIDES_HELP)] = False,
    asr: Annotated[Optional[str], typer.Option(help="这门课专用的转写方案；填 default 表示改回 config.toml 的 [asr]")] = None,
    config: ConfigOption = None,
) -> None:
    """修改课程设置。已处理的讲按新设置重做：改了授课语言或现场翻译要 `lec run 课程 讲次 --force transcribe`；
    改了学科或提示词要 `--force extract`。"""
    cfg = load_cfg(config)
    with friendly_errors():
        fields = _course_fields(subject=subject, teaching=teaching, interpreted_to=interpreted_to)
        if instructions_file is not None or clear_instructions:
            fields["instructions"] = "" if clear_instructions else _read_instructions(instructions_file)
        if no_slides:
            fields["slides"] = "none"
        if asr is not None:
            fields["asr"] = None if asr == "default" else asr
        if not fields:
            console.print("[yellow]没有要修改的设置[/yellow]")
            raise typer.Exit(1)
        with open_library(cfg) as library:
            old = library.course(name)
            changed = {key: fields[key] for key in ("subject", "teaching", "interpreted_to") if key in fields}
            settings = build_settings(**{**old.settings.model_dump(), **changed})
            _check_asr(cfg, fields.get("asr", old.asr), settings)
            course = library.update_course(old.id, **fields)
    settings = course.settings
    console.print(f"已更新课程「{escape(course.name)}」：{settings.subject}，{settings.teaching_description}，"
                  f"课件{SLIDES_LABELS[course.slides]}，课程提示词 {len(course.instructions)} 字")


def _check_asr(cfg: AppConfig, asr: str | None, settings: CourseSettings) -> None:
    """转写方案不支持课程录音里的语言时拦下（与网页的课程设置同一条规则）。"""
    problem = language_problem(cfg, asr, settings)
    if problem:
        raise LibraryError(problem)


def _course_fields(**options: str | None) -> dict[str, object]:
    """命令行给了的课程设置项；--interpreted-to none 表示去掉现场翻译。"""
    fields: dict[str, object] = {key: value for key, value in options.items() if value is not None}
    if str(fields.get("interpreted_to", "")).lower() == "none":
        fields["interpreted_to"] = ""
    return fields


def _read_instructions(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LibraryError(f"课程提示词文件需要是 UTF-8 编码：{path}") from exc


@course_app.command("remove")
def course_remove(
    name: Annotated[str, typer.Argument(help="课程名")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="不再确认")] = False,
    config: ConfigOption = None,
) -> None:
    """删除一门课：书库里的讲、笔记、课程记忆、资料副本、报告和问答都会删除；原始音视频不受影响。"""
    from .pipeline.organize import delete_course

    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        course = library.course(name)
        count = len(library.lectures(course))
        if not yes and not typer.confirm(f"删除「{course.name}」（{count} 讲）的全部数据？原始音视频不会删除"):
            raise typer.Exit(1)
        delete_course(library, cfg.data_dir, course)
    console.print(f"已删除课程「{escape(name)}」")


@course_app.command("list")
def course_list(config: ConfigOption = None) -> None:
    """列出全部课程。"""
    cfg = load_cfg(config)
    with open_library(cfg) as library:
        rows = [(course, len(library.lectures(course))) for course in library.courses()]
    table = Table(title="课程")
    for column in ("课程", "学科", "授课方式", "课件", "转写方案", "课程提示词", "讲数"):
        table.add_column(column)
    for course, count in rows:
        settings = course.settings
        table.add_row(course.name, settings.subject, settings.teaching_description,
                      SLIDES_LABELS[course.slides], course.asr or "按 config.toml",
                      f"{len(course.instructions)} 字" if course.instructions else "-", str(count))
    console.print(table)


# ---------------- 讲（章） ----------------


MediaOption = Annotated[
    list[Path],
    typer.Option("--media", exists=True, dir_okay=False, help="录音或录像文件；一讲有多节时重复 --media，按顺序是第 1、2… 节"),
]


@lecture_app.command("add")
def lecture_add(
    course: Annotated[str, typer.Argument(help="课程名")],
    seq: Annotated[int, typer.Option("--seq", help="第几讲")],
    media: MediaOption,
    slides: Annotated[
        Optional[Path],
        typer.Option(exists=True, dir_okay=False, help="这一讲的课件（PPTX/PDF 等）：导入课程资料，并固定对应到这一讲"),
    ] = None,
    title: Annotated[str, typer.Option(help="标题；不填则整理笔记后自动生成")] = "",
    date: Annotated[str, typer.Option(help="上课日期，如 2026-10-14")] = "",
    kind: Annotated[LectureKind, typer.Option(help="normal 普通课 / exercise 习题课 / review 复习课")] = LectureKind.normal,
    config: ConfigOption = None,
) -> None:
    """添加一讲。原始音视频不会被复制，只记录路径；课件会复制进课程资料库。"""
    from .pipeline.ingest import import_materials, pin_lecture_pages

    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        target = library.course(course)
        lecture = library.add_lecture(target, seq, [path.resolve() for path in media], title=title, date=date,
                                      kind=kind.value)
        if slides is not None:
            imported = import_materials(library, cfg.data_dir, target, [slides.resolve()], "slides", report=progress)
            pin_lecture_pages(library, lecture.id, imported.materials[0])
    parts = f"（{len(lecture.parts)} 节）" if len(lecture.parts) > 1 else ""
    console.print(f"已添加「{escape(course)}」第 {lecture.seq} 讲{parts}，运行 `lec run {escape(course)} {seq}` 开始处理")


@lecture_app.command("add-part")
def lecture_add_part(
    course: Annotated[str, typer.Argument(help="课程名")],
    seq: Annotated[int, typer.Argument(help="第几讲")],
    media: MediaOption,
    config: ConfigOption = None,
) -> None:
    """给已有的一讲追加节（按顺序接在后面）。已转写的节不会重新转写。"""
    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        lecture = library.lecture(library.course(course), seq)
        for path in media:
            lecture = library.add_part(lecture.id, path.resolve())
    console.print(f"第 {seq} 讲现在共 {len(lecture.parts)} 节，运行 `lec run {escape(course)} {seq}` 重新处理")


@lecture_app.command("move")
def lecture_move(
    course: Annotated[str, typer.Argument(help="现在所在的课程")],
    seq: Annotated[int, typer.Argument(help="第几讲")],
    to: Annotated[str, typer.Option("--to", help="移到哪门课")],
    new_seq: Annotated[Optional[int], typer.Option("--seq", help="在新课程里是第几讲；不填沿用原讲次，冲突时排到最后")] = None,
    config: ConfigOption = None,
) -> None:
    """把放错课程的一讲移到正确的课程。"""
    from .pipeline.organize import move_lecture
    from .pipeline.run import STEP_LABELS

    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        lecture = library.lecture(library.course(course), seq)
        result = move_lecture(library, cfg.data_dir, lecture, library.course(to), new_seq)
    moved = result.lecture
    console.print(f"已移到「{escape(to)}」第 {moved.seq} 讲；运行 `lec run {escape(to)} {moved.seq}` "
                  f"从「{STEP_LABELS[result.rerun_from]}」起重新处理")
    if result.stale:
        console.print("[yellow]这些讲的课程记忆待更新：" + "、".join(f"第 {x.seq} 讲" for x in result.stale) + "[/yellow]")


@lecture_app.command("remove")
def lecture_remove(
    course: Annotated[str, typer.Argument(help="课程名")],
    seq: Annotated[int, typer.Argument(help="第几讲")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="不再确认")] = False,
    config: ConfigOption = None,
) -> None:
    """删除一讲（转写、笔记、导出文件和中间文件）；原始音视频不受影响。后面整合过的讲标记为课程记忆待更新。"""
    from .pipeline.organize import delete_lecture

    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        lecture = library.lecture(library.course(course), seq)
        if not yes and not typer.confirm(f"删除「{course}」第 {seq} 讲的全部数据？原始音视频不会删除"):
            raise typer.Exit(1)
        stale = delete_lecture(library, cfg.data_dir, lecture)
    console.print(f"已删除「{escape(course)}」第 {seq} 讲")
    if stale:
        console.print(f"[yellow]这些讲的课程记忆待更新，运行 `lec rebuild {escape(course)}`：" +
                      "、".join(f"第 {x.seq} 讲" for x in stale) + "[/yellow]")


@lecture_app.command("list")
def lecture_list(
    course: Annotated[Optional[str], typer.Argument(help="只看这门课")] = None, config: ConfigOption = None
) -> None:
    """列出各讲和处理进度。"""
    from .pipeline.run import STEPS

    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        target = library.course(course) if course else None
        names = {c.id: c.name for c in library.courses()}
        lectures = library.lectures(target)
    table = Table(title="讲")
    for column in ("课程", "讲次", "标题", "节", "时长", "进度", "文件"):
        table.add_column(column, overflow="fold")
    for lecture in lectures:
        done = [step for step, info in lecture.status.items() if info.get("state") in ("done", "skipped")]
        failed = [step for step, info in lecture.status.items() if info.get("state") == "failed"]
        progress = f"[red]失败：{', '.join(failed)}[/red]" if failed else f"{len(done)}/{len(STEPS)} 步"
        if lecture.step_state("integrate") == "stale":
            progress += "（课程记忆待更新）"
        duration = format_ms(lecture.duration_ms) if lecture.duration_ms else "-"
        files = "\n".join(escape(Path(part.media_path).name) for part in lecture.parts)
        table.add_row(names[lecture.course_id], str(lecture.seq), escape(lecture.title or "-"), str(len(lecture.parts)),
                      duration, progress, files)
    console.print(table)


# ---------------- 处理、课程记忆、书架、费用 ----------------


@app.command()
def run(
    course: Annotated[str, typer.Argument(help="课程名")],
    seq: Annotated[Optional[int], typer.Argument(help="第几讲；不填则按讲次依次处理所有没完成的讲")] = None,
    model: Annotated[Optional[str], typer.Option(help="临时指定整理笔记和整合课程记忆用的模型档案")] = None,
    asr: Annotated[Optional[str], typer.Option(help="临时指定转写方案，如 llm:gemini、qwen3:1.7b")] = None,
    force: Annotated[Optional[list[str]], typer.Option(help="强制重跑某一步及依赖它的步骤（可重复）；all 表示全部重跑")] = None,
    config: ConfigOption = None,
) -> None:
    """处理讲：抽音频 → 转写 → 匹配课件 → 整理笔记 → 校验原话 → 整合课程记忆 → 上架。可以断点续跑。"""
    from .pipeline.run import STEPS, RunOptions, check_slides_choice, process_lectures

    cfg = load_cfg(config)
    forced = set(force or [])
    unknown = forced - set(STEPS) - {"all"}
    if unknown:
        console.print(f"[red]未知的步骤：{', '.join(sorted(unknown))}（可选：{', '.join(STEPS)}、all）[/red]")
        raise typer.Exit(1)
    options = RunOptions(force=frozenset(STEPS if "all" in forced else forced), model=model, asr=asr)
    with friendly_errors(), open_library(cfg) as library:
        target = library.course(course)
        check_slides_choice(target)  # 课件关卡：先选择是否提供课件
        lectures = None if seq is None else [library.lecture(target, seq)]
        processed = process_lectures(
            cfg, library, target, lectures, options, progress,
            announce=lambda lecture: console.print(f"[bold]「{escape(course)}」第 {lecture.seq} 讲[/bold]"),
        )
    if not processed:
        console.print("所有讲都已处理完成")
        return
    console.print("[green]完成[/green]：运行 `lec serve` 打开书架阅读")


@app.command()
def rebuild(
    course: Annotated[str, typer.Argument(help="课程名")],
    from_seq: Annotated[Optional[int], typer.Option("--from", help="从第几讲起重新整合；不填则从第一讲需要更新的讲开始")] = None,
    model: Annotated[Optional[str], typer.Option(help="临时指定整合用的模型档案")] = None,
    config: ConfigOption = None,
) -> None:
    """重建课程记忆：前面的讲直接重放已有结果，从起点开始的讲逐讲重新整合，然后重新导出。"""
    from .pipeline.run import rebuild_course

    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        outcome = rebuild_course(cfg, library, library.course(course), from_seq, model, report=progress)
    if outcome is None:
        console.print("课程记忆已是最新，不需要重建")
        return
    console.print(f"[green]完成[/green]：重放 {outcome.replayed} 讲、重新整合 {len(outcome.integrated)} 讲，已重新导出")


@app.command()
def review(
    course: Annotated[str, typer.Argument(help="课程名")],
    regenerate_all: Annotated[bool, typer.Option("--all", help="所有章节都重新生成")] = False,
    model: Annotated[Optional[str], typer.Option(help="临时指定写复习讲义用的模型档案")] = None,
    config: ConfigOption = None,
) -> None:
    """生成或更新这门课的总复习讲义：按章节调用模型，内容没变的章节不重新生成。"""
    from .books import export_course
    from .pipeline.materials import ensure_mapped
    from .pipeline.review import generate_review, plan_review, review_context

    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        target = library.course(course)
        remapped = ensure_mapped(cfg, library, target, report=progress)  # 试题与知识点的对应过期时先更新（星级要用）
        planned = plan_review(library, target, regenerate_all)
        if not planned:
            console.print("这门课还没有课程记忆：先处理至少一讲")
            return
        needed = [p for p in planned if p.needed]
        if not needed:
            if remapped:
                export_course(library, cfg.data_dir, target)
            note = "，已按课程资料更新星级并重新导出" if remapped else ""
            console.print(f"复习讲义已是最新，不需要重新生成{note}（要全部重写请加 --all）")
            return
        console.print(f"将调用模型生成 {len(needed)} 章（共 {len(planned)} 章）：" + "、".join(p.input.chapter for p in needed))
        generate_review(review_context(cfg, library, progress, model), target, planned)
        export_course(library, cfg.data_dir, target)
    console.print("[green]完成[/green]：在书架上打开这门课的「总复习」，或查看导出的 总复习.md")


@app.command()
def serve(
    port: Annotated[int, typer.Option(help="端口")] = 8765,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="启动后用浏览器打开")] = True,
    config: ConfigOption = None,
) -> None:
    """启动本地网页书架（只监听本机 127.0.0.1）。"""
    import uvicorn

    from .web import create_app

    cfg = load_cfg(config)
    url = f"http://127.0.0.1:{port}/"
    console.print(f"书架地址：{url}（按 Ctrl+C 停止）")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=port, log_level="warning")


@app.command()
def desktop(config: ConfigOption = None) -> None:
    """打开桌面窗口（安装版双击「课程书架」即可；开发时用这个命令）。还没有书库时先显示首次设置。"""
    from .desktop import main as desktop_main

    desktop_main(config)


@app.command()
def cost(
    course: Annotated[Optional[str], typer.Argument(help="只看这门课")] = None, config: ConfigOption = None
) -> None:
    """按步骤和模型汇总 token 用量与估算费用（按模型档案里的单价估算）。"""
    cfg = load_cfg(config)
    with friendly_errors(), open_library(cfg) as library:
        rows = library.cost_summary(library.course(course) if course else None)
    table = Table(title="模型用量")
    for column in ("步骤", "模型", "调用次数", "输入 token", "输出 token", "估算费用（美元）"):
        table.add_column(column)
    for row in rows:
        estimated = f"{row['cost_usd']:.4f}" if row["cost_usd"] is not None else "-"
        table.add_row(row["step"], row["model"], str(row["calls"]), f"{row['input_tokens']:,}",
                      f"{row['output_tokens']:,}", estimated)
    console.print(table)


if __name__ == "__main__":
    app(prog_name="lec")  # 安装版的 lec.cmd 用 python -m 启动
