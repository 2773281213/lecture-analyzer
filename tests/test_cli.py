from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from lecture_analyzer import cli, cli_common, cli_materials
from lecture_analyzer.cli import app
from lecture_analyzer.config import load_config
from lecture_analyzer.library import Library

runner = CliRunner()


@pytest.fixture(autouse=True)
def wide_console(monkeypatch):
    """测试环境不是终端，rich 默认只有 80 列，表格内容会换行，这里放宽以便断言。"""
    console = Console(width=200)
    for module in (cli, cli_common, cli_materials):
        monkeypatch.setattr(module, "console", console)


def test_models_lists_profiles_and_problems(example_config: Path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://relay.example/")
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    result = runner.invoke(app, ["models", "--config", str(example_config)])
    assert result.exit_code == 0, result.output
    assert "claude-opus-5-5" in result.output
    assert "https://relay.example/" in result.output  # Claude 走中转
    assert "官方" in result.output  # 其他档案直连
    assert "缺少密钥 ANTHROPIC_API_KEY" in result.output
    assert "未配置" in result.output
    assert "page_digest" in result.output


def test_models_without_config_exits_with_hint(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEC_CONFIG", raising=False)
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 1
    assert "config.example.toml" in result.output


def test_course_and_lecture_management(example_config: Path, tmp_path: Path):
    media = tmp_path / "病毒学2-1.mp4"
    media.write_bytes(b"fake")
    args = ["--config", str(example_config)]
    added = runner.invoke(app, ["course", "add", "病毒学", "--subject", "病毒学", "--teaching", "ko+zh", *args])
    assert added.exit_code == 0 and "老师用韩语授课" in added.output
    duplicate = runner.invoke(app, ["course", "add", "病毒学", *args])
    assert duplicate.exit_code == 1 and "已存在" in duplicate.output

    listed = runner.invoke(app, ["course", "list", *args])
    assert "病毒学" in listed.output and "韩语" in listed.output

    lecture = runner.invoke(app, ["lecture", "add", "病毒学", "--seq", "2", "--media", str(media), "--kind", "review", *args])
    assert lecture.exit_code == 0, lecture.output
    lectures = runner.invoke(app, ["lecture", "list", "病毒学", *args])
    assert "病毒学2-1.mp4" in lectures.output and "0/7 步" in lectures.output

    second = tmp_path / "病毒学2-2.mp4"
    second.write_bytes(b"fake")
    appended = runner.invoke(app, ["lecture", "add-part", "病毒学", "2", "--media", str(second), *args])
    assert appended.exit_code == 0 and "共 2 节" in appended.output
    assert "病毒学2-2.mp4" in runner.invoke(app, ["lecture", "list", *args]).output

    missing = runner.invoke(app, ["lecture", "add", "不存在的课", "--seq", "1", "--media", str(media), *args])
    assert missing.exit_code == 1 and "没有名为" in missing.output


def test_lecture_with_several_parts_can_be_moved(example_config: Path, tmp_path: Path):
    args = ["--config", str(example_config)]
    files = []
    for name in ("1.1.mp4", "1-2.mp4"):
        files += ["--media", str(tmp_path / name)]
        (tmp_path / name).write_bytes(b"fake")
    for course in ("微生物", "病毒学"):
        assert runner.invoke(app, ["course", "add", course, *args]).exit_code == 0
    added = runner.invoke(app, ["lecture", "add", "微生物", "--seq", "1", *files, *args])
    assert added.exit_code == 0 and "（2 节）" in added.output

    moved = runner.invoke(app, ["lecture", "move", "微生物", "1", "--to", "病毒学", *args])
    assert moved.exit_code == 0, moved.output
    assert "已移到「病毒学」第 1 讲" in moved.output and "从「整合课程记忆」起" in moved.output  # 两门课设置相同
    assert "1-2.mp4" in runner.invoke(app, ["lecture", "list", "病毒学", *args]).output
    again = runner.invoke(app, ["lecture", "move", "病毒学", "1", "--to", "病毒学", *args])
    assert again.exit_code == 1 and "已经在" in again.output


def test_remove_lecture_and_course(example_config: Path, tmp_path: Path):
    args = ["--config", str(example_config)]
    media = tmp_path / "1.mp4"
    media.write_bytes(b"fake")
    runner.invoke(app, ["course", "add", "微生物", *args])
    for seq in ("1", "2"):
        runner.invoke(app, ["lecture", "add", "微生物", "--seq", seq, "--media", str(media), *args])
    declined = runner.invoke(app, ["lecture", "remove", "微生物", "1", *args], input="n\n")
    assert declined.exit_code == 1 and "原始音视频不会删除" in declined.output
    removed = runner.invoke(app, ["lecture", "remove", "微生物", "1", "--yes", *args])
    assert removed.exit_code == 0 and "已删除「微生物」第 1 讲" in removed.output
    with Library(load_config(example_config).data_dir / "lecture.db") as library:
        assert [lecture.seq for lecture in library.lectures(library.course("微生物"))] == [2]
    missing = runner.invoke(app, ["lecture", "remove", "微生物", "1", "--yes", *args])
    assert missing.exit_code == 1 and "Traceback" not in missing.output
    gone = runner.invoke(app, ["course", "remove", "微生物", "--yes", *args])
    assert gone.exit_code == 0 and "已删除课程「微生物」" in gone.output
    assert "微生物" not in runner.invoke(app, ["course", "list", *args]).output and media.is_file()


def test_course_set_instructions(example_config: Path, tmp_path: Path):
    args = ["--config", str(example_config)]
    prompt = tmp_path / "病毒学提示词.txt"
    prompt.write_text("老师常考名词解释", encoding="utf-8-sig")  # 记事本另存为 UTF-8 时带 BOM
    runner.invoke(app, ["course", "add", "病毒学", *args])
    updated = runner.invoke(app, ["course", "set", "病毒学", "--instructions-file", str(prompt), *args])
    assert updated.exit_code == 0 and "课程提示词 8 字" in updated.output
    assert "8 字" in runner.invoke(app, ["course", "list", *args]).output
    cleared = runner.invoke(app, ["course", "set", "病毒学", "--clear-instructions", *args])
    assert "课程提示词 0 字" in cleared.output
    nothing = runner.invoke(app, ["course", "set", "病毒学", *args])
    assert nothing.exit_code == 1 and "没有要修改的设置" in nothing.output


def test_course_languages_and_no_slides(example_config: Path):
    args = ["--config", str(example_config)]
    added = runner.invoke(app, ["course", "add", "Biology", "--teaching", "en,fr", "--interpreted-to", "ja",
                                "--no-slides", *args])
    assert added.exit_code == 0, added.output
    assert "老师用英语、法语授课，现场有人翻译成日语" in added.output
    listed = runner.invoke(app, ["course", "list", *args]).output
    assert "不提供" in listed

    changed = runner.invoke(app, ["course", "set", "Biology", "--interpreted-to", "none", "--teaching", "ko", *args])
    assert changed.exit_code == 0 and "韩语授课" in changed.output and "翻译" not in changed.output

    bad = runner.invoke(app, ["course", "add", "Chemistry", "--teaching", "韩语", *args])
    assert bad.exit_code == 1 and "语言代码格式不对" in bad.output and "Traceback" not in bad.output
    bad_set = runner.invoke(app, ["course", "set", "Biology", "--interpreted-to", "zh,ko", *args])
    assert bad_set.exit_code == 1 and "现场翻译只能选一种语言" in bad_set.output
    # 转写方案不支持课程的语言时拦下（M4 修订：授课语言不能只支持中文）
    german = runner.invoke(app, ["course", "add", "German", "--teaching", "de", "--asr", "funasr:paraformer", *args])
    assert german.exit_code == 1 and "本地 Paraformer 不支持德语" in german.output
    switched = runner.invoke(app, ["course", "set", "Biology", "--asr", "funasr:sensevoice", *args])
    assert switched.exit_code == 0  # SenseVoice 支持韩语
    assert runner.invoke(app, ["course", "set", "Biology", "--teaching", "fr", *args]).exit_code == 1


def test_run_all_and_rebuild_when_nothing_to_do(example_config: Path):
    args = ["--config", str(example_config)]
    runner.invoke(app, ["course", "add", "药理学", *args])
    blocked = runner.invoke(app, ["run", "药理学", *args])
    assert blocked.exit_code == 1 and "是否提供课件" in blocked.output  # 还没有讲也先过课件关卡
    runner.invoke(app, ["course", "set", "药理学", "--no-slides", *args])
    assert "所有讲都已处理完成" in runner.invoke(app, ["run", "药理学", *args]).output
    assert "课程记忆已是最新" in runner.invoke(app, ["rebuild", "药理学", *args]).output


def test_run_rejects_unknown_force_step(example_config: Path):
    result = runner.invoke(app, ["run", "病毒学", "2", "--force", "everything", "--config", str(example_config)])
    assert result.exit_code == 1 and "未知的步骤" in result.output


def test_remote_listing_for_unknown_profile_fails(example_config: Path):
    result = runner.invoke(app, ["models", "--config", str(example_config), "--remote", "不存在"])
    assert result.exit_code == 1
    assert "没有名为" in result.output


def test_material_commands_and_lecture_pages(example_config: Path, tmp_path: Path):
    from .test_materials import _pdf

    args = ["--config", str(example_config)]
    runner.invoke(app, ["course", "add", "病毒学", *args])
    media = tmp_path / "1.mp4"
    media.write_bytes(b"fake")
    runner.invoke(app, ["lecture", "add", "病毒学", "--seq", "1", "--media", str(media), *args])
    blocked = runner.invoke(app, ["run", "病毒学", *args])
    assert blocked.exit_code == 1 and "是否提供课件" in blocked.output  # 课件关卡

    added = runner.invoke(app, ["material", "add", "病毒学", str(_pdf(tmp_path / "ch1-介绍.pdf", 3)), "--kind", "slides",
                                "--no-process", *args])
    assert added.exit_code == 0, added.output
    listed = runner.invoke(app, ["material", "list", "病毒学", *args]).output
    assert "ch1-介绍" in listed and "0/3 页" in listed and "已提供" in listed

    pages = runner.invoke(app, ["lecture", "pages", "病毒学", "1", "--pages", "1:2-3", *args])
    assert pages.exit_code == 0 and "手动指定" in pages.output and "第 2–3 页" in pages.output
    bad = runner.invoke(app, ["lecture", "pages", "病毒学", "1", "--pages", "1:5-9", *args])
    assert bad.exit_code == 1 and "超出范围" in bad.output

    removed = runner.invoke(app, ["material", "remove", "病毒学", "1", "--yes", *args])
    assert removed.exit_code == 0 and "重新匹配课件" in removed.output
    assert "未选择" in runner.invoke(app, ["material", "list", "病毒学", *args]).output
    missing = runner.invoke(app, ["material", "add", "病毒学", str(tmp_path / "1.mp4"), "--kind", "slides", *args])
    assert missing.exit_code == 1 and "没有找到可以导入的文档" in missing.output
