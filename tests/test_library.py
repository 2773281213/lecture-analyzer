import json
import sqlite3
from pathlib import Path

import pytest

from lecture_analyzer.asr.base import Segment
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.library import CheckedSignal, Library, LibraryError
from lecture_analyzer.library_schema import MIGRATIONS, SCHEMA_VERSION
from lecture_analyzer.llm import ModelProfile, Usage
from lecture_analyzer.schemas import LectureNotes, PageDigest, Signal


@pytest.fixture
def library(tmp_path: Path):
    with Library(tmp_path / "db" / "lecture.db") as lib:
        yield lib


def test_courses_and_lectures(library: Library, tmp_path: Path):
    settings = CourseSettings(subject="病毒学", teaching="ko+zh", note_language="ko")
    course = library.add_course("病毒学", settings, asr="llm:gemini")
    assert library.course("病毒学") == course and course.settings == settings and course.asr == "llm:gemini"
    with pytest.raises(LibraryError, match="已存在"):
        library.add_course("病毒学", settings)
    with pytest.raises(LibraryError, match="没有名为"):
        library.course("不存在")

    media = [tmp_path / "病毒学2-1.mp4", tmp_path / "病毒学2-2.mp4"]
    lecture = library.add_lecture(course, 2, media, date="2026-10-14", kind="review")
    assert library.lecture(course, 2) == lecture and lecture.kind == "review" and lecture.status == {}
    assert [(p.part, Path(p.media_path).name) for p in lecture.parts] == [(1, "病毒学2-1.mp4"), (2, "病毒学2-2.mp4")]
    with pytest.raises(LibraryError, match="已有第 2 讲"):
        library.add_lecture(course, 2, [tmp_path / "x.mp4"])
    with pytest.raises(LibraryError, match="讲的类型"):
        library.add_lecture(course, 3, [tmp_path / "x.mp4"], kind="lab")
    with pytest.raises(LibraryError, match="至少要有一个"):
        library.add_lecture(course, 3, [])
    assert [c.name for c in library.courses()] == ["病毒学"]
    assert library.lectures(course) == library.lectures() == [lecture]


def test_parts_durations_and_offsets(library: Library, tmp_path: Path):
    course = library.add_course("病毒学", CourseSettings())
    lecture = library.add_lecture(course, 1, [tmp_path / "1.1.mp4"])
    library.set_step(lecture.id, "media", "done")
    library.set_step(lecture.id, "extract", "done")
    lecture = library.add_part(lecture.id, tmp_path / "1-2.mp4")
    assert [p.part for p in lecture.parts] == [1, 2]
    assert lecture.step_state("media") == "" and lecture.step_state("extract") == "done"  # 只清抽音频，后续由依赖关系重跑
    library.set_part_duration(lecture.id, 1, 2_700_000)
    library.set_part_duration(lecture.id, 2, 1_800_000)
    assert library.lecture_by_id(lecture.id).part_offsets() == [0, 2_700_000]


def test_status_and_updates(library: Library, tmp_path: Path):
    lecture = library.add_lecture(library.add_course("药理学", CourseSettings()), 1, [tmp_path / "a.m4a"])
    library.set_step(lecture.id, "media", "done", "时长 01:28:22")
    library.update_lecture(lecture.id, title="量效关系", duration_ms=5_302_000, kind="review")
    reloaded = library.lecture_by_id(lecture.id)
    assert reloaded.step_state("media") == "done" and reloaded.step_state("extract") == ""
    assert reloaded.title == "量效关系" and reloaded.duration_ms == 5_302_000 and reloaded.kind == "review"
    assert reloaded.last_activity >= reloaded.created_at
    with pytest.raises(LibraryError, match="不能直接修改"):
        library.update_lecture(lecture.id, seq=9)
    library.reset_steps(lecture.id, ["media"])
    assert library.lecture_by_id(lecture.id).status == {}


def test_course_settings_order_and_instructions(library: Library):
    a = library.add_course("病毒学", CourseSettings())
    b = library.add_course("药理学", CourseSettings())
    c = library.add_course("微生物", CourseSettings())
    library.move_course(c.id, -1)
    assert [x.name for x in library.courses()] == ["病毒学", "微生物", "药理学"]
    library.move_course(a.id, -1)  # 已在最前，不动
    assert [x.name for x in library.courses()][0] == "病毒学"

    updated = library.update_course(b.id, instructions="  老师常考名词解释  ", subject="药理")
    assert updated.instructions == "老师常考名词解释" and updated.settings.subject == "药理"
    with pytest.raises(LibraryError, match="最多 4000 字"):
        library.update_course(b.id, instructions="长" * 4001)
    with pytest.raises(LibraryError, match="不能直接修改"):
        library.update_course(b.id, name="x")


def test_course_languages_and_slides_choice(library: Library):
    course = library.add_course("Biology", CourseSettings())
    assert course.slides == "unset"
    updated = library.update_course(course.id, teaching="EN, fr", interpreted_to="zh", note_language="en", slides="none")
    assert (updated.settings.teaching, updated.settings.interpreted_to, updated.settings.note_language) == ("en,fr", "zh", "zh")
    assert updated.slides == "none"
    legacy = library.update_course(course.id, teaching="ko+ja")  # 旧写法里的翻译语言为准
    assert (legacy.settings.teaching, legacy.settings.interpreted_to) == ("ko", "ja")
    cleared = library.update_course(course.id, interpreted_to="")
    assert not cleared.settings.interpreted and cleared.settings.teaching == "ko"
    with pytest.raises(LibraryError, match="课程设置不正确：语言代码格式不对"):
        library.update_course(course.id, teaching="韩语")
    with pytest.raises(LibraryError, match="课件选择"):
        library.update_course(course.id, slides="maybe")
    assert library.course_by_id(course.id).settings.teaching == "ko"  # 出错时不改动


def test_move_lecture_between_courses(library: Library, tmp_path: Path):
    wrong = library.add_course("微生物", CourseSettings())
    right = library.add_course("病毒学", CourseSettings())
    lecture = library.add_lecture(wrong, 3, [tmp_path / "a.mp4"])
    library.add_lecture(right, 3, [tmp_path / "b.mp4"])
    with pytest.raises(LibraryError, match="已有第 3 讲"):
        library.move_lecture(lecture.id, right, 3)
    library.move_lecture(lecture.id, right, 4)
    assert library.lecture(right, 4).id == lecture.id and library.lectures(wrong) == []


def test_tags_and_review_progress(library: Library, tmp_path: Path):
    course = library.add_course("病毒学", CourseSettings())
    lecture = library.add_lecture(course, 1, [tmp_path / "a.mp4"])
    library.add_tag(lecture.id, " 期末重点 ")
    library.add_tag(lecture.id, "期末重点")  # 重复添加无影响
    library.add_tag(lecture.id, "已复习")
    assert library.tags(lecture.id) == ["已复习", "期末重点"]
    assert library.course_tags(course) == {lecture.id: ["已复习", "期末重点"]}
    for bad in ("", "  ", "x" * 21, "a\nb"):
        with pytest.raises(LibraryError, match="标签"):
            library.add_tag(lecture.id, bad)
    library.remove_tag(lecture.id, "已复习")
    assert library.tags(lecture.id) == ["期末重点"]

    library.set_reviewed(course.id, "受体", True)
    library.set_reviewed(course.id, "受体", True)
    assert library.reviewed_names(course.id) == {"受体"}
    library.set_reviewed(course.id, "受体", False)
    assert library.reviewed_names(course.id) == set()


def test_content_roundtrip(library: Library, tmp_path: Path):
    lecture = library.add_lecture(library.add_course("药理学", CourseSettings()), 1, [tmp_path / "a.m4a"])
    segments = [Segment(0, 900, "第一句", "zh"), Segment(900, 1800, "둘째", "ko")]
    library.replace_segments(lecture.id, segments)
    assert library.segments(lecture.id) == segments

    pages = [PageDigest(page_no=1, title="受体", content_md="$K_d$", terms=["受体", "亲和力"])]
    library.replace_pages("lecture", lecture.id, pages)
    assert library.pages("lecture", lecture.id) == pages

    notes = LectureNotes(title="t", outline=[], knowledge_points=[], examples=[], homework=[], signals=[])
    library.save_notes(lecture.id, notes, "claude", "claude-opus-5")
    assert library.notes(lecture.id) == notes and library.notes_model(lecture.id) == ("claude", "claude-opus-5")

    signal = Signal(type="emphasis", knowledge_point="kp", strength=2, quote="第一句", segment_ids=[2], slide_pages=[1])
    library.replace_signals(lecture.id, [CheckedSignal(signal, True, "global", [1])])
    stored = library.signals(lecture.id)[0]
    assert stored.valid and stored.method == "global" and stored.segment_ids == [1]
    assert len(library.signal_ids(lecture.id)) == 1


def test_cost_summary(library: Library, tmp_path: Path):
    course = library.add_course("药理学", CourseSettings())
    lecture = library.add_lecture(course, 1, [tmp_path / "a.m4a"])
    claude = ModelProfile(name="claude", adapter="anthropic", model="claude-opus-5", price_in=5.0, price_out=25.0)
    gemini = ModelProfile(name="gemini", adapter="openai_compat", model="gemini-3.8-flash-high")
    library.log_call(lecture.id, "extract", claude, Usage(1000, 200))
    library.log_call(lecture.id, "extract", claude, Usage(500, 100))
    library.log_call(None, "transcribe", gemini, Usage(300, 30))  # 没填单价：费用记为空
    summary = library.cost_summary(course)
    assert summary == [{"step": "extract", "model": "claude-opus-5", "calls": 2, "input_tokens": 1500,
                        "output_tokens": 300, "cost_usd": pytest.approx((1500 * 5 + 300 * 25) / 1e6)}]
    assert {row["model"]: row["cost_usd"] for row in library.cost_summary()}["gemini-3.8-flash-high"] is None


def test_migrates_v1_database_with_backup(tmp_path: Path):
    """M1 建的 v1 书库：打开时升级到最新版本，数据保留，旧的音视频路径变成第 1 节，并留有备份。"""
    path = tmp_path / "lecture.db"
    conn = sqlite3.connect(path)
    conn.executescript(MIGRATIONS[1])
    conn.execute(
        "INSERT INTO course (name, subject, teaching, note_language, asr, created_at) VALUES ('药理学', '药理学', 'zh',"
        " 'zh', NULL, '2026-10-14T10:00:00')"
    )
    conn.execute(
        "INSERT INTO course (name, subject, teaching, note_language, asr, created_at) VALUES ('病毒学', '病毒学', 'ko+zh',"
        " 'ko', NULL, '2026-10-14T10:00:00')"
    )
    conn.execute(
        "INSERT INTO lecture (course_id, seq, title, media_path, duration_ms, status, created_at) VALUES (1, 3, '量效关系',"
        " 'F:/录课/药理学3.m4a', 5302000, ?, '2026-10-14T10:00:00')",
        (json.dumps({"book": {"state": "done", "at": "2026-10-14T22:29:33", "detail": ""}}),),
    )
    conn.commit()
    conn.close()

    with Library(path) as library:
        assert library.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        course = library.course("药理学")
        assert course.instructions == "" and course.position == 1
        lecture = library.lecture(course, 3)
        assert lecture.title == "量效关系" and lecture.step_state("book") == "done"
        assert [(p.part, p.media_path, p.duration_ms) for p in lecture.parts] == [(1, "F:/录课/药理学3.m4a", 5302000)]
        columns = {row["name"] for row in library.conn.execute("PRAGMA table_info(lecture)")}
        assert "media_path" not in columns
        # v3：旧的 "ko+zh" 拆成授课语言和翻译语言；课件选择默认为"未选择"；资料、报告、任务表已建好；v4：问答表
        virology = library.course("病毒学")
        assert (virology.settings.teaching, virology.settings.interpreted_to) == ("ko", "zh")
        assert (course.settings.teaching, course.settings.interpreted_to) == ("zh", "")
        assert course.slides == virology.slides == "unset"
        tables = {row["name"] for row in library.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"material", "material_item", "item_kp", "lecture_page", "report", "job", "qa"} <= tables

    backup = sqlite3.connect(tmp_path / "lecture.db.bak-v1")
    assert backup.execute("PRAGMA user_version").fetchone()[0] == 1
    assert backup.execute("SELECT media_path FROM lecture").fetchone()[0] == "F:/录课/药理学3.m4a"
    backup.close()
