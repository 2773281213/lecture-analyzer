import pytest

from lecture_analyzer.asr.base import Segment
from lecture_analyzer.course import CourseSettings, SettingsError, build_settings, parse_languages, recap_pattern
from lecture_analyzer.pipeline.integrate import _cue_section


def test_parse_languages_normalizes_and_rejects_bad_codes():
    assert parse_languages("ko, zh") == ["ko", "zh"]
    assert parse_languages(" KO，zh、ko ") == ["ko", "zh"]
    assert parse_languages("zh-TW en") == ["zh-tw", "en"]
    assert parse_languages("") == []
    with pytest.raises(ValueError, match="语言代码格式不对"):
        parse_languages("韩语")


def test_legacy_interpretation_value_is_converted():
    settings = CourseSettings(subject="病毒学", teaching="ko+zh", note_language="ko")
    assert (settings.teaching, settings.interpreted_to) == ("ko", "zh")
    assert settings.interpreted and settings.spoken_languages == ["ko", "zh"] and settings.asr_language is None
    assert settings.teaching_description == "老师用韩语授课，现场有人翻译成中文"
    assert settings.note_language_name == "中文" and settings.html_lang == "zh-CN"  # 笔记一律中文，旧的取值也按中文


@pytest.mark.parametrize(
    ("fields", "spoken", "asr_language", "description"),
    [
        ({"teaching": "en"}, ["en"], "en", "英语授课"),
        ({"teaching": "ko,en"}, ["ko", "en"], None, "韩语、英语授课"),
        ({"teaching": "ja", "interpreted_to": "zh"}, ["ja", "zh"], None, "老师用日语授课，现场有人翻译成中文"),
        ({"teaching": "zh", "interpreted_to": "zh"}, ["zh"], "zh", "老师用中文授课，现场有人翻译成中文"),
        ({"teaching": "sw"}, ["sw"], "sw", "sw授课"),  # 名称表里没有的语言直接显示代码
    ],
)
def test_any_language_combination(fields, spoken, asr_language, description):
    settings = build_settings(**fields)
    assert settings.spoken_languages == spoken and settings.asr_language == asr_language
    assert settings.teaching_description == description
    assert settings.html_lang == "zh-CN"


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"teaching": ""}, "至少要有一种授课语言"),
        ({"teaching": "中文"}, "语言代码格式不对"),
        ({"interpreted_to": "zh,ko"}, "现场翻译只能选一种语言"),
    ],
)
def test_build_settings_gives_readable_errors(fields, message):
    with pytest.raises(SettingsError, match=message) as error:
        build_settings(**fields)
    assert "Value error" not in str(error.value) and "pydantic" not in str(error.value)


def test_recap_pattern_follows_course_languages():
    both = recap_pattern(["ko", "zh"])
    assert both.search("지난 시간에 배웠던 내용") and both.search("上次讲过的受体")
    assert recap_pattern(["en"]).search("As I mentioned LAST TIME")
    assert recap_pattern(["zh-tw"]).search("上次")  # 地区代码按语言取词表
    assert recap_pattern(["sw"]) is None


def test_cue_section_without_cue_list_is_empty():
    segments = [Segment(0, 1000, "As we discussed last week"), Segment(1000, 2000, "New topic")]
    assert [item.full for item in _cue_section(segments, recap_pattern(["en"])).items] == [
        "[S1][00:00] As we discussed last week"
    ]
    assert _cue_section(segments, None).items == []
