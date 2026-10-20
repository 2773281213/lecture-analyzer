import pytest

from lecture_analyzer.textnorm import cer, normalize_zh
from lecture_analyzer.timecode import format_ms, parse_timecode


def test_normalize_zh_removes_punctuation_and_width():
    assert normalize_zh("Ｈｅｌｌｏ，世界！ OK\n") == "hello世界ok"


@pytest.mark.parametrize(
    ("ref", "hyp", "expected"),
    [
        ("牛顿第二定律", "牛顿第二定律", 0.0),
        ("牛顿定律", "牛顿定理", 0.25),
        ("加速度，与力成正比。", "加速度与力成正比", 0.0),
        ("", "", 0.0),
        ("", "多余", 1.0),
    ],
)
def test_cer(ref, hyp, expected):
    assert cer(ref, hyp) == pytest.approx(expected)


@pytest.mark.parametrize(("ms", "text"), [(0, "00:00"), (61_000, "01:01"), (3_661_000, "1:01:01"), (-5, "00:00")])
def test_format_ms(ms, text):
    assert format_ms(ms) == text


@pytest.mark.parametrize(("text", "ms"), [("01:15", 75_000), ("1:01:01", 3_661_000), ("00:01.5", 1_500)])
def test_parse_timecode(text, ms):
    assert parse_timecode(text) == ms


@pytest.mark.parametrize("text", ["abc", "15", "1:2:3:4", "-1:00"])
def test_parse_timecode_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_timecode(text)
