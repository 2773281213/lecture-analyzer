from lecture_analyzer.asr.base import Segment
from lecture_analyzer.pipeline.validate import TranscriptIndex, check_signal, pass_rate, validate_signals
from lecture_analyzer.schemas import Signal

SEGMENTS = [
    Segment(0, 1000, "上节课讲了受体。"),
    Segment(1000, 2000, "这个期末一定会考，"),
    Segment(2000, 3000, "大家记住阿司匹林的三段剂量。"),
    Segment(3000, 4000, "第二个问题是蓄积。"),
    Segment(4000, 5000, "바이러스 명칭을 정할 때"),
]


def _signal(quote: str, ids: list[int], type_: str = "exam_hint", pages: list[int] | None = None) -> Signal:
    return Signal(type=type_, knowledge_point="kp", strength=2, quote=quote, segment_ids=ids, slide_pages=pages or [])


def test_window_match_spans_adjacent_segments():
    checked = check_signal(_signal("这个期末一定会考。大家记住阿司匹林", [2, 3]), TranscriptIndex(SEGMENTS))
    assert checked.valid and checked.method == "window" and checked.segment_ids == [2, 3]


def test_window_allows_one_neighbor_each_side():
    checked = check_signal(_signal("上节课讲了受体", [2]), TranscriptIndex(SEGMENTS))
    assert checked.valid and checked.method == "window"


def test_fuzzy_tolerates_small_differences():
    checked = check_signal(_signal("大家记住阿斯匹林的三段剂量", [3]), TranscriptIndex(SEGMENTS))
    assert checked.valid and checked.method == "fuzzy"


def test_global_match_corrects_segment_ids():
    checked = check_signal(_signal("第二个问题是蓄积", [1]), TranscriptIndex(SEGMENTS))
    assert checked.valid and checked.method == "global" and checked.segment_ids == [4]


def test_korean_quote_and_invalid_ids():
    checked = check_signal(_signal("바이러스 명칭을", [99]), TranscriptIndex(SEGMENTS))
    assert checked.valid and checked.method == "global" and checked.segment_ids == [5]


def test_fabricated_quote_fails():
    checked = check_signal(_signal("这一章不考", [2]), TranscriptIndex(SEGMENTS))
    assert not checked.valid and checked.method == ""


def test_empty_quote_only_allowed_for_skipped_slide():
    index = TranscriptIndex(SEGMENTS)
    assert check_signal(_signal("", [], "skipped_slide", [12]), index).method == "slide"
    assert not check_signal(_signal("", [], "skipped_slide"), index).valid
    assert not check_signal(_signal("  ", [2], "emphasis"), index).valid


def test_locate_maps_offsets_to_segments():
    index = TranscriptIndex(SEGMENTS)
    assert index.locate("会考大家记住") == [2, 3]
    assert index.locate("不存在的话") is None


def test_pass_rate():
    checked = validate_signals([_signal("这个期末一定会考", [2]), _signal("编造的话", [2])], SEGMENTS)
    assert pass_rate(checked) == 0.5
    assert pass_rate([]) == 1.0
