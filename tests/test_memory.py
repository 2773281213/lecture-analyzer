import pytest

from lecture_analyzer.pipeline.memory import (
    BudgetError, Item, Section, Window, assemble, input_budget, merge_notes, plan_windows, trimmed_note,
)
from lecture_analyzer.schemas import LectureNotes

from .conftest import make_profile


def _profile(context=10_000, output=2_000):
    # SAFETY_TOKENS = 4000：输入预算 = 上下文 − 输出 − 4000
    return make_profile(max_context=context, max_output=output, tokens_per_char=1.0)


def test_input_budget_uses_context_and_cap():
    assert input_budget(_profile(128_000, 8_000)) == 116_000
    assert input_budget(_profile(128_000, 8_000), cap=60_000) == 60_000


def test_whole_lecture_in_one_window_when_it_fits():
    assert plan_windows([100] * 10, [0, 5], 500, _profile()) == [Window(0, 10)]


def test_split_by_parts_then_merge_neighbours():
    # 输入预算 4000、输出 2000（可容纳约 1500/0.35 ≈ 1428 字的转写）
    windows = plan_windows([100] * 40, [0, 10, 20, 30], 500, _profile(10_000, 2_000))
    assert windows == [Window(0, 10), Window(10, 20), Window(20, 30), Window(30, 40)]
    assert plan_windows([50] * 40, [0, 10, 20, 30], 500, _profile(10_000, 2_000)) == [Window(0, 20), Window(20, 40)]


def test_single_part_too_long_is_halved():
    windows = plan_windows([100] * 30, [0], 500, _profile(10_000, 2_000))
    assert len(windows) > 1 and windows[0].start == 0 and windows[-1].end == 30
    assert all(w.end - w.start <= 14 for w in windows)


def test_fixed_content_too_large_raises():
    with pytest.raises(BudgetError):
        plan_windows([100] * 5, [0], 9_000, _profile(10_000, 2_000))


def test_merge_notes_concatenates_and_merges_same_points():
    def notes(title, kps, outline):
        return LectureNotes(
            title=title, outline=outline, examples=[], homework=["作业"], signals=[],
            knowledge_points=[{"name": n, "summary_md": s, "slide_pages": [p], "segment_ids": [i], "minutes": 5}
                              for n, s, p, i in kps],
        )

    merged = merge_notes([
        notes("病毒分类", [("ICTV 分类", "一", 1, 3), ("命名", "二", 2, 5)], ["a"]),
        notes("衣壳", [("ICTV分类", "三", 4, 40)], ["b"]),
    ])
    assert merged.title == "病毒分类；衣壳" and merged.outline == ["a", "b"] and merged.homework == ["作业"]
    first = merged.knowledge_points[0]
    assert (first.summary_md, first.segment_ids, first.slide_pages, first.minutes) == ("一\n\n三", [3, 40], [1, 4], 10)
    assert len(merged.knowledge_points) == 2


def test_assemble_trims_in_rank_and_keep_order():
    sections = [
        Section("必带", [Item("本讲" * 100, "本讲" * 100)]),
        Section("最近几讲", [Item("旧" * 300, keep=1), Item("新" * 300, keep=2)], trim_rank=1),
        Section("知识点", [Item("A" * 50 + "要点" * 100, "A" * 50, keep=1)], trim_rank=3),
    ]
    whole = assemble(sections, budget=10_000, profile=_profile())
    assert whole.trimmed == {} and "旧" in whole.text
    trimmed = assemble(sections, budget=950, profile=_profile())
    assert trimmed.trimmed == {"最近几讲": 1} and "旧" not in trimmed.text and "新" in trimmed.text
    harder = assemble(sections, budget=400, profile=_profile())
    assert harder.trimmed == {"最近几讲": 2, "知识点": 1} and "要点" not in harder.text
    assert "最近几讲 2 条、知识点 1 条" in trimmed_note(harder.trimmed)
    with pytest.raises(BudgetError, match="超出模型档案"):
        assemble(sections, budget=100, profile=_profile())
