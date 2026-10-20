import json
from pathlib import Path

import pytest

from lecture_analyzer.asr.base import Segment
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.knowledge import CourseMemory, KnowledgePoint, KPAppearance
from lecture_analyzer.library import Library
from lecture_analyzer.pipeline.integrate import (
    IntegrateContext, candidate_keys, integrate_through, plan_rebuild, rebuild, resolve,
)
from lecture_analyzer.pipeline.validate import validate_signals
from lecture_analyzer.schemas import IntegrationResult, LectureNotes

from .conftest import RoutingProvider, make_profile


def _notes(points, signals=()):
    return LectureNotes(
        title="t", outline=["o"], examples=[], homework=[],
        knowledge_points=[{"name": n, "summary_md": f"{n}的要点", "slide_pages": [], "segment_ids": [i], "minutes": 5}
                          for i, n in enumerate(points, start=1)],
        signals=[{"type": t, "knowledge_point": kp, "strength": 2, "quote": q, "segment_ids": [1], "slide_pages": []}
                 for t, kp, q in signals],
    )


LECTURES = {
    1: (_notes(["病毒命名", "ICTV 分类"], [("emphasis", "ICTV 分类", "复制方式很重要")]),
        [Segment(0, 1000, "复制方式很重要"), Segment(1000, 2000, "病毒按宿主命名")]),
    2: (_notes(["ICTV 分类标准", "卫星病毒"], [("recap", "ICTV 分类标准", "昨天学过分类")]),
        [Segment(0, 1000, "昨天学过分类"), Segment(1000, 2000, "卫星病毒需要辅助病毒")]),
    3: (_notes(["卫星病毒 HDV", "衣壳对称"], [("emphasis", "衣壳对称", "衣壳一定要记住")]),
        [Segment(0, 1000, "衣壳一定要记住"), Segment(1000, 2000, "上次讲的卫星病毒")]),
}


def _result(points, refs=(), signals=None):
    return json.dumps({
        "knowledge_points": [{"note_kp": i, "target": t, "name": n, "chapter": c, "role": r, "aliases": []}
                             for i, (t, n, c, r) in enumerate(points, start=1)],
        "signals": signals if signals is not None else [{"signal": 1, "note_kp": 1}],
        "references": [{"note_kp": n, "target": t, "lecture_seq": s, "relation": rel, "explanation": "联系",
                        "segment_ids": [1]} for n, t, s, rel in refs],
    }, ensure_ascii=False)


RESULTS = {
    1: _result([("new", "病毒命名", "命名", "first"), ("new", "ICTV 分类", "分类", "first")], signals=[{"signal": 1, "note_kp": 2}]),
    2: _result([("K2", "ICTV 分类", "分类", "deepen"), ("new", "卫星病毒", "分类", "first")], refs=[(1, "K2", 1, "recap")]),
    3: _result([("K3", "卫星病毒", "分类", "review"), ("new", "衣壳对称", "形态", "first")], refs=[(1, "K3", 2, "recap")],
               signals=[{"signal": 1, "note_kp": 2}]),
}


@pytest.fixture
def course_env(tmp_path: Path):
    with Library(tmp_path / "lecture.db") as library:
        course = library.add_course("病毒学", CourseSettings(subject="病毒学"))
        lectures = {}
        for seq, (notes, segments) in LECTURES.items():
            lecture = library.add_lecture(course, seq, [tmp_path / f"{seq}.mp4"])
            library.replace_segments(lecture.id, segments)
            library.save_notes(lecture.id, notes, "fake", "fake-model")
            library.replace_signals(lecture.id, validate_signals(notes.signals, segments))
            library.set_step(lecture.id, "validate", "done")
            lectures[seq] = lecture.id
        yield library, course, lectures


def _context(library, responses):
    provider = RoutingProvider({"IntegrationResult": list(responses)}, make_profile())
    return IntegrateContext(library, CourseMemory(library), provider), provider


def _tree(library, course):
    return [(kp.key, kp.name, kp.chapter, [(a.seq, a.role) for a in kp.appearances])
            for kp in CourseMemory(library).tree(course)]


def test_integrate_through_builds_memory_in_order(course_env):
    library, course, ids = course_env
    ctx, provider = _context(library, [RESULTS[1], RESULTS[2], RESULTS[3]])
    outcome = integrate_through(ctx, course, library.lecture_by_id(ids[3]))
    assert [lecture.seq for lecture in outcome.integrated] == [1, 2, 3] and len(provider.calls) == 3
    assert _tree(library, course) == [
        (1, "病毒命名", "命名", [(1, "first")]),
        (2, "ICTV 分类", "分类", [(1, "first"), (2, "deepen")]),
        (3, "卫星病毒", "分类", [(2, "first"), (3, "review")]),
        (4, "衣壳对称", "形态", [(3, "first")]),
    ]
    memory = CourseMemory(library)
    ref = memory.references(ids[3])[0]
    assert (ref.key, ref.target_seq, ref.relation, ref.target_note_index) == (3, 2, "recap", 2)
    assert memory.signal_keys(ids[1]) == [2]  # 第 1 讲的信号归入 ICTV 分类
    alias_kp = next(kp for kp in memory.tree(course) if kp.key == 2)
    assert alias_kp.aliases == ("ICTV 分类标准",)  # 第 2 讲的叫法补成别名
    # 第 2 讲的输入里带着第 1 讲的知识点、历史原话和回顾线索
    second_input = provider.calls[1]["parts"][0].text
    assert "[K2] ICTV 分类｜章节：分类" in second_input and "复制方式很重要" in second_input
    assert "[S1][00:00] 昨天学过分类" in second_input
    assert all(library.lecture_by_id(i).step_state("integrate") == "done" for i in ids.values())


def test_rebuild_from_keeps_earlier_keys_and_replays_without_calls(course_env):
    library, course, ids = course_env
    ctx, _ = _context(library, [RESULTS[1], RESULTS[2], RESULTS[3]])
    integrate_through(ctx, course, library.lecture_by_id(ids[3]))
    before = _tree(library, course)

    ctx, provider = _context(library, [RESULTS[2], RESULTS[3]])
    assert [lecture.seq for lecture in plan_rebuild(library, ctx.memory, course, 2)] == [2, 3]
    outcome = rebuild(ctx, course, from_seq=2)
    assert outcome.replayed == 1 and [lecture.seq for lecture in outcome.integrated] == [2, 3]
    assert len(provider.calls) == 2 and _tree(library, course) == before
    assert rebuild(ctx, course) is None  # 已是最新


def test_reintegrating_middle_lecture_marks_later_stale(course_env):
    library, course, ids = course_env
    ctx, _ = _context(library, [RESULTS[1], RESULTS[2], RESULTS[3]])
    integrate_through(ctx, course, library.lecture_by_id(ids[3]))
    ctx, provider = _context(library, [RESULTS[2]])
    outcome = integrate_through(ctx, course, library.lecture_by_id(ids[2]))
    assert len(provider.calls) == 1 and [lecture.seq for lecture in outcome.stale] == [3]
    assert library.lecture_by_id(ids[3]).step_state("integrate") == "stale"
    assert [key for key, *_ in _tree(library, course)] == [1, 2, 3]  # 第 3 讲的内容暂时不在课程记忆里
    assert [lecture.seq for lecture in plan_rebuild(library, ctx.memory, course, None)] == [3]


def _tree_points():
    return [
        KnowledgePoint(1, "病毒命名", "命名", (), (KPAppearance(1, 1, 1, "first", 5),)),
        KnowledgePoint(2, "ICTV 分类", "分类", ("国际分类",), (KPAppearance(1, 1, 2, "first", 5),)),
    ]


def test_resolve_fixes_invalid_output():
    notes = _notes(["国际分类", "全新概念", "全新概念", "漏写的"],
                   [("emphasis", "全新概念", "q"), ("recap", "不相关", "q"), ("emphasis", "漏写的", "q")])
    output = IntegrationResult.model_validate_json(json.dumps({
        "knowledge_points": [
            {"note_kp": 1, "target": "K9", "name": "国际分类", "chapter": "分类", "role": "first", "aliases": []},
            {"note_kp": 2, "target": "new", "name": "全新概念", "chapter": "新章", "role": "first", "aliases": []},
            {"note_kp": 2, "target": "K1", "name": "重复的", "chapter": "", "role": "deepen", "aliases": []},
            {"note_kp": 3, "target": "new", "name": "全新概念", "chapter": "新章", "role": "first", "aliases": []},
        ],
        "signals": [{"signal": 1, "note_kp": 2}, {"signal": 2, "note_kp": 0}],
        "references": [
            {"note_kp": 1, "target": "K2", "lecture_seq": 7, "relation": "recap", "explanation": " 回顾 ",
             "segment_ids": [99]},
            {"note_kp": 1, "target": "K2", "lecture_seq": 1, "relation": "recap", "explanation": "重复", "segment_ids": []},
            {"note_kp": 9, "target": "K1", "lecture_seq": 1, "relation": "repeat", "explanation": "", "segment_ids": []},
            {"note_kp": 2, "target": "K3", "lecture_seq": 1, "relation": "builds_on", "explanation": "", "segment_ids": []},
        ],
    }, ensure_ascii=False))
    resolved, stats = resolve(output, notes, seq=2, segment_count=4, tree=_tree_points())
    kps = [(kp.note_index, kp.key, kp.is_new, kp.role) for kp in resolved.kps]
    # K9 不存在，但名称与 K2 的别名相同 → 并入 K2；两个「全新概念」合并成一个新编号；漏写的按新知识点处理
    assert kps == [(1, 2, False, "deepen"), (2, 3, True, "first"), (3, 3, False, "first"), (4, 4, True, "first")]
    assert resolved.signal_keys == [3, None, 4]  # 第 2 条没有归类；第 3 条按名称匹配到「漏写的」
    assert (stats.new, stats.linked, stats.unmatched_signals) == (2, 1, 1)
    assert [(r.key, r.target_seq, r.explanation, r.segment_ids) for r in resolved.references] == [(2, 1, "回顾", [1])]


def test_candidate_keys_by_name_and_previous_lecture():
    notes = _notes(["ICTV 分类标准"], [("emphasis", "衣壳结构", "q")])
    tree = _tree_points() + [KnowledgePoint(3, "无关", "", (), (KPAppearance(5, 2, 1, "first", 1),))]
    assert candidate_keys(notes, tree, previous_lecture_id=None) == {2}
    assert candidate_keys(notes, tree, previous_lecture_id=5) == {2, 3}
