import json
from types import SimpleNamespace

import pytest

from lecture_analyzer.config import ScoringConfig
from lecture_analyzer.knowledge import HistorySignal, KnowledgePoint, KPAppearance
from lecture_analyzer.library import Library, LibraryError
from lecture_analyzer.llm import LLMError
from lecture_analyzer.materials import MaterialStore
from lecture_analyzer.pipeline import materials as materials_module
from lecture_analyzer.knowledge import CourseMemory
from lecture_analyzer.pipeline import report as report_module
from lecture_analyzer.pipeline import review as review_module
from lecture_analyzer.pipeline.report import (
    borda, build_plan, generate_report, normalize_id, parse_numbers, resolve_ranking, save_report,
)
from lecture_analyzer.pipeline.scoring import KPScore, is_excluded, score_parts
from lecture_analyzer.reports import ReportStore
from lecture_analyzer.schemas import KPRanking, PageDigest

from .conftest import RoutingProvider
from .test_pipeline_run import _run


def test_borda():
    rankings = {"claude": [1, 2, 3], "gemini": [2, 1, 4]}
    assert borda(rankings, 3, {2: 10, 1: 5}) == [(2, 5), (1, 5), (3, 1), (4, 1)]  # 同分按参考分


def test_resolve_ranking_validates_keys_and_evidence():
    output = KPRanking.model_validate({"ranking": [
        {"kp": "K2", "reason": " 老师说必考 ", "question_types": ["简答题", " "], "evidence": ["q1", "M 2", "X9", "Q1"]},
        {"kp": "K9", "reason": "不存在", "question_types": [], "evidence": []},
        {"kp": "K2", "reason": "重复", "question_types": [], "evidence": []},
        {"kp": "k3", "reason": "第二", "question_types": [], "evidence": ["P1"]},
        {"kp": "K4", "reason": "超出数量", "question_types": [], "evidence": []},
    ]})
    entries = resolve_ranking(output, {2, 3, 4}, {"Q1", "M2", "P1"}, top_n=2)
    assert [(e.key, e.reason, e.types, e.evidence) for e in entries] == [
        (2, "老师说必考", ["简答题"], ["Q1", "M2"]), (3, "第二", [], ["P1"])
    ]
    assert normalize_id(" e 12 ") == "E12" and normalize_id("K3") == ""


def _kp(key, seqs=(1,), minutes=12.0):
    appearances = tuple(KPAppearance(lecture_id=s, seq=s, note_index=1, role="first", minutes=minutes) for s in seqs)
    return KnowledgePoint(key=key, name=f"知识点{key}", chapter="总论", aliases=(), appearances=appearances)


def _signal(type_, strength=2, seq=1):
    return HistorySignal(key=1, lecture_id=seq, seq=seq, type=type_, strength=strength, quote="原话", segment_ids=[1])


def _hit(kind, item_type="term", score=None, item_id=1):
    material = SimpleNamespace(id=10, kind=kind, title="模拟题（样例）")
    return SimpleNamespace(material=material, item=SimpleNamespace(id=item_id, idx=item_id, label=f"{item_id}",
                                                                   item_type=item_type, score=score))


def test_score_parts_follow_weights_and_bonuses():
    item = KPScore(_kp(1, seqs=(1, 3), minutes=12), [_signal("exam_hint", 3), _signal("skipped_slide", 1)],
                   [_hit("mock_exam", score=10, item_id=1)])
    cfg = ScoringConfig(signals={"exam_hint": 6})
    parts = score_parts(item, cfg, {1: "normal", 3: "review"}, {1: 2.0})
    labels = {part.label: part.points for part in parts}
    assert labels["第1讲 提示考试 强度3 ×6"] == 18  # 配置覆盖默认权重
    assert labels["第1讲 跳过的课件 强度1 ×-1"] == -1
    assert any(label.endswith("×4×分值系数2") and points == 8 for label, points in labels.items())
    assert labels["在 2 讲出现"] == 2 and labels["讲了约 24 分钟"] == 2
    assert labels["复习课（第3讲）里出现"] == 3 and labels["课堂和资料里都出现"] == 3
    item.parts = parts
    assert item.score == 18 - 1 + 8 + 2 + 2 + 3 + 3


def test_exclusion_needs_no_include():
    assert is_excluded(KPScore(_kp(1), [_signal("scope_exclude", 3)], []))
    assert not is_excluded(KPScore(_kp(1), [_signal("scope_exclude", 2)], []))  # "不用都背"之类：只扣分
    assert not is_excluded(KPScore(_kp(1), [_signal("scope_exclude", 3), _signal("exam_hint")], []))
    assert is_excluded(KPScore(_kp(1), [], [_hit("notice", "scope_exclude")]))
    assert not is_excluded(KPScore(_kp(1), [], [_hit("notice", "scope_exclude"), _hit("notice", "scope_include")]))


def test_parse_numbers():
    assert parse_numbers("1, 2，5", "讲次") == [1, 2, 5] and parse_numbers(None, "讲次") == []
    with pytest.raises(LibraryError, match="讲次"):
        parse_numbers("1-3", "讲次")


# ---------- 端到端（假模型） ----------


def _prepare(m1_env, monkeypatch):
    """处理完一讲，再导入一份模拟题并对应到 K1。"""
    _run(m1_env)
    routes = {
        "ItemBatch": [json.dumps({"continuation_md": "", "items": [
            {"label": "1", "item_type": "short", "content_md": "阿司匹林小剂量的作用？", "score": 10, "answer_md": "",
             "pages": [1]}]}, ensure_ascii=False)],
        "ItemMappingBatch": [json.dumps({"mappings": [{"item": 1, "kps": ["K1"]}]})],
    }
    monkeypatch.setattr(materials_module, "create_provider",
                        lambda profile: RoutingProvider(routes, profile.model_copy(update={"model": "fake-model"})))
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        store = MaterialStore(library)
        mock = store.add(course, "mock_exam", "模拟题（样例）", "mock.pdf")
        store.set_file(mock.id, "materials/x/mock.pdf", 1)
        store.save_pages(mock.id, [PageDigest(page_no=1, title="", content_md="1. 阿司匹林小剂量的作用？", terms=[])])
        store.set_step(mock.id, "digest", "done")
        materials_module.process_materials(m1_env.cfg, library, course, report=lambda _: None)


REVIEW = {"overview_md": "总论一章", "entries": [{"kp": "K1", "summary_md": "小于 75 mg 抗血凝；大剂量解热镇痛",
                                                  "sources": ["L3N1"], "emphasis": ["Q1"], "examples": []}]}


def _fake_models(monkeypatch, routes, failing=()):
    """排名和更新总复习都换成假模型（总复习在报告里按需生成）；failing 里的档案直接报错。"""
    created: list[RoutingProvider] = []

    def fake_provider(profile):
        if profile.name in failing:
            raise LLMError("中转站不可用")
        created.append(RoutingProvider(routes, profile.model_copy(update={"model": "fake-model"})))
        return created[-1]

    monkeypatch.setattr(report_module, "create_provider", fake_provider)
    monkeypatch.setattr(review_module, "create_provider", fake_provider)
    return created


def _schemas(created):
    return [call["schema"].__name__ for provider in created for call in provider.calls]


def test_report_gives_points_themselves(m1_env, monkeypatch):
    # M4 修订：只给判断后的知识点本身（名称、星级、可能题型、要点），不给理由、分数、名次和原话
    _prepare(m1_env, monkeypatch)
    ranking = {"ranking": [{"kp": "K1", "reason": "老师说期末一定会考，模拟题也出了", "question_types": ["简答题"],
                            "evidence": ["Q1", "M1"]}]}
    created = _fake_models(monkeypatch, {"KPRanking": [json.dumps(ranking, ensure_ascii=False)],
                                         "ReviewChapter": [json.dumps(REVIEW, ensure_ascii=False)]}, failing={"gemini"})
    messages: list[str] = []
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        plan = build_plan(m1_env.cfg, library, course, profiles=["claude", "gemini"])
        assert plan.calls == 2
        result = generate_report(m1_env.cfg, library, plan, report=messages.append)
        report_id, path = save_report(library, m1_env.cfg.data_dir, plan, result)
        stored = ReportStore(library).get(report_id)
        assert "药理学总论" in CourseMemory(library).reviews(course.id)  # 要点取自总复习，相关章节一并生成好了

    assert result["version"] == 2 and set(result) == {
        "version", "course", "generated_at", "models", "scope", "points", "excluded", "warnings"}
    assert result["points"] == [{"rank": 1, "key": 1, "name": "阿司匹林的剂量分段", "chapter": "药理学总论", "priority": 3,
                                 "types": ["简答题"], "summary_md": "小于 75 mg 抗血凝；大剂量解热镇痛", "lectures": [3]}]
    assert any("gemini 没有给出排名" in w for w in result["warnings"])
    assert any("正在更新总复习里相关的 1 章" in m for m in messages)
    assert stored.models == ["claude", "gemini"] and path.name.startswith("考前报告-")
    markdown = path.read_text(encoding="utf-8")
    assert "## 1. 阿司匹林的剂量分段 ★★★" in markdown and "可能题型：简答题" in markdown
    assert "小于 75 mg 抗血凝" in markdown and "这个期末一定会考" not in markdown and "理由" not in markdown
    book = json.loads((path.parent / "book.json").read_text(encoding="utf-8"))
    assert [r["file"] for r in book["reports"]] == [path.name]

    with Library(m1_env.cfg.data_dir / "lecture.db") as library:  # 总复习没变：再出一份报告不重新生成
        generate_report(m1_env.cfg, library, build_plan(m1_env.cfg, library, library.course_by_id(m1_env.course_id)),
                        report=lambda _: None)
    assert _schemas(created).count("ReviewChapter") == 1


def test_report_scope_and_all_models_failing(m1_env, monkeypatch):
    _prepare(m1_env, monkeypatch)
    monkeypatch.setattr(report_module, "create_provider", lambda profile: (_ for _ in ()).throw(LLMError("断开")))
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        with pytest.raises(LibraryError, match="没有第 9 讲"):
            build_plan(m1_env.cfg, library, course, lectures=[9])
        with pytest.raises(LibraryError, match="没有带标签"):
            build_plan(m1_env.cfg, library, course, tag="期中")
        with pytest.raises(LibraryError, match="没有这些编号的资料"):
            build_plan(m1_env.cfg, library, course, materials=[999])
        plan = build_plan(m1_env.cfg, library, course, lectures=[3])
        assert plan.lecture_seqs == [3] and plan.profiles == ["claude"]
        with pytest.raises(LLMError, match="所有模型都没能给出排名"):
            generate_report(m1_env.cfg, library, plan, report=lambda _: None)


def test_review_failure_falls_back_to_lecture_notes(m1_env, monkeypatch):
    _prepare(m1_env, monkeypatch)
    ranking = {"ranking": [{"kp": "K1", "reason": "老师说会考", "question_types": [], "evidence": ["Q1"]}]}
    _fake_models(monkeypatch, {"KPRanking": [json.dumps(ranking, ensure_ascii=False)],
                               "ReviewChapter": [LLMError("中转站不可用")]})
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        plan = build_plan(m1_env.cfg, library, library.course_by_id(m1_env.course_id))
        result = generate_report(m1_env.cfg, library, plan, report=lambda _: None)
    point = result["points"][0]
    assert point["name"] == "阿司匹林的剂量分段" and "**抗血凝**" in point["summary_md"]  # 暂用各讲笔记原文
    assert any("知识点要点没能整理成复习讲义" in w for w in result["warnings"])
