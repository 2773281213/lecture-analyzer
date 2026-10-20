import json

from lecture_analyzer.library import Library
from lecture_analyzer.materials import Material, MaterialItem, MaterialStore
from lecture_analyzer.pipeline import materials as materials_module
from lecture_analyzer.pipeline.materials import (
    map_items, material_hits, memory_fingerprint, process_materials, split_items, stale_materials,
)
from lecture_analyzer.pipeline.review import kp_priority, render_review_markdown, review_book
from lecture_analyzer.schemas import PageDigest

from .conftest import FakeProvider, RoutingProvider
from .test_pipeline_run import _run

MOCK = Material(id=7, course_id=1, kind="mock_exam", title="模拟题（样例）", path="", source="", position=1, page_count=3)


def _items_json(items, continuation=""):
    return json.dumps({"continuation_md": continuation, "items": items}, ensure_ascii=False)


def _item(label, content, item_type="term", score=0, pages=(1,), answer=""):
    return {"label": label, "item_type": item_type, "content_md": content, "score": score, "answer_md": answer,
            "pages": list(pages)}


def test_split_items_joins_continuation_and_cleans_output(monkeypatch):
    monkeypatch.setattr(materials_module, "ITEMS_BATCH_CHARS", 10)  # 每页一批
    pages = [PageDigest(page_no=n, title="", content_md=f"第{n}页的题目内容", terms=[]) for n in (1, 2)]
    provider = FakeProvider([
        _items_json([_item("一、1", "受体的定义", score=5, pages=(1, 9)), _item("一、2", "论述阿司匹林的作用", "essay")]),
        _items_json([_item("", "", "short"), _item("二、1", "什么是半衰期", "unknown-type", pages=(2,))],
                    continuation="（接上题）以及不良反应"),
    ])
    items, usage = split_items(provider, MOCK, pages, "药理学")
    assert [(i.label, i.item_type, i.score, i.pages) for i in items] == [
        ("一、1", "term", 5, [1]), ("一、2", "essay", None, [1, 2]), ("二、1", "other", None, [2])
    ]
    assert items[1].content_md.endswith("以及不良反应") and usage.input_tokens == 20
    second = provider.calls[1]["parts"]
    assert "上一批最后一条：一、2 论述阿司匹林的作用" in second[1].text
    assert "一份模拟题" in provider.calls[0]["system"]


def test_map_items_keeps_valid_keys_only():
    from lecture_analyzer.knowledge import KnowledgePoint

    tree = [KnowledgePoint(key=k, name=f"知识点{k}", chapter="总论", aliases=(), appearances=()) for k in (1, 2, 3, 4)]
    items = [MaterialItem(label="", item_type="term", content_md=f"题{n}", score=None, answer_md="", pages=[1],
                          id=100 + n, material_id=7, idx=n) for n in (1, 2, 3)]
    response = {"mappings": [{"item": 1, "kps": ["K2", "k9", "K2", "K1", "K3", "K4"]}, {"item": 2, "kps": ["无"]},
                             {"item": 99, "kps": ["K1"]}]}
    mapping, _ = map_items(FakeProvider([json.dumps(response)]), tree, items, "药理学", MOCK)
    assert mapping == {101: [2, 1, 3], 102: [], 103: []}  # 去重、丢掉不存在的编号、最多 3 个；漏掉的题目为空


def test_mock_exam_raises_review_stars(m1_env, monkeypatch):
    _run(m1_env)  # 课程记忆里有 K1「阿司匹林的剂量分段」
    routes = {
        "ItemBatch": [_items_json([_item("1", "阿司匹林小剂量的作用？", "short", score=10)])],
        "ItemMappingBatch": [json.dumps({"mappings": [{"item": 1, "kps": ["K1"]}]})],
    }
    calls: list[RoutingProvider] = []

    def fake_provider(profile):
        calls.append(RoutingProvider(routes, profile.model_copy(update={"model": "fake-model"})))
        return calls[-1]

    monkeypatch.setattr(materials_module, "create_provider", fake_provider)
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        store = MaterialStore(library)
        mock = store.add(course, "mock_exam", "模拟题（样例）", "mock.pdf")
        store.set_file(mock.id, "materials/x/mock.pdf", 1)
        store.save_pages(mock.id, [PageDigest(page_no=1, title="", content_md="1. 阿司匹林小剂量的作用？（10分）", terms=[])])
        store.set_step(mock.id, "digest", "done")
        process_materials(m1_env.cfg, library, course, report=lambda _: None)
        process_materials(m1_env.cfg, library, course, report=lambda _: None)  # 已处理完：不再调用模型
        assert len(calls) == 2

        hits = material_hits(library, course)
        assert [(h.item.label, h.item.score) for h in hits[1]] == [("1", 10)]
        entry = review_book(library, course)[0].entries[0]
        assert entry.priority == 3 and entry.materials[0]["type"] == "简答题，10 分"
        assert "资料：「模拟题（样例）」1（简答题，10 分）" in render_review_markdown(course, review_book(library, course))

        # 课程记忆变了：对应关系过期，不再参与星级，直到重新对应
        store.set_step(mock.id, "kp_map", "done", fingerprint="old")
        assert [m.id for m in stale_materials(library, course)] == [mock.id] and material_hits(library, course) == {}


def test_notice_scope_exclude_and_include():
    from types import SimpleNamespace

    notice = SimpleNamespace(kind="notice")
    exclude = SimpleNamespace(material=notice, item=SimpleNamespace(item_type="scope_exclude"))
    include = SimpleNamespace(material=notice, item=SimpleNamespace(item_type="scope_include"))
    past = SimpleNamespace(material=SimpleNamespace(kind="past_exam"), item=SimpleNamespace(item_type="term"))
    assert kp_priority([], 1, [exclude]) == 0
    assert kp_priority([], 1, [exclude, past]) == 0  # 往年题不能推翻今年"不考"的说明
    assert kp_priority([], 1, [include]) == 3 and kp_priority([], 1, [past]) == 3 and kp_priority([], 1) == 1


def test_memory_fingerprint_changes_with_names():
    from lecture_analyzer.knowledge import KnowledgePoint

    a = [KnowledgePoint(key=1, name="受体", chapter="", aliases=(), appearances=())]
    b = [KnowledgePoint(key=1, name="受体学说", chapter="", aliases=(), appearances=())]
    assert memory_fingerprint(a) != memory_fingerprint(b) and memory_fingerprint(a) == memory_fingerprint(list(a))


def test_one_failing_material_does_not_stop_the_others(tmp_path, monkeypatch, example_config):
    import pytest

    from lecture_analyzer.config import load_config
    from lecture_analyzer.course import CourseSettings
    from lecture_analyzer.llm import LLMError

    cfg = load_config(example_config)
    done: list[str] = []

    def fake_digest(cfg, library, material, report):
        if material.title == "坏的":
            raise LLMError("中转站断开")
        done.append(material.title)

    monkeypatch.setattr(materials_module, "digest_material", fake_digest)
    with Library(tmp_path / "lecture.db") as library:
        course = library.add_course("病毒学", CourseSettings())
        store = MaterialStore(library)
        for title in ("坏的", "好的"):
            store.add(course, "slides", title, f"{title}.pdf")
        with pytest.raises(LLMError, match="1 份资料没处理完（坏的）"):
            process_materials(cfg, library, library.course_by_id(course.id), report=lambda _: None)
    assert done == ["好的"]
