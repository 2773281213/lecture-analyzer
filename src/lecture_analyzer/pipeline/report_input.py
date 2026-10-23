"""考前报告的输入：由代码整理证据并编号（原话 Q、资料题目 M、课件页 P、例题 E），模型只用编号引用。

证据都来自书库：原话只用通过校验的考点信号；资料题目只用与当前课程记忆一致的对应关系；
课件页取自各讲笔记里知识点标注的页（按整理笔记时的快照换算成"资料名 第 N 页"）；例题按句子编号归到最近的知识点。
只用所选范围（讲、资料）里的证据。问答（qa.py）也用这里的证据编号，另加笔记要点 N 和转写片段 T。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..knowledge import CourseMemory, LectureCache
from ..library import Course, Library, PageRef
from ..materials import MaterialStore
from ..render import SIGNAL_LABELS
from ..timecode import format_ms
from .materials import ITEM_TYPES, MaterialHit
from .memory import Item, Section, snippet
from .review import hit_view, nearest_kp
from .scoring import KPScore, Scope, Scoring

EXAM_KINDS = ("past_exam", "mock_exam", "exercise")  # 题型统计用的资料
MAX_PAGES_PER_KP = 4
MAX_EXAMPLES_PER_KP = 3


@dataclass
class Evidence:
    """编号 → 展示用的信息（网页和 Markdown 共用）；by_kp：知识点 → 它的证据编号。"""

    views: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_kp: dict[int, list[str]] = field(default_factory=dict)
    item_ids: dict[int, str] = field(default_factory=dict)  # 一道题对应几个知识点时共用一个 M 编号

    def add(self, prefix: str, key: int, view: dict[str, Any]) -> str:
        number = f"{prefix}{sum(1 for k in self.views if k.startswith(prefix)) + 1}"
        self.views[number] = {"id": number, **view}
        self.link(key, number)
        return number

    def link(self, key: int, number: str) -> None:
        numbers = self.by_kp.setdefault(key, [])
        if number not in numbers:
            numbers.append(number)

    def of(self, key: int, prefix: str = "") -> list[str]:
        return [n for n in self.by_kp.get(key, []) if n.startswith(prefix)]


def build_evidence(library: Library, course: Course, scoring: Scoring, scope: Scope = Scope()) -> Evidence:
    evidence = Evidence()
    cache = LectureCache(library)
    store = MaterialStore(library)
    examples = _examples_by_kp(library, course, cache, scope)
    for item in [*scoring.ranked, *scoring.excluded]:
        key = item.kp.key
        for signal in sorted(item.signals, key=lambda s: (-s.strength, s.seq)):
            evidence.add("Q", key, {
                "label": SIGNAL_LABELS.get(signal.type, signal.type), "type": signal.type, "strength": signal.strength,
                "quote": signal.quote, "meaning": signal.meaning, "seq": signal.seq, "lecture_id": signal.lecture_id,
                "t": cache.time(signal.lecture_id, signal.segment_ids),
            })
        for hit in item.hits:
            _add_item(evidence, key, hit)
        for ref, label, title in _kp_pages(library, store, cache, item)[:MAX_PAGES_PER_KP]:
            evidence.add("P", key, {"material_id": ref.material_id, "page_no": ref.page_no, "label": label,
                                    "title": title})
        for lecture_id, seq, index in examples.get(key, [])[:MAX_EXAMPLES_PER_KP]:
            example = cache.notes(lecture_id).examples[index - 1]  # type: ignore[union-attr]
            evidence.add("E", key, {"title": example.title, "problem_md": example.problem_md,
                                    "solution_md": example.solution_md, "seq": seq, "lecture_id": lecture_id,
                                    "t": cache.time(lecture_id, example.segment_ids)})
    return evidence


def _add_item(evidence: Evidence, key: int, hit: MaterialHit) -> None:
    number = evidence.item_ids.get(hit.item.id)
    if number is None:
        number = evidence.add("M", key, hit_view(hit))
        evidence.item_ids[hit.item.id] = number
    else:
        evidence.link(key, number)


def _kp_pages(
    library: Library, store: MaterialStore, cache: LectureCache, item: KPScore
) -> list[tuple[PageRef, str, str]]:
    """知识点在各讲笔记里标注的课件页，换算成课程资料里的页。"""
    refs: list[PageRef] = []
    for appearance in item.kp.appearances:
        note = cache.note_kp(appearance.lecture_id, appearance.note_index)
        snapshot = library.note_pages(appearance.lecture_id)
        for p in note.slide_pages if note else []:
            if 1 <= p <= len(snapshot) and snapshot[p - 1] not in refs:
                refs.append(snapshot[p - 1])
    titles = store.page_digests(refs)
    return [(ref, label, titles[ref].title if ref in titles else "")
            for ref, label in zip(refs, store.page_labels(refs))]


def _examples_by_kp(
    library: Library, course: Course, cache: LectureCache, scope: Scope
) -> dict[int, list[tuple[int, int, int]]]:
    """范围内各讲的例题按句子编号归到最近的知识点：{K 编号: [(讲的编号, 讲次, 第几道例题)]}。"""
    memory = CourseMemory(library)
    result: dict[int, list[tuple[int, int, int]]] = {}
    for lecture in library.lectures(course):
        if not scope.has_lecture(lecture.id):
            continue
        notes = cache.notes(lecture.id)
        keys = memory.kp_keys(lecture.id)
        for index, example in enumerate(notes.examples if notes else [], start=1):
            note_index = nearest_kp(example.segment_ids, notes)  # type: ignore[arg-type]
            if note_index in keys:
                result.setdefault(keys[note_index], []).append((lecture.id, lecture.seq, index))
    return result


# ---------- 范围说明与题型统计 ----------


def scope_statements(scoring: Scoring, evidence: Evidence) -> list[dict[str, Any]]:
    """老师在课上说的"要考 / 不考"，以及考试通知里的条目。"""
    statements: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*scoring.ranked, *scoring.excluded]:
        for number in evidence.of(item.kp.key, "Q"):
            view = evidence.views[number]
            if view["type"] in ("scope_include", "scope_exclude", "exam_hint") and number not in seen:
                seen.add(number)
                statements.append({"id": number, "include": view["type"] != "scope_exclude", "kp": item.kp.name,
                                   "text": view["quote"], "source": _lecture_source(view)})
        for hit in item.hits:
            if hit.material.kind == "notice" and f"notice-{hit.item.id}" not in seen:
                seen.add(f"notice-{hit.item.id}")
                statements.append({"id": evidence.item_ids.get(hit.item.id, ""), "kp": item.kp.name,
                                   "include": hit.item.item_type != "scope_exclude", "text": hit.item.content_md,
                                   "source": f"「{hit.material.title}」{hit.item.label or f'第 {hit.item.idx} 条'}"})
    return statements


def _lecture_source(view: dict[str, Any]) -> str:
    when = f" {format_ms(view['t'])}" if view.get("t") is not None else ""
    return f"第{view['seq']}讲{when}"


def type_stats(library: Library, course: Course, material_ids: frozenset[int] | None) -> list[dict[str, Any]]:
    """往年真题、模拟题、习题里各题型的数量和分值（所有题目都算，不论是否对应到知识点）。"""
    store = MaterialStore(library)
    counts: Counter[str] = Counter()
    scores: Counter[str] = Counter()
    for material in store.list(course, EXAM_KINDS):
        if material_ids is not None and material.id not in material_ids:
            continue
        for item in store.items(material.id):
            label = ITEM_TYPES.get(item.item_type, "其他")
            counts[label] += 1
            scores[label] += item.score or 0
    return [{"type": label, "count": count, "score": scores[label]} for label, count in counts.most_common()]


# ---------- 给模型看的输入 ----------


def rank_sections(scoring: Scoring, evidence: Evidence, statements: Sequence[dict[str, Any]],
                  stats: Sequence[dict[str, Any]]) -> list[Section]:
    """各模型排名时的输入；超出预算时先精简课件页和例题，再精简资料题目和原话。"""
    kp_lines = [_kp_line(item) for item in scoring.ranked]
    excluded = [f"[K{item.kp.key}] {item.kp.name}（老师或考试通知说不考）" for item in scoring.excluded]
    return [
        Section("课程知识点（按程序参考分从高到低）", [Item(line, line) for line in kp_lines + excluded]),
        *evidence_sections(evidence),
        Section("考试范围说明", [Item(_statement_text(s), _statement_text(s)) for s in statements]),
        Section("题型统计（往年真题、模拟题、习题）", [Item(_stat_text(s), _stat_text(s)) for s in stats]),
    ]


def evidence_sections(evidence: Evidence) -> list[Section]:
    """原话、资料题目、课件页、例题四部分（报告和问答共用）；数字小的部分先精简。"""
    return [
        Section("老师原话（已与录音转写核对）", evidence_items(evidence, "Q", _quote_text), trim_rank=4),
        Section("资料里的题目", evidence_items(evidence, "M", _item_text), trim_rank=3),
        Section("课件页", evidence_items(evidence, "P", _page_text), trim_rank=1),
        Section("课堂例题", evidence_items(evidence, "E", _example_text), trim_rank=2),
    ]


def _kp_line(item: KPScore) -> str:
    lectures = "、".join(str(seq) for seq in item.lectures)
    taught = f"第{lectures}讲，约 {item.minutes:.0f} 分钟" if lectures else "课上没讲到（只在资料里）"
    chapter = f"｜{item.kp.chapter}" if item.kp.chapter else ""
    types = f"｜适合题型：{'、'.join(item.types)}" if item.types else ""  # 有考试方案时才有
    return f"[K{item.kp.key}] {item.kp.name}{chapter}｜{taught}{types}｜参考分 {item.score:g}"


def evidence_items(evidence: Evidence, prefix: str, render: Any) -> list[Item]:
    """某一类证据给模型看的文字：[编号] 所属知识点 · 内容。"""
    owners: dict[str, list[int]] = {}
    for key, numbers in evidence.by_kp.items():
        for number in numbers:
            owners.setdefault(number, []).append(key)
    items = []
    for number, view in evidence.views.items():
        if not number.startswith(prefix):
            continue
        keys = "、".join(f"K{key}" for key in owners.get(number, []))
        text = f"[{number}] {keys} · {render(view)}"
        items.append(Item(text, "", keep=view.get("strength", 0)))
    return items


def _quote_text(view: dict[str, Any]) -> str:
    return f"{view['label']}·强度{view['strength']} · {_lecture_source(view)}：「{view['quote']}」"


def _item_text(view: dict[str, Any]) -> str:
    return f"「{view['title']}」{view['label']}（{view['kind']}·{view['type']}）：{snippet(view['content_md'], 300)}"


def _page_text(view: dict[str, Any]) -> str:
    return f"{view['label']}：{view['title']}"


def _example_text(view: dict[str, Any]) -> str:
    return f"第{view['seq']}讲 {view['title']}：{snippet(view['problem_md'], 200)}"


def _statement_text(statement: dict[str, Any]) -> str:
    kind = "要考" if statement["include"] else "不考"
    return f"- 【{kind}】{statement['source']}（{statement['kp']}）：{snippet(statement['text'], 200)}"


def _stat_text(stat: dict[str, Any]) -> str:
    score = f"，共 {stat['score']:g} 分" if stat["score"] else ""
    return f"- {stat['type']}：{stat['count']} 题{score}"
