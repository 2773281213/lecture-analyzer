"""考点打分：纯代码计算，权重全部可配置（[scoring]），每个知识点都给出分数的构成。

- 课堂：只计通过原话校验的信号，权重 × 强度；
- 资料：往年真题、模拟题、复习提纲、考试通知等里出现的题目，权重 × 分值系数（按同一份资料里分值的中位数折算，0.5–上限）；
- 加分：多讲出现、讲解时长、复习课里出现、课堂和资料里都出现；
- 老师明确说不考（强度 3）或考试通知说不考、又没有说要考的知识点，分数清零，单独列出；
  较弱的"不考"（如"不用每种都背""这部分了解即可"）往往只排除一部分内容，只扣分，不整体排除。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import median

from ..config import ScoringConfig
from ..knowledge import CourseMemory, HistorySignal, KnowledgePoint
from ..library import Course, Library
from ..materials import MATERIAL_KINDS
from ..render import KIND_LABELS, SIGNAL_LABELS
from .materials import ITEM_TYPES, MaterialHit, material_hits

DEFAULT_SIGNAL_WEIGHTS = {
    "exam_hint": 5, "scope_include": 4, "emphasis": 3, "common_mistake": 2, "worked_example": 2, "derivation": 2,
    "homework": 2, "recap": 1, "off_slide": 1, "skipped_slide": -1, "scope_exclude": -1,
}
DEFAULT_MATERIAL_WEIGHTS = {
    "past_exam": 5, "mock_exam": 4, "review_outline": 4, "notice": 3, "exercise": 2, "other": 1,
}
INCLUDE_TYPES = {"exam_hint", "scope_include"}  # 说明要考（课堂信号，或考试通知里的条目）
STRONG_EXCLUDE = 3  # 课堂上"不考"的说法强度达到这个值（明确说不考）才整体排除知识点


@dataclass
class ScorePart:
    label: str  # 构成说明，如「提示考试 强度3 ×5」
    points: float


@dataclass
class KPScore:
    kp: KnowledgePoint
    signals: list[HistorySignal]
    hits: list[MaterialHit]
    parts: list[ScorePart] = field(default_factory=list)
    excluded: bool = False  # 老师或考试通知说不考

    @property
    def score(self) -> float:
        return 0.0 if self.excluded else round(sum(part.points for part in self.parts), 2)

    @property
    def lectures(self) -> list[int]:
        return sorted({a.seq for a in self.kp.appearances})

    @property
    def minutes(self) -> float:
        return sum(a.minutes for a in self.kp.appearances)


@dataclass
class Scoring:
    ranked: list[KPScore]  # 按分数从高到低，不含"不考"的
    excluded: list[KPScore]

    def by_key(self) -> dict[int, KPScore]:
        return {item.kp.key: item for item in [*self.ranked, *self.excluded]}


@dataclass(frozen=True)
class Scope:
    """参与打分的范围：哪些讲、哪些资料；None 表示全部。"""

    lecture_ids: frozenset[int] | None = None
    material_ids: frozenset[int] | None = None

    def has_lecture(self, lecture_id: int) -> bool:
        return self.lecture_ids is None or lecture_id in self.lecture_ids

    def has_material(self, material_id: int) -> bool:
        return self.material_ids is None or material_id in self.material_ids


def score_course(library: Library, course: Course, cfg: ScoringConfig, scope: Scope = Scope()) -> Scoring:
    memory = CourseMemory(library)
    kinds = {lecture.id: lecture.kind for lecture in library.lectures(course)}
    signals = _group_signals(memory.history_signals(course), scope)
    hits = {key: [h for h in values if scope.has_material(h.material.id)]
            for key, values in material_hits(library, course).items()}
    factors = _score_factors([h for values in hits.values() for h in values], cfg.max_score_factor)
    scores = []
    for kp in _in_scope(memory.tree(course), scope, hits):
        item = KPScore(kp, signals.get(kp.key, []), hits.get(kp.key, []))
        item.parts = score_parts(item, cfg, kinds, factors)
        item.excluded = is_excluded(item)
        scores.append(item)
    ranked = sorted((s for s in scores if not s.excluded), key=lambda s: (-s.score, _first_seq(s), s.kp.key))
    return Scoring(ranked, [s for s in scores if s.excluded])


def _in_scope(
    tree: Sequence[KnowledgePoint], scope: Scope, hits: dict[int, list[MaterialHit]]
) -> list[KnowledgePoint]:
    """所选范围里讲过、或在资料里出现过的知识点；出现记录只保留范围内的讲。"""
    points = []
    for kp in tree:
        appearances = tuple(a for a in kp.appearances if scope.has_lecture(a.lecture_id))
        if appearances or hits.get(kp.key):
            points.append(KnowledgePoint(kp.key, kp.name, kp.chapter, kp.aliases, appearances))
    return points


def _group_signals(history: Sequence[HistorySignal], scope: Scope) -> dict[int, list[HistorySignal]]:
    grouped: dict[int, list[HistorySignal]] = {}
    for signal in history:
        if scope.has_lecture(signal.lecture_id):
            grouped.setdefault(signal.key, []).append(signal)
    return grouped


def _first_seq(item: KPScore) -> int:
    return min((a.seq for a in item.kp.appearances), default=10**6)


def is_excluded(item: KPScore) -> bool:
    return excluded(item.signals, item.hits)


def excluded(signals: Sequence[HistorySignal], hits: Sequence[MaterialHit]) -> bool:
    """老师明确说不考（强度 3）或考试通知说不考，而且没有任何"要考"的说法（总复习的星级也用这条规则）。"""
    notice = {h.item.item_type for h in hits if h.material.kind == "notice"}
    include = bool(notice & INCLUDE_TYPES) or any(s.type in INCLUDE_TYPES for s in signals)
    strong = "scope_exclude" in notice or any(
        s.type == "scope_exclude" and s.strength >= STRONG_EXCLUDE for s in signals
    )
    return strong and not include


def score_parts(item: KPScore, cfg: ScoringConfig, kinds: dict[int, str], factors: dict[int, float]) -> list[ScorePart]:
    signal_weights = {**DEFAULT_SIGNAL_WEIGHTS, **cfg.signals}
    material_weights = {**DEFAULT_MATERIAL_WEIGHTS, **cfg.materials}
    parts = []
    for signal in item.signals:
        weight = signal_weights.get(signal.type, 0)
        if weight:
            label = f"第{signal.seq}讲 {SIGNAL_LABELS.get(signal.type, signal.type)} 强度{signal.strength} ×{weight:g}"
            parts.append(ScorePart(label, weight * signal.strength))
    for hit in item.hits:
        if hit.item.item_type == "scope_exclude":
            continue
        weight = material_weights.get(hit.material.kind, material_weights.get("other", 1))
        factor = factors.get(hit.item.id, 1.0)
        kind = MATERIAL_KINDS.get(hit.material.kind, hit.material.kind)
        label = (f"「{hit.material.title}」{hit.item.label or f'第 {hit.item.idx} 条'}"
                 f"（{kind}·{ITEM_TYPES.get(hit.item.item_type, '其他')}）"
                 f" ×{weight:g}" + (f"×分值系数{factor:.2g}" if factor != 1 else ""))
        parts.append(ScorePart(label, round(weight * factor, 2)))
    return parts + _bonuses(item, cfg, kinds)


def _bonuses(item: KPScore, cfg: ScoringConfig, kinds: dict[int, str]) -> list[ScorePart]:
    bonuses = []
    lectures = item.lectures
    if len(lectures) > 1:
        bonuses.append(ScorePart(f"在 {len(lectures)} 讲出现", cfg.per_extra_lecture * (len(lectures) - 1)))
    minutes_bonus = min(cfg.max_minutes_bonus, cfg.per_10_minutes * int(item.minutes // 10))
    if minutes_bonus > 0:
        bonuses.append(ScorePart(f"讲了约 {item.minutes:.0f} 分钟", minutes_bonus))
    review = [a.seq for a in item.kp.appearances if kinds.get(a.lecture_id) == "review"]
    if review:
        bonuses.append(ScorePart(f"{KIND_LABELS['review']}（第{review[0]}讲）里出现", cfg.review_lecture))
    exam_hits = [h for h in item.hits if h.item.item_type != "scope_exclude"]
    if exam_hits and (item.signals or item.kp.appearances):
        bonuses.append(ScorePart("课堂和资料里都出现", cfg.class_and_material))
    return bonuses


def _score_factors(hits: Sequence[MaterialHit], cap: float) -> dict[int, float]:
    """分值系数：题目分值 ÷ 同一份资料里有分值的题目的中位数，限制在 0.5 到 cap 之间；没写分值的为 1。"""
    by_material: dict[int, dict[int, float]] = {}  # 一道题对应几个知识点时只算一次
    for hit in hits:
        if hit.item.score:
            by_material.setdefault(hit.material.id, {})[hit.item.id] = hit.item.score
    factors = {}
    for hit in hits:
        scores = by_material.get(hit.material.id)
        if hit.item.score and scores:
            factors[hit.item.id] = min(cap, max(0.5, hit.item.score / median(scores.values())))
    return factors
