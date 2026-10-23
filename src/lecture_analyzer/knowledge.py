"""课程记忆（L2）与总复习的存储。

L2 完全由各讲的整合结果按讲次顺序叠加而成：整合日志保存每讲解析成课程内编号（K 编号）后的结果，
重建时清空一门课的 L2，再按讲次重放日志，不调用模型，所以结果是确定的。
对已有知识点只补别名（沿用最早的规范名称），章节为空时才补，保证重放顺序决定一切。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from pydantic import BaseModel

from .asr.base import Segment
from .library import Course, Library
from .llm import ModelProfile
from .render import first_time
from .schemas import KnowledgePointNote, LectureNotes
from .textnorm import normalize_zh


class ResolvedKP(BaseModel):
    note_index: int  # 本讲笔记中第几个知识点（从 1 开始）
    key: int  # 课程知识点编号（提示词里的 K 编号）
    is_new: bool
    name: str  # 新知识点的规范名称；已有知识点时是本讲的叫法（用来补别名）
    chapter: str
    aliases: list[str]
    role: str  # first / deepen / review / apply
    minutes: float


class ResolvedReference(BaseModel):
    note_index: int
    key: int
    target_seq: int  # 被联系内容所在的讲次
    relation: str  # recap / builds_on / contrast / repeat
    explanation: str
    segment_ids: list[int]


class ResolvedIntegration(BaseModel):
    """一讲整合结果（已校验、已分配编号），写入整合日志，重建时原样重放。"""

    kps: list[ResolvedKP]
    signal_keys: list[int | None]  # 本讲第 i 条信号归属的课程知识点编号；未归类为 None
    references: list[ResolvedReference]


@dataclass(frozen=True)
class KPAppearance:
    lecture_id: int
    seq: int
    note_index: int
    role: str
    minutes: float


@dataclass(frozen=True)
class KnowledgePoint:
    key: int
    name: str
    chapter: str
    aliases: tuple[str, ...]
    appearances: tuple[KPAppearance, ...]  # 按讲次排序

    @property
    def norm_names(self) -> set[str]:
        return {normalize_zh(name) for name in (self.name, *self.aliases)}


@dataclass(frozen=True)
class Reference:
    lecture_id: int  # 引用方（本讲）
    note_index: int
    key: int
    kp_name: str
    target_lecture_id: int
    target_seq: int
    target_note_index: int | None  # 被联系的知识点在那一讲笔记中的位置，用来算跳转时间
    relation: str
    explanation: str
    segment_ids: list[int]


@dataclass(frozen=True)
class HistorySignal:
    """已归入课程知识点、通过原话校验的考点信号。"""

    key: int
    lecture_id: int
    seq: int
    type: str
    strength: int
    quote: str
    segment_ids: list[int]
    signal_id: int = 0
    meaning: str = ""  # 原话不是笔记语言时的意思


@dataclass(frozen=True)
class StoredReview:
    chapter: str
    fingerprint: str
    result_json: str
    model: str
    created_at: str


class LectureCache:
    """按需读取并缓存各讲的笔记和转写：整合、总复习、网页展示时会反复查各讲知识点的位置和时间。"""

    def __init__(self, library: Library) -> None:
        self.library = library
        self._notes: dict[int, LectureNotes | None] = {}
        self._segments: dict[int, list[Segment]] = {}

    def notes(self, lecture_id: int) -> LectureNotes | None:
        if lecture_id not in self._notes:
            self._notes[lecture_id] = self.library.notes(lecture_id)
        return self._notes[lecture_id]

    def segments(self, lecture_id: int) -> list[Segment]:
        if lecture_id not in self._segments:
            self._segments[lecture_id] = self.library.segments(lecture_id)
        return self._segments[lecture_id]

    def time(self, lecture_id: int, segment_ids: Sequence[int]) -> int | None:
        """这些句子里第一句的开始时间（毫秒）。"""
        return first_time(segment_ids, self.segments(lecture_id))

    def kp_question_types(self, kp: KnowledgePoint) -> list[str]:
        """课程知识点适合的题型：合并它在各讲笔记里标注的题型（按考试方案整理的笔记才有），按出现顺序去重。"""
        types: list[str] = []
        for appearance in kp.appearances:
            note = self.note_kp(appearance.lecture_id, appearance.note_index)
            types += note.question_types if note else []
        return list(dict.fromkeys(t.strip() for t in types if t.strip()))

    def note_kp(self, lecture_id: int, note_index: int | None) -> KnowledgePointNote | None:
        """某讲笔记中第 note_index 个知识点。"""
        notes = self.notes(lecture_id)
        if notes is None or note_index is None or not 1 <= note_index <= len(notes.knowledge_points):
            return None
        return notes.knowledge_points[note_index - 1]

    def kp_time(self, lecture_id: int, note_index: int | None) -> int | None:
        """某讲笔记中第 note_index 个知识点开始讲的时间（毫秒）。"""
        note = self.note_kp(lecture_id, note_index)
        return self.time(lecture_id, note.segment_ids) if note else None


class CourseMemory:
    """一门课的 L2 读写；连接沿用书库的连接。"""

    def __init__(self, library: Library) -> None:
        self.library = library
        self.conn = library.conn

    # ---------- 写 ----------

    def reset(self, course_id: int) -> None:
        """清空一门课的 L2（各讲的对应、信号归属、跨讲联系随知识点级联删除）；整合日志保留，供重放。"""
        self.conn.execute("DELETE FROM knowledge_point WHERE course_id = ?", (course_id,))
        self.conn.commit()

    def apply(self, course: Course, lecture_id: int, resolved: ResolvedIntegration) -> None:
        """把一讲的整合结果叠加到课程的 L2 上。"""
        ids = self._kp_ids(course.id)
        for kp in resolved.kps:
            if kp.key in ids:
                self._merge_names(ids[kp.key], kp)
            else:
                aliases = json.dumps(_unique(kp.aliases, kp.name), ensure_ascii=False)
                cursor = self.conn.execute(
                    "INSERT INTO knowledge_point (course_id, key, name, chapter, aliases) VALUES (?, ?, ?, ?, ?)",
                    (course.id, kp.key, kp.name, kp.chapter, aliases),
                )
                ids[kp.key] = cursor.lastrowid
            self.conn.execute(
                "INSERT INTO kp_lecture (kp_id, lecture_id, note_index, role, minutes) VALUES (?, ?, ?, ?, ?)",
                (ids[kp.key], lecture_id, kp.note_index, kp.role, kp.minutes),
            )
        signal_ids = self.library.signal_ids(lecture_id)
        self.conn.executemany(
            "INSERT INTO kp_signal (signal_id, kp_id) VALUES (?, ?)",
            [(sid, ids[key]) for sid, key in zip(signal_ids, resolved.signal_keys) if key in ids],
        )
        by_seq = {lecture.seq: lecture.id for lecture in self.library.lectures(course)}
        self.conn.executemany(
            "INSERT INTO kp_reference (lecture_id, note_index, kp_id, target_lecture_id, relation, explanation,"
            " segment_ids) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (lecture_id, ref.note_index, ids[ref.key], by_seq[ref.target_seq], ref.relation, ref.explanation,
                 json.dumps(ref.segment_ids))
                for ref in resolved.references
                if ref.key in ids and ref.target_seq in by_seq
            ],
        )
        self.conn.commit()

    def _kp_ids(self, course_id: int) -> dict[int, int]:
        rows = self.conn.execute("SELECT id, key FROM knowledge_point WHERE course_id = ?", (course_id,))
        return {row["key"]: row["id"] for row in rows}

    def _merge_names(self, kp_id: int, kp: ResolvedKP) -> None:
        row = self.conn.execute("SELECT name, chapter, aliases FROM knowledge_point WHERE id = ?", (kp_id,)).fetchone()
        aliases = _unique([*json.loads(row["aliases"]), kp.name, *kp.aliases], row["name"])
        chapter = row["chapter"] or kp.chapter
        self.conn.execute(
            "UPDATE knowledge_point SET chapter = ?, aliases = ? WHERE id = ?",
            (chapter, json.dumps(aliases, ensure_ascii=False), kp_id),
        )

    def save_log(self, lecture_id: int, resolved: ResolvedIntegration, profile: ModelProfile) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO integration_log (lecture_id, result_json, model_profile, model, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (lecture_id, resolved.model_dump_json(), profile.name, profile.model, _now()),
        )
        self.conn.commit()

    def log(self, lecture_id: int) -> ResolvedIntegration | None:
        row = self.conn.execute(
            "SELECT result_json FROM integration_log WHERE lecture_id = ?", (lecture_id,)
        ).fetchone()
        return ResolvedIntegration.model_validate_json(row["result_json"]) if row else None

    def log_model(self, lecture_id: int) -> str | None:
        row = self.conn.execute("SELECT model FROM integration_log WHERE lecture_id = ?", (lecture_id,)).fetchone()
        return row["model"] if row else None

    def drop_lecture(self, lecture_id: int) -> None:
        """一讲移出课程时：删掉它在 L2 里的痕迹和整合日志（被别的讲联系到的条目一并删除）。"""
        for sql in (
            "DELETE FROM kp_lecture WHERE lecture_id = ?",
            "DELETE FROM kp_reference WHERE lecture_id = ? OR target_lecture_id = ?",
            "DELETE FROM kp_signal WHERE signal_id IN (SELECT id FROM signal WHERE lecture_id = ?)",
            "DELETE FROM integration_log WHERE lecture_id = ?",
        ):
            self.conn.execute(sql, (lecture_id,) * sql.count("?"))
        self.conn.commit()

    # ---------- 读 ----------

    def tree(self, course: Course) -> list[KnowledgePoint]:
        """课程知识点，按编号（即首次出现的先后）排序；没有出现在任何讲里的知识点不列出。"""
        seq_of = {lecture.id: lecture.seq for lecture in self.library.lectures(course)}
        appearances: dict[int, list[KPAppearance]] = {}
        rows = self.conn.execute(
            "SELECT kl.* FROM kp_lecture kl JOIN knowledge_point k ON k.id = kl.kp_id WHERE k.course_id = ?",
            (course.id,),
        )
        for row in rows:
            appearances.setdefault(row["kp_id"], []).append(
                KPAppearance(row["lecture_id"], seq_of.get(row["lecture_id"], 0), row["note_index"], row["role"],
                             row["minutes"])
            )
        points = []
        for row in self.conn.execute("SELECT * FROM knowledge_point WHERE course_id = ? ORDER BY key", (course.id,)):
            found = sorted(appearances.get(row["id"], []), key=lambda a: (a.seq, a.note_index))
            if found:
                aliases = tuple(json.loads(row["aliases"]))
                points.append(KnowledgePoint(row["key"], row["name"], row["chapter"], aliases, tuple(found)))
        return points

    def kp_keys(self, lecture_id: int) -> dict[int, int]:
        """本讲笔记第几个知识点 → 课程知识点编号。"""
        rows = self.conn.execute(
            "SELECT kl.note_index, k.key FROM kp_lecture kl JOIN knowledge_point k ON k.id = kl.kp_id"
            " WHERE kl.lecture_id = ?",
            (lecture_id,),
        )
        return {row["note_index"]: row["key"] for row in rows}

    def signal_keys(self, lecture_id: int) -> list[int | None]:
        """本讲各条信号归属的课程知识点编号，顺序与 Library.signals() 一致。"""
        rows = self.conn.execute(
            "SELECT k.key FROM signal s LEFT JOIN kp_signal ks ON ks.signal_id = s.id"
            " LEFT JOIN knowledge_point k ON k.id = ks.kp_id WHERE s.lecture_id = ? ORDER BY s.id",
            (lecture_id,),
        )
        return [row["key"] for row in rows]

    def references(self, lecture_id: int) -> list[Reference]:
        """本讲与前几讲的联系。"""
        rows = self.conn.execute(
            "SELECT r.*, k.key, k.name, l.seq AS target_seq,"
            " (SELECT MIN(note_index) FROM kp_lecture WHERE kp_id = r.kp_id AND lecture_id = r.target_lecture_id)"
            " AS target_note_index"
            " FROM kp_reference r JOIN knowledge_point k ON k.id = r.kp_id JOIN lecture l ON l.id = r.target_lecture_id"
            " WHERE r.lecture_id = ? ORDER BY r.id",
            (lecture_id,),
        )
        return [_reference(row) for row in rows]

    def history_signals(self, course: Course) -> list[HistorySignal]:
        """这门课已整合的讲里、通过校验的考点信号，按讲次和原顺序排列。"""
        rows = self.conn.execute(
            "SELECT k.key, s.id, s.lecture_id, l.seq, s.type, s.strength, s.quote, s.meaning, s.segment_ids"
            " FROM kp_signal ks"
            " JOIN signal s ON s.id = ks.signal_id JOIN knowledge_point k ON k.id = ks.kp_id"
            " JOIN lecture l ON l.id = s.lecture_id WHERE k.course_id = ? AND s.valid = 1 ORDER BY l.seq, s.id",
            (course.id,),
        )
        return [
            HistorySignal(row["key"], row["lecture_id"], row["seq"], row["type"], row["strength"], row["quote"],
                          json.loads(row["segment_ids"]), row["id"], row["meaning"])
            for row in rows
        ]

    def course_references(self, course: Course) -> list[Reference]:
        rows = []
        for lecture in self.library.lectures(course):
            rows.extend(self.references(lecture.id))
        return rows

    # ---------- 总复习 ----------

    def reviews(self, course_id: int) -> dict[str, StoredReview]:
        rows = self.conn.execute("SELECT * FROM review_chapter WHERE course_id = ?", (course_id,))
        return {
            row["chapter"]: StoredReview(row["chapter"], row["fingerprint"], row["result_json"], row["model"],
                                         row["created_at"])
            for row in rows
        }

    def save_review(
        self, course_id: int, chapter: str, fingerprint: str, result_json: str, profile: ModelProfile
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO review_chapter (course_id, chapter, fingerprint, result_json, model_profile, model,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (course_id, chapter, fingerprint, result_json, profile.name, profile.model, _now()),
        )
        self.conn.commit()

    def prune_reviews(self, course_id: int, chapters: Sequence[str]) -> None:
        """删掉已经不存在的章节的复习讲义。"""
        keep = set(chapters)
        removed = [(course_id, chapter) for chapter in self.reviews(course_id) if chapter not in keep]
        self.conn.executemany("DELETE FROM review_chapter WHERE course_id = ? AND chapter = ?", removed)
        self.conn.commit()


def _unique(names: Sequence[str], canonical: str) -> list[str]:
    """去掉空白、与规范名称相同（归一化后）以及重复的叫法，保持原顺序。"""
    seen = {normalize_zh(canonical)}
    result = []
    for name in names:
        value = name.strip()
        norm = normalize_zh(value)
        if value and norm not in seen:
            seen.add(norm)
            result.append(value)
    return result


def _reference(row) -> Reference:
    return Reference(
        lecture_id=row["lecture_id"],
        note_index=row["note_index"],
        key=row["key"],
        kp_name=row["name"],
        target_lecture_id=row["target_lecture_id"],
        target_seq=row["target_seq"],
        target_note_index=row["target_note_index"],
        relation=row["relation"],
        explanation=row["explanation"],
        segment_ids=json.loads(row["segment_ids"]),
    )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
