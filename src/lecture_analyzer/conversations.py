"""问答记录的存取：一门课里可以有多段对话，每段对话按顺序一问一答；回答连同证据存成 JSON。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .library import Course, Library, LibraryError


@dataclass(frozen=True)
class QATurn:
    id: int
    course_id: int
    lecture_id: int | None  # 限定某一讲时的讲；问整门课时为空
    thread: int  # 第几段对话
    question: str
    result: dict[str, Any]  # 回答、引用的证据、范围说明等（见 pipeline/qa.py）
    models: list[str]
    created_at: str


@dataclass(frozen=True)
class ThreadSummary:
    thread: int
    first_question: str
    turns: int
    updated_at: str


class QAStore:
    def __init__(self, library: Library) -> None:
        self.conn = library.conn

    def add(self, course: Course, lecture_id: int | None, thread: int, question: str, result: dict[str, Any],
            models: list[str]) -> int:
        cursor = self.conn.execute(
            "INSERT INTO qa (course_id, lecture_id, thread, question, result_json, models, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (course.id, lecture_id, thread, question, json.dumps(result, ensure_ascii=False),
             json.dumps(models, ensure_ascii=False), datetime.now().isoformat(timespec="seconds")),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def get(self, turn_id: int) -> QATurn:
        row = self.conn.execute("SELECT * FROM qa WHERE id = ?", (turn_id,)).fetchone()
        if row is None:
            raise LibraryError(f"没有编号为 {turn_id} 的问答")
        return _turn(row)

    def thread(self, course: Course, thread: int) -> list[QATurn]:
        rows = self.conn.execute("SELECT * FROM qa WHERE course_id = ? AND thread = ? ORDER BY id", (course.id, thread))
        return [_turn(row) for row in rows]

    def threads(self, course: Course) -> list[ThreadSummary]:
        """这门课的各段对话，最近更新的在前。"""
        rows = self.conn.execute(
            "SELECT thread, COUNT(*) AS turns, MAX(created_at) AS updated,"
            " (SELECT question FROM qa q2 WHERE q2.course_id = qa.course_id AND q2.thread = qa.thread ORDER BY id"
            " LIMIT 1) AS first FROM qa WHERE course_id = ? GROUP BY thread ORDER BY updated DESC, thread DESC",
            (course.id,),
        )
        return [ThreadSummary(row["thread"], row["first"], row["turns"], row["updated"]) for row in rows]

    def next_thread(self, course: Course) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(thread), 0) + 1 FROM qa WHERE course_id = ?", (course.id,))
        return int(row.fetchone()[0])

    def delete_thread(self, course: Course, thread: int) -> None:
        self.conn.execute("DELETE FROM qa WHERE course_id = ? AND thread = ?", (course.id, thread))
        self.conn.commit()


def _turn(row: Any) -> QATurn:
    return QATurn(
        id=row["id"], course_id=row["course_id"], lecture_id=row["lecture_id"], thread=row["thread"],
        question=row["question"], result=json.loads(row["result_json"]), models=json.loads(row["models"]),
        created_at=row["created_at"],
    )
