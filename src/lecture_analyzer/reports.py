"""考前报告的存取：报告内容是代码拼装好的 JSON（证据已按编号填入），网页和 Markdown 导出都用它。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .library import Course, Library, LibraryError


@dataclass(frozen=True)
class StoredReport:
    id: int
    course_id: int
    title: str
    params: dict[str, Any]
    result: dict[str, Any]
    models: list[str]
    created_at: str


class ReportStore:
    def __init__(self, library: Library) -> None:
        self.conn = library.conn

    def add(self, course: Course, title: str, params: dict[str, Any], result: dict[str, Any], models: list[str]) -> int:
        cursor = self.conn.execute(
            "INSERT INTO report (course_id, title, params, result_json, models, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (course.id, title, json.dumps(params, ensure_ascii=False), json.dumps(result, ensure_ascii=False),
             json.dumps(models, ensure_ascii=False), datetime.now().isoformat(timespec="seconds")),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def get(self, report_id: int) -> StoredReport:
        row = self.conn.execute("SELECT * FROM report WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            raise LibraryError(f"没有编号为 {report_id} 的考前报告")
        return _report(row)

    def list(self, course: Course) -> list[StoredReport]:
        rows = self.conn.execute("SELECT * FROM report WHERE course_id = ? ORDER BY id DESC", (course.id,))
        return [_report(row) for row in rows]

    def remove(self, report_id: int) -> None:
        self.conn.execute("DELETE FROM report WHERE id = ?", (report_id,))
        self.conn.commit()


def _report(row: Any) -> StoredReport:
    return StoredReport(
        id=row["id"], course_id=row["course_id"], title=row["title"], params=json.loads(row["params"]),
        result=json.loads(row["result_json"]), models=json.loads(row["models"]), created_at=row["created_at"],
    )
