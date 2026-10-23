"""书库（SQLite）：一门课是一本书，一讲是书里的一章；一讲可以由多个文件（节）组成，处理时合并成一讲。

这里保存课程设置、讲的信息与处理进度、转写、课件页、笔记、考点信号、标签、复习进度和模型调用记录；
课程记忆（L2）和总复习的读写在 knowledge.py。表结构与迁移在 library_schema.py。
每次操作用独立连接即可（单机、低并发）；网页服务按请求各开一个连接。
后台任务在子进程里执行，和界面同时读写书库：书库用 WAL 模式（读写互不阻塞），写入遇到对方正在写时等待；
因此不要在没提交的写操作期间调用模型（会让其他进程等待超时）。
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from .asr.base import Segment
from .course import CourseSettings, SettingsError, build_settings
from .library_schema import SCHEMA_VERSION, migrate
from .llm import ModelProfile, Usage
from .schemas import LectureNotes, PageDigest, Signal

LECTURE_KINDS = ("normal", "exercise", "review")  # 普通课 / 习题课 / 复习课
MAX_INSTRUCTIONS_CHARS = 4000
BUSY_TIMEOUT_SECONDS = 30  # 其他进程正在写书库时，最多等这么久
MAX_TAG_CHARS = 20
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

# 允许修改的字段各用一条固定语句，不拼接 SQL
_UPDATE_LECTURE_SQL = {
    "title": "UPDATE lecture SET title = ? WHERE id = ?",
    "duration_ms": "UPDATE lecture SET duration_ms = ? WHERE id = ?",
    "date": "UPDATE lecture SET date = ? WHERE id = ?",
    "kind": "UPDATE lecture SET kind = ? WHERE id = ?",
}
_UPDATE_COURSE_SQL = {
    "subject": "UPDATE course SET subject = ? WHERE id = ?",
    "teaching": "UPDATE course SET teaching = ? WHERE id = ?",
    "interpreted_to": "UPDATE course SET interpreted_to = ? WHERE id = ?",
    "note_language": "UPDATE course SET note_language = ? WHERE id = ?",
    "asr": "UPDATE course SET asr = ? WHERE id = ?",
    "instructions": "UPDATE course SET instructions = ? WHERE id = ?",
    "slides": "UPDATE course SET slides = ? WHERE id = ?",
    "exam_plan": "UPDATE course SET exam_plan = ? WHERE id = ?",
}
_SETTINGS_FIELDS = ("subject", "teaching", "interpreted_to", "note_language")
SLIDES_LABELS = {"unset": "未选择", "none": "不提供", "provided": "已提供"}  # 分析前必须先选择是否提供课件
SLIDES_CHOICES = tuple(SLIDES_LABELS)
# 考试方案：整理笔记前必须已确认或已跳过（方案关卡，见 pipeline/exam_plan.py）
EXAM_PLAN_LABELS = {"unset": "未设置", "draft": "待确认", "confirmed": "已确认", "skipped": "已跳过"}
_PARTS_OF_LECTURE = "SELECT * FROM lecture_part WHERE lecture_id = ? ORDER BY part"
_PARTS_OF_COURSE = (
    "SELECT p.* FROM lecture_part p JOIN lecture l ON l.id = p.lecture_id WHERE l.course_id = ?"
    " ORDER BY p.lecture_id, p.part"
)
_ALL_PARTS = "SELECT * FROM lecture_part ORDER BY lecture_id, part"


class LibraryError(LookupError):
    """书库里找不到对象、输入不合法，或数据不一致。"""


@dataclass(frozen=True)
class Course:
    id: int
    name: str
    settings: CourseSettings
    asr: str | None  # 课程专用的转写方案；为空时用 config.toml 的 [asr]
    instructions: str = ""  # 用户对这门课的补充要求（课程提示词）
    position: int = 0  # 书架上的自定义顺序
    slides: str = "unset"  # 是否提供课件：unset 还没选择 / none 不提供 / provided 已提供
    exam_plan: str = "unset"  # 考试方案：unset 未设置 / draft 待确认 / confirmed 已确认 / skipped 已跳过


@dataclass(frozen=True)
class LecturePart:
    part: int  # 第几节，从 1 开始
    media_path: str
    duration_ms: int | None  # 抽音频后填写


@dataclass(frozen=True)
class Lecture:
    id: int
    course_id: int
    seq: int
    title: str
    date: str
    kind: str
    duration_ms: int | None
    created_at: str = ""
    parts: tuple[LecturePart, ...] = ()
    status: dict[str, dict[str, str]] = field(default_factory=dict)

    def step_state(self, step: str) -> str:
        return self.status.get(step, {}).get("state", "")

    def part_offsets(self) -> list[int]:
        """各节在整讲时间轴上的起点（毫秒）；时长未知的节按 0 计（只在抽音频之前出现）。"""
        offsets, total = [], 0
        for part in self.parts:
            offsets.append(total)
            total += part.duration_ms or 0
        return offsets

    @property
    def last_activity(self) -> str:
        """最近一次处理的时间（没处理过时为添加时间），用于"最近更新"排序。"""
        return max([info.get("at", "") for info in self.status.values()] + [self.created_at])


@dataclass(frozen=True, order=True)
class PageRef:
    """课程资料里的一页。"""

    material_id: int
    page_no: int  # 从 1 开始


@dataclass(frozen=True)
class CheckedSignal:
    """通过原话校验后的考点信号；segment_ids 可能已校正为原话实际所在的位置。"""

    signal: Signal
    valid: bool
    method: str  # window / fuzzy / global / slide；未通过时为空
    segment_ids: list[int]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clean_tag(tag: str) -> str:
    value = tag.strip()
    if not value or len(value) > MAX_TAG_CHARS or _CONTROL_CHARS.search(value):
        raise LibraryError(f"标签要求 1–{MAX_TAG_CHARS} 个字，且不含控制字符")
    return value


def clean_instructions(text: str) -> str:
    value = text.strip()
    if len(value) > MAX_INSTRUCTIONS_CHARS:
        raise LibraryError(f"课程提示词最多 {MAX_INSTRUCTIONS_CHARS} 字，当前 {len(value)} 字")
    return value


class Library:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_SECONDS)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        # 从旧版本升级前先备份（lecture.db.bak-v1、bak-v2……），迁移出问题时可以恢复
        migrate(self.conn, {v: str(path.with_name(f"{path.name}.bak-v{v}")) for v in range(1, SCHEMA_VERSION)})
        with suppress(sqlite3.OperationalError):  # 设置一次后记在书库文件里；其他进程正占用时切换不了，下次再试
            self.conn.execute("PRAGMA journal_mode = WAL")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Library:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------- 课程（书） ----------

    def add_course(self, name: str, settings: CourseSettings, asr: str | None = None, instructions: str = "") -> Course:
        position = self.conn.execute("SELECT COALESCE(MAX(position), 0) + 1 FROM course").fetchone()[0]
        try:
            self.conn.execute(
                "INSERT INTO course (name, subject, teaching, interpreted_to, note_language, asr, instructions,"
                " position, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, settings.subject, settings.teaching, settings.interpreted_to, settings.note_language, asr,
                 clean_instructions(instructions), position, _now()),
            )
        except sqlite3.IntegrityError:
            raise LibraryError(f"课程「{name}」已存在") from None
        self.conn.commit()
        return self.course(name)

    def course(self, name: str) -> Course:
        row = self.conn.execute("SELECT * FROM course WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise LibraryError(f"没有名为「{name}」的课程，可用 `lec course list` 查看")
        return _course(row)

    def course_by_id(self, course_id: int) -> Course:
        row = self.conn.execute("SELECT * FROM course WHERE id = ?", (course_id,)).fetchone()
        if row is None:
            raise LibraryError(f"没有编号为 {course_id} 的课程")
        return _course(row)

    def courses(self) -> list[Course]:
        return [_course(row) for row in self.conn.execute("SELECT * FROM course ORDER BY position, id")]

    def update_course(self, course_id: int, **fields: Any) -> Course:
        unknown = set(fields) - set(_UPDATE_COURSE_SQL)
        if unknown:
            raise LibraryError(f"不能直接修改这些字段：{', '.join(sorted(unknown))}")
        if "instructions" in fields:
            fields["instructions"] = clean_instructions(fields["instructions"])
        if fields.get("slides", "unset") not in SLIDES_CHOICES:
            raise LibraryError(f"课件选择只能是 {', '.join(SLIDES_CHOICES)}")
        if fields.get("exam_plan", "unset") not in EXAM_PLAN_LABELS:
            raise LibraryError(f"考试方案状态只能是 {', '.join(EXAM_PLAN_LABELS)}")
        if set(fields) & set(_SETTINGS_FIELDS):  # 语言等设置合在一起校验，并换算成规范写法
            current = self.course_by_id(course_id).settings.model_dump()
            try:
                settings = build_settings(**{**current, **{k: v for k, v in fields.items() if k in _SETTINGS_FIELDS}})
            except SettingsError as exc:
                raise LibraryError(f"课程设置不正确：{exc}") from None
            fields.update({k: getattr(settings, k) for k in _SETTINGS_FIELDS if k in fields or k == "interpreted_to"})
        for name, value in fields.items():
            self.conn.execute(_UPDATE_COURSE_SQL[name], (value, course_id))
        self.conn.commit()
        return self.course_by_id(course_id)

    def move_course(self, course_id: int, offset: int) -> None:
        """在书架上前移（offset < 0）或后移一格；已在两端时不动。"""
        order = [course.id for course in self.courses()]
        if course_id not in order:
            raise LibraryError(f"没有编号为 {course_id} 的课程")
        index = order.index(course_id)
        target = index + (1 if offset > 0 else -1)
        if not 0 <= target < len(order):
            return
        order[index], order[target] = order[target], order[index]
        positions = [(position, cid) for position, cid in enumerate(order, start=1)]
        self.conn.executemany("UPDATE course SET position = ? WHERE id = ?", positions)
        self.conn.commit()

    # ---------- 讲（章）与节 ----------

    def add_lecture(
        self,
        course: Course,
        seq: int,
        media: Sequence[Path],
        *,
        title: str = "",
        date: str = "",
        kind: str = "normal",
    ) -> Lecture:
        """添加一讲；media 按顺序是第 1、2… 节。"""
        if not media:
            raise LibraryError("一讲至少要有一个录音或录像文件")
        if kind not in LECTURE_KINDS:
            raise LibraryError(f"讲的类型只能是 {', '.join(LECTURE_KINDS)}")
        try:
            cursor = self.conn.execute(
                "INSERT INTO lecture (course_id, seq, title, date, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (course.id, seq, title, date, kind, _now()),
            )
        except sqlite3.IntegrityError:
            raise LibraryError(f"课程「{course.name}」已有第 {seq} 讲") from None
        self.conn.executemany(
            "INSERT INTO lecture_part (lecture_id, part, media_path) VALUES (?, ?, ?)",
            [(cursor.lastrowid, i, str(path)) for i, path in enumerate(media, start=1)],
        )
        self.conn.commit()
        return self.lecture(course, seq)

    def add_part(self, lecture_id: int, media_path: Path) -> Lecture:
        """给一讲追加一节；抽音频及其后续步骤需要重跑。"""
        lecture = self.lecture_by_id(lecture_id)
        self.conn.execute(
            "INSERT INTO lecture_part (lecture_id, part, media_path) VALUES (?, ?, ?)",
            (lecture_id, len(lecture.parts) + 1, str(media_path)),
        )
        self.conn.commit()
        self.reset_steps(lecture_id, ["media"])
        return self.lecture_by_id(lecture_id)

    def set_part_duration(self, lecture_id: int, part: int, duration_ms: int) -> None:
        self.conn.execute(
            "UPDATE lecture_part SET duration_ms = ? WHERE lecture_id = ? AND part = ?", (duration_ms, lecture_id, part)
        )
        self.conn.commit()

    def lecture(self, course: Course, seq: int) -> Lecture:
        row = self.conn.execute("SELECT * FROM lecture WHERE course_id = ? AND seq = ?", (course.id, seq)).fetchone()
        if row is None:
            raise LibraryError(f"课程「{course.name}」没有第 {seq} 讲，可用 `lec lecture list` 查看")
        return _lecture(row, self._parts(_PARTS_OF_LECTURE, row["id"]))

    def lecture_by_id(self, lecture_id: int) -> Lecture:
        row = self.conn.execute("SELECT * FROM lecture WHERE id = ?", (lecture_id,)).fetchone()
        if row is None:
            raise LibraryError(f"没有编号为 {lecture_id} 的讲")
        return _lecture(row, self._parts(_PARTS_OF_LECTURE, lecture_id))

    def lectures(self, course: Course | None = None) -> list[Lecture]:
        """按课程、讲次排序。"""
        if course is None:
            rows = self.conn.execute("SELECT * FROM lecture ORDER BY course_id, seq").fetchall()
            parts = self._parts(_ALL_PARTS)
        else:
            rows = self.conn.execute("SELECT * FROM lecture WHERE course_id = ? ORDER BY seq", (course.id,)).fetchall()
            parts = self._parts(_PARTS_OF_COURSE, course.id)
        return [_lecture(row, parts) for row in rows]

    def _parts(self, query: str, *params: Any) -> dict[int, list[LecturePart]]:
        grouped: dict[int, list[LecturePart]] = {}
        for row in self.conn.execute(query, params):
            grouped.setdefault(row["lecture_id"], []).append(
                LecturePart(row["part"], row["media_path"], row["duration_ms"])
            )
        return grouped

    def update_lecture(self, lecture_id: int, **fields: Any) -> None:
        unknown = set(fields) - set(_UPDATE_LECTURE_SQL)
        if unknown:
            raise LibraryError(f"不能直接修改这些字段：{', '.join(sorted(unknown))}")
        if "kind" in fields and fields["kind"] not in LECTURE_KINDS:
            raise LibraryError(f"讲的类型只能是 {', '.join(LECTURE_KINDS)}")
        for name, value in fields.items():
            self.conn.execute(_UPDATE_LECTURE_SQL[name], (value, lecture_id))
        self.conn.commit()

    def move_lecture(self, lecture_id: int, course: Course, seq: int) -> None:
        """把一讲移到另一门课（改正放错的课程）；后续的重跑和课程记忆由调用方处理。"""
        if seq < 1:
            raise LibraryError("讲次必须是正整数")
        try:
            self.conn.execute("UPDATE lecture SET course_id = ?, seq = ? WHERE id = ?", (course.id, seq, lecture_id))
        except sqlite3.IntegrityError:
            raise LibraryError(f"课程「{course.name}」已有第 {seq} 讲") from None
        self.conn.commit()

    def delete_lecture(self, lecture_id: int) -> None:
        """删除一讲在书库里的全部内容（转写、笔记、信号、各节记录……，由外键级联删除）；原始音视频不动。
        课程记忆和导出文件由调用方处理（见 pipeline/organize.delete_lecture）。"""
        self.lecture_by_id(lecture_id)  # 不存在时报错
        self.conn.execute("DELETE FROM doc_page WHERE owner_type = 'lecture' AND owner_id = ?", (lecture_id,))
        self.conn.execute("DELETE FROM lecture WHERE id = ?", (lecture_id,))
        self.conn.commit()

    def delete_course(self, course_id: int) -> None:
        """删除一门课及其全部讲、课程记忆、资料、报告、任务和问答（外键级联）；文件由调用方清理。"""
        self.course_by_id(course_id)
        owners = [("lecture", lid) for (lid,) in self.conn.execute("SELECT id FROM lecture WHERE course_id = ?",
                                                                    (course_id,))]
        owners += [("material", mid) for (mid,) in self.conn.execute("SELECT id FROM material WHERE course_id = ?",
                                                                      (course_id,))]
        self.conn.executemany("DELETE FROM doc_page WHERE owner_type = ? AND owner_id = ?", owners)
        self.conn.execute("DELETE FROM course WHERE id = ?", (course_id,))
        self.conn.commit()

    def set_step(self, lecture_id: int, step: str, state: str, detail: str = "") -> None:
        status = self.lecture_by_id(lecture_id).status
        status[step] = {"state": state, "at": _now(), "detail": detail}
        self._save_status(lecture_id, status)

    def reset_steps(self, lecture_id: int, steps: Iterable[str]) -> None:
        """清除这些步骤的进度，下次运行时重跑（依赖它们的步骤也会跟着重跑）。"""
        status = self.lecture_by_id(lecture_id).status
        for step in steps:
            status.pop(step, None)
        self._save_status(lecture_id, status)

    def _save_status(self, lecture_id: int, status: dict[str, dict[str, str]]) -> None:
        payload = json.dumps(status, ensure_ascii=False)
        self.conn.execute("UPDATE lecture SET status = ? WHERE id = ?", (payload, lecture_id))
        self.conn.commit()

    # ---------- 标签与复习进度 ----------

    def add_tag(self, lecture_id: int, tag: str) -> None:
        self.lecture_by_id(lecture_id)  # 不存在时报错
        value = clean_tag(tag)
        self.conn.execute("INSERT OR IGNORE INTO lecture_tag (lecture_id, tag) VALUES (?, ?)", (lecture_id, value))
        self.conn.commit()

    def remove_tag(self, lecture_id: int, tag: str) -> None:
        self.conn.execute("DELETE FROM lecture_tag WHERE lecture_id = ? AND tag = ?", (lecture_id, tag.strip()))
        self.conn.commit()

    def tags(self, lecture_id: int) -> list[str]:
        rows = self.conn.execute("SELECT tag FROM lecture_tag WHERE lecture_id = ? ORDER BY tag", (lecture_id,))
        return [row["tag"] for row in rows]

    def course_tags(self, course: Course) -> dict[int, list[str]]:
        """这门课各讲的标签：{讲的编号: [标签]}。"""
        rows = self.conn.execute(
            "SELECT t.lecture_id, t.tag FROM lecture_tag t JOIN lecture l ON l.id = t.lecture_id"
            " WHERE l.course_id = ? ORDER BY t.tag",
            (course.id,),
        )
        grouped: dict[int, list[str]] = {}
        for row in rows:
            grouped.setdefault(row["lecture_id"], []).append(row["tag"])
        return grouped

    def reviewed_names(self, course_id: int) -> set[str]:
        rows = self.conn.execute("SELECT name_norm FROM kp_review WHERE course_id = ?", (course_id,))
        return {row["name_norm"] for row in rows}

    def set_reviewed(self, course_id: int, name_norm: str, reviewed: bool) -> None:
        """按归一化的知识点名称记录复习进度，课程记忆重建后名称不变的进度仍然有效。"""
        if reviewed:
            self.conn.execute(
                "INSERT OR REPLACE INTO kp_review (course_id, name_norm, reviewed_at) VALUES (?, ?, ?)",
                (course_id, name_norm, _now()),
            )
        else:
            self.conn.execute("DELETE FROM kp_review WHERE course_id = ? AND name_norm = ?", (course_id, name_norm))
        self.conn.commit()

    # ---------- 内容 ----------

    def replace_segments(self, lecture_id: int, segments: Sequence[Segment]) -> None:
        self.conn.execute("DELETE FROM segment WHERE lecture_id = ?", (lecture_id,))
        self.conn.executemany(
            "INSERT INTO segment (lecture_id, idx, start_ms, end_ms, text, lang) VALUES (?, ?, ?, ?, ?, ?)",
            [(lecture_id, i, s.start_ms, s.end_ms, s.text, s.lang) for i, s in enumerate(segments, start=1)],
        )
        self.conn.commit()

    def segments(self, lecture_id: int) -> list[Segment]:
        rows = self.conn.execute(
            "SELECT start_ms, end_ms, text, lang FROM segment WHERE lecture_id = ? ORDER BY idx", (lecture_id,)
        )
        return [Segment(row["start_ms"], row["end_ms"], row["text"], row["lang"]) for row in rows]

    def replace_pages(self, owner_type: str, owner_id: int, digests: Iterable[PageDigest]) -> None:
        self.conn.execute("DELETE FROM doc_page WHERE owner_type = ? AND owner_id = ?", (owner_type, owner_id))
        self.conn.executemany(
            "INSERT INTO doc_page (owner_type, owner_id, page_no, title, content_md, terms) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (owner_type, owner_id, d.page_no, d.title, d.content_md, json.dumps(d.terms, ensure_ascii=False))
                for d in digests
            ],
        )
        self.conn.commit()

    def pages(self, owner_type: str, owner_id: int) -> list[PageDigest]:
        rows = self.conn.execute(
            "SELECT * FROM doc_page WHERE owner_type = ? AND owner_id = ? ORDER BY page_no", (owner_type, owner_id)
        )
        return [
            PageDigest(page_no=r["page_no"], title=r["title"], content_md=r["content_md"], terms=json.loads(r["terms"]))
            for r in rows
        ]

    def save_notes(
        self, lecture_id: int, notes: LectureNotes, model_profile: str, model: str,
        pages: Sequence[PageRef] = (),
    ) -> None:
        """pages：整理笔记时给模型看的课件页，笔记里的页码 P1、P2… 依次对应其中各页。"""
        self.conn.execute(
            "INSERT OR REPLACE INTO lecture_note (lecture_id, notes_json, model_profile, model, created_at, pages)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (lecture_id, notes.model_dump_json(), model_profile, model, _now(),
             json.dumps([[ref.material_id, ref.page_no] for ref in pages])),
        )
        self.conn.commit()

    def notes(self, lecture_id: int) -> LectureNotes | None:
        row = self.conn.execute("SELECT notes_json FROM lecture_note WHERE lecture_id = ?", (lecture_id,)).fetchone()
        return LectureNotes.model_validate_json(row["notes_json"]) if row else None

    def note_pages(self, lecture_id: int) -> list[PageRef]:
        """笔记里的页码 P1、P2… 对应的课件页（整理笔记时的快照，之后重新匹配课件也不会错位）。"""
        row = self.conn.execute("SELECT pages FROM lecture_note WHERE lecture_id = ?", (lecture_id,)).fetchone()
        return [PageRef(material_id, page_no) for material_id, page_no in json.loads(row["pages"])] if row else []

    def notes_model(self, lecture_id: int) -> tuple[str, str] | None:
        row = self.conn.execute(
            "SELECT model_profile, model FROM lecture_note WHERE lecture_id = ?", (lecture_id,)
        ).fetchone()
        return (row["model_profile"], row["model"]) if row else None

    def replace_signals(self, lecture_id: int, checked: Sequence[CheckedSignal]) -> None:
        self.conn.execute("DELETE FROM signal WHERE lecture_id = ?", (lecture_id,))
        self.conn.executemany(
            "INSERT INTO signal (lecture_id, type, knowledge_point, strength, quote, meaning, segment_ids, slide_pages,"
            " valid, match_method) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    lecture_id, c.signal.type, c.signal.knowledge_point, c.signal.strength, c.signal.quote,
                    c.signal.meaning, json.dumps(c.segment_ids), json.dumps(c.signal.slide_pages), int(c.valid),
                    c.method,
                )
                for c in checked
            ],
        )
        self.conn.commit()

    def signals(self, lecture_id: int) -> list[CheckedSignal]:
        rows = self.conn.execute("SELECT * FROM signal WHERE lecture_id = ? ORDER BY id", (lecture_id,))
        return [_checked_signal(row) for row in rows]

    def signal_ids(self, lecture_id: int) -> list[int]:
        """各条信号在表里的编号，顺序与 signals() 一致。"""
        rows = self.conn.execute("SELECT id FROM signal WHERE lecture_id = ? ORDER BY id", (lecture_id,))
        return [row["id"] for row in rows]

    # ---------- 模型调用记录 ----------

    def log_call(
        self, lecture_id: int | None, step: str, profile: ModelProfile, usage: Usage, course_id: int | None = None
    ) -> None:
        """记录一次模型调用的用量，费用按模型档案里的单价估算；按课程的调用（如总复习）记 course_id。"""
        self.conn.execute(
            "INSERT INTO llm_call (lecture_id, course_id, step, model_profile, model, input_tokens, output_tokens,"
            " cache_read_tokens, cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (lecture_id, course_id, step, profile.name, profile.model, usage.input_tokens, usage.output_tokens,
             usage.cache_read_tokens, profile.estimate_cost(usage), _now()),
        )
        self.conn.commit()

    def cost_summary(self, course: Course | None = None) -> list[dict[str, Any]]:
        """按步骤和模型汇总用量与估算费用。"""
        query = (
            "SELECT step, model, COUNT(*) AS calls, SUM(input_tokens) AS input_tokens,"
            " SUM(output_tokens) AS output_tokens, SUM(cost_usd) AS cost_usd FROM llm_call"
        )
        params: tuple[Any, ...] = ()
        if course is not None:
            query += " WHERE lecture_id IN (SELECT id FROM lecture WHERE course_id = ?) OR course_id = ?"
            params = (course.id, course.id)
        query += " GROUP BY step, model ORDER BY step, model"
        return [dict(row) for row in self.conn.execute(query, params)]


def _course(row: sqlite3.Row) -> Course:
    settings = CourseSettings(subject=row["subject"], teaching=row["teaching"],
                              interpreted_to=row["interpreted_to"], note_language=row["note_language"])
    return Course(
        id=row["id"], name=row["name"], settings=settings, asr=row["asr"],
        instructions=row["instructions"], position=row["position"], slides=row["slides"],
        exam_plan=row["exam_plan"],
    )


def _lecture(row: sqlite3.Row, parts: dict[int, list[LecturePart]]) -> Lecture:
    return Lecture(
        id=row["id"],
        course_id=row["course_id"],
        seq=row["seq"],
        title=row["title"],
        date=row["date"],
        kind=row["kind"],
        duration_ms=row["duration_ms"],
        created_at=row["created_at"],
        parts=tuple(parts.get(row["id"], ())),
        status=json.loads(row["status"]),
    )


def _checked_signal(row: sqlite3.Row) -> CheckedSignal:
    signal = Signal(
        type=row["type"],
        knowledge_point=row["knowledge_point"],
        strength=row["strength"],
        quote=row["quote"],
        meaning=row["meaning"],
        segment_ids=json.loads(row["segment_ids"]),
        slide_pages=json.loads(row["slide_pages"]),
    )
    return CheckedSignal(signal, bool(row["valid"]), row["match_method"], json.loads(row["segment_ids"]))
