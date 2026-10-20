"""课程资料库的存取：课件、教材、往年真题、模拟题等（M3）。

一份资料是一个文件（几张图片合成的算一份）。原文件复制到 data/materials/<编号>/，拆出的页面图片也放在那里；
逐页识别的结果存在 doc_page（owner_type = "material"），拆出的题目或条目存在 material_item，
每讲对应哪些课件页存在 lecture_page（自动匹配或手动指定）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

from .library import Course, Library, LibraryError, PageRef
from .schemas import PageDigest

MATERIAL_KINDS = {
    "slides": "课件",
    "textbook": "教材",
    "past_exam": "往年真题",
    "mock_exam": "模拟题",
    "exercise": "习题",
    "review_outline": "复习提纲",
    "notice": "考试通知",
    "other": "其他资料",
}
PAGE_KINDS = ("slides", "textbook", "other")  # 按页匹配到各讲，整理笔记时参考
ITEM_KINDS = ("past_exam", "mock_exam", "exercise", "review_outline", "notice")  # 拆成题目或条目
SLIDE_KINDS = ("slides", "textbook")  # 导入后课程算作"已提供课件"
PAGE_SOURCES = ("auto", "manual")


@dataclass(frozen=True)
class Material:
    id: int
    course_id: int
    kind: str
    title: str
    path: str  # 副本相对数据目录的路径
    source: str  # 原始位置（压缩包里的文件写成「压缩包!内部路径」）
    position: int
    page_count: int
    status: dict[str, dict[str, Any]] = field(default_factory=dict)
    created_at: str = ""

    @property
    def kind_label(self) -> str:
        return MATERIAL_KINDS.get(self.kind, self.kind)

    def step_state(self, step: str) -> str:
        return self.status.get(step, {}).get("state", "")


@dataclass(frozen=True)
class NewItem:
    """拆题结果，写入前的形式。"""

    label: str  # 资料上的原题号，如「一、3」
    item_type: str
    content_md: str
    score: float | None
    answer_md: str
    pages: list[int]


@dataclass(frozen=True)
class MaterialItem(NewItem):
    id: int = 0
    material_id: int = 0
    idx: int = 0  # 资料内的顺序，从 1 开始
    kp_keys: tuple[int, ...] = ()


@dataclass(frozen=True)
class LecturePage:
    ref: PageRef
    source: str  # auto 自动匹配 / manual 手动指定


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class MaterialStore:
    def __init__(self, library: Library) -> None:
        self.library = library
        self.conn = library.conn

    # ---------- 资料 ----------

    def add(self, course: Course, kind: str, title: str, source: str) -> Material:
        """登记一份资料（文件由调用方复制后再用 set_file 记下）；课件类资料让课程变为"已提供课件"。"""
        if kind not in MATERIAL_KINDS:
            raise LibraryError(f"资料类型只能是 {', '.join(MATERIAL_KINDS)}")
        position = self.conn.execute(
            "SELECT COALESCE(MAX(position), 0) + 1 FROM material WHERE course_id = ?", (course.id,)
        ).fetchone()[0]
        cursor = self.conn.execute(
            "INSERT INTO material (course_id, kind, title, path, source, position, created_at)"
            " VALUES (?, ?, ?, '', ?, ?, ?)",
            (course.id, kind, " ".join(title.split()) or "未命名资料", source, position, _now()),
        )
        self.conn.commit()
        if kind in SLIDE_KINDS and course.slides != "provided":
            self.library.update_course(course.id, slides="provided")
        return self.get(cursor.lastrowid)

    def set_file(self, material_id: int, path: str, page_count: int) -> None:
        self.conn.execute("UPDATE material SET path = ?, page_count = ? WHERE id = ?", (path, page_count, material_id))
        self.conn.commit()

    def get(self, material_id: int) -> Material:
        row = self.conn.execute("SELECT * FROM material WHERE id = ?", (material_id,)).fetchone()
        if row is None:
            raise LibraryError(f"没有编号为 {material_id} 的资料，可用 `lec material list 课程` 查看")
        return _material(row)

    def list(self, course: Course, kinds: Sequence[str] | None = None) -> list[Material]:
        rows = self.conn.execute("SELECT * FROM material WHERE course_id = ? ORDER BY position, id", (course.id,))
        materials = [_material(row) for row in rows]
        return [m for m in materials if kinds is None or m.kind in kinds]

    def remove(self, material_id: int) -> list[int]:
        """删除资料及其页面、题目和各讲的对应关系；返回因此需要重新匹配课件的讲。

        删掉最后一份课件类资料后，课程回到"还没选择是否提供课件"。
        """
        material = self.get(material_id)
        affected = self.lectures_using(material_id)
        self.conn.execute("DELETE FROM doc_page WHERE owner_type = 'material' AND owner_id = ?", (material_id,))
        self.conn.execute("DELETE FROM material WHERE id = ?", (material_id,))
        self.conn.commit()
        course = self.library.course_by_id(material.course_id)
        if material.kind in SLIDE_KINDS and not self.list(course, SLIDE_KINDS) and course.slides == "provided":
            self.library.update_course(course.id, slides="unset")
        for lecture_id in affected:
            self.library.reset_steps(lecture_id, ["slides"])
        return affected

    def set_step(self, material_id: int, step: str, state: str, detail: str = "", **extra: Any) -> None:
        status = self.get(material_id).status
        status[step] = {"state": state, "at": _now(), "detail": detail, **extra}
        self._save_status(material_id, status)

    def reset_steps(self, material_id: int, steps: Iterable[str]) -> None:
        status = self.get(material_id).status
        for step in steps:
            status.pop(step, None)
        self._save_status(material_id, status)

    def _save_status(self, material_id: int, status: dict[str, dict[str, Any]]) -> None:
        payload = json.dumps(status, ensure_ascii=False)
        self.conn.execute("UPDATE material SET status = ? WHERE id = ?", (payload, material_id))
        self.conn.commit()

    # ---------- 页面 ----------

    def save_pages(self, material_id: int, digests: Iterable[PageDigest]) -> None:
        """写入（或覆盖）若干页的识别结果；逐批写入，中断后已识别的页不用重做。"""
        self.conn.executemany(
            "INSERT OR REPLACE INTO doc_page (owner_type, owner_id, page_no, title, content_md, terms)"
            " VALUES ('material', ?, ?, ?, ?, ?)",
            [(material_id, d.page_no, d.title, d.content_md, json.dumps(d.terms, ensure_ascii=False)) for d in digests],
        )
        self.conn.commit()

    def pages(self, material_id: int) -> list[PageDigest]:
        return self.library.pages("material", material_id)

    def clear_pages(self, material_id: int) -> None:
        self.conn.execute("DELETE FROM doc_page WHERE owner_type = 'material' AND owner_id = ?", (material_id,))
        self.conn.commit()

    def page_labels(self, refs: Sequence[PageRef]) -> list[str]:
        """各页的出处：「资料名 第 N 页」；资料已经删除时注明。"""
        titles = {row["id"]: row["title"] for row in self.conn.execute("SELECT id, title FROM material")}
        return [f"{titles.get(ref.material_id, '（已删除的资料）')} 第 {ref.page_no} 页" for ref in refs]

    def page_digests(self, refs: Sequence[PageRef]) -> dict[PageRef, PageDigest]:
        """按页取识别结果；没识别的页不在结果里。"""
        wanted = set(refs)
        found: dict[PageRef, PageDigest] = {}
        for material_id in sorted({ref.material_id for ref in wanted}):
            for digest in self.pages(material_id):
                ref = PageRef(material_id, digest.page_no)
                if ref in wanted:
                    found[ref] = digest
        return found

    # ---------- 题目与条目 ----------

    def replace_items(self, material_id: int, items: Sequence[NewItem]) -> None:
        self.conn.execute("DELETE FROM material_item WHERE material_id = ?", (material_id,))
        self.conn.executemany(
            "INSERT INTO material_item (material_id, idx, label, item_type, content_md, score, answer_md, pages)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (material_id, idx, item.label, item.item_type, item.content_md, item.score, item.answer_md,
                 json.dumps(item.pages))
                for idx, item in enumerate(items, start=1)
            ],
        )
        self.conn.commit()

    def items(self, material_id: int) -> list[MaterialItem]:
        keys: dict[int, list[int]] = {}
        for row in self.conn.execute(
            "SELECT k.item_id, k.kp_key FROM item_kp k JOIN material_item i ON i.id = k.item_id"
            " WHERE i.material_id = ? ORDER BY k.kp_key",
            (material_id,),
        ):
            keys.setdefault(row["item_id"], []).append(row["kp_key"])
        rows = self.conn.execute("SELECT * FROM material_item WHERE material_id = ? ORDER BY idx", (material_id,))
        return [_item(row, keys.get(row["id"], [])) for row in rows]

    def set_item_kps(self, material_id: int, mapping: dict[int, Sequence[int]]) -> None:
        """mapping：{题目编号 item.id: [知识点 K 编号]}，覆盖这份资料原有的对应关系。"""
        self.conn.execute(
            "DELETE FROM item_kp WHERE item_id IN (SELECT id FROM material_item WHERE material_id = ?)", (material_id,)
        )
        self.conn.executemany(
            "INSERT OR IGNORE INTO item_kp (item_id, kp_key) VALUES (?, ?)",
            [(item_id, key) for item_id, keys in mapping.items() for key in keys],
        )
        self.conn.commit()

    # ---------- 每讲对应的课件页 ----------

    def lecture_pages(self, lecture_id: int) -> list[LecturePage]:
        rows = self.conn.execute(
            "SELECT material_id, page_no, source FROM lecture_page WHERE lecture_id = ? ORDER BY material_id, page_no",
            (lecture_id,),
        )
        return [LecturePage(PageRef(row["material_id"], row["page_no"]), row["source"]) for row in rows]

    def set_lecture_pages(self, lecture_id: int, refs: Iterable[PageRef], source: str) -> None:
        """覆盖这一讲对应的全部课件页。"""
        if source not in PAGE_SOURCES:
            raise LibraryError(f"页面来源只能是 {', '.join(PAGE_SOURCES)}")
        self.conn.execute("DELETE FROM lecture_page WHERE lecture_id = ?", (lecture_id,))
        self.conn.executemany(
            "INSERT OR IGNORE INTO lecture_page (lecture_id, material_id, page_no, source) VALUES (?, ?, ?, ?)",
            [(lecture_id, ref.material_id, ref.page_no, source) for ref in refs],
        )
        self.conn.commit()

    def lectures_using(self, material_id: int) -> list[int]:
        rows = self.conn.execute("SELECT DISTINCT lecture_id FROM lecture_page WHERE material_id = ?", (material_id,))
        return [row["lecture_id"] for row in rows]


def _material(row: Any) -> Material:
    return Material(
        id=row["id"], course_id=row["course_id"], kind=row["kind"], title=row["title"], path=row["path"],
        source=row["source"], position=row["position"], page_count=row["page_count"],
        status=json.loads(row["status"]), created_at=row["created_at"],
    )


def _item(row: Any, keys: list[int]) -> MaterialItem:
    return MaterialItem(
        label=row["label"], item_type=row["item_type"], content_md=row["content_md"], score=row["score"],
        answer_md=row["answer_md"], pages=json.loads(row["pages"]), id=row["id"], material_id=row["material_id"],
        idx=row["idx"], kp_keys=tuple(keys),
    )
