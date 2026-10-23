"""考试方案：这门课怎么考（考核方式、成绩构成、题型分值、答题要求、考试范围）。

- 先于知识点分析：由用户提供的考试说明、考试通知 / 复习提纲资料和来源讲（默认第 1 讲）的转写分析得出
  （pipeline/exam_plan.py），用户在课程页核对、修改并确认后，整理笔记、总复习、打分、考前报告和问答都以它为依据；
- 每一条都带出处：用户说明、资料条目，或课堂原话（与转写核对过，核对不上的标出来让用户判断）；
- 题型的合计分值和占比由代码计算，不让模型算；题型尽量归到资料题型（ITEM_TYPES）的写法，方便和真题对照；
- 只有已确认的方案进入后续各步骤的提示词；没有方案或已跳过时，提示词与没有这个功能时完全一样。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .library import Course, Library, LibraryError

PlanSource = Literal["user", "material", "lecture"]
SOURCE_LABELS = {"user": "考试说明", "material": "资料", "lecture": "课堂"}
SCOPE_LABELS = {"include": "要考", "exclude": "不考", "focus": "重点"}

# 题型：与资料题型（pipeline/materials.py 的 ITEM_TYPES）里的题型部分一致（测试保证），
# 这里单独列出是为了不让书库层依赖流水线模块
QUESTION_TYPES = {
    "choice": "选择题", "fill": "填空题", "judge": "判断题", "term": "名词解释", "short": "简答题",
    "calculation": "计算题", "essay": "论述题", "case": "病例分析", "other": "其他",
}
# 常见写法 → 题型（按顺序匹配，先匹配更具体的）
_TYPE_PATTERNS: list[tuple[str, str]] = [
    ("case", r"病例|案例|病案|临床分析|情景"),
    ("term", r"名词解释|名解|概念解释|术语解释"),
    ("essay", r"论述|分析论述|综述"),
    ("short", r"简答|问答|简述"),
    ("calculation", r"计算"),
    ("fill", r"填空"),
    ("judge", r"判断|是非|正误"),
    ("choice", r"选择|单选|多选|[A-EＡ-Ｅ]\s*[1-4１-４]?\s*型题|配伍|最佳答案"),
]
MAX_TYPES = 12
MAX_ITEMS = 30  # 答题要求、考试范围、成绩构成各自最多几条
MAX_TEXT = 300


def normalize_type(name: str) -> str:
    """题型名称 → 题型代码（choice、term……）；认不出时为 other。"""
    text = name.strip()
    for key, label in QUESTION_TYPES.items():
        if text == label or text == key:
            return key
    for key, pattern in _TYPE_PATTERNS:
        if re.search(pattern, text, re.I):
            return key
    return "other"


class PlanEvidence(BaseModel):
    """一条内容的出处。lecture：课堂原话（已与转写核对，verified 为核对结果）；user / material：说明或资料原文。"""

    source: PlanSource = "user"
    quote: str = ""
    lecture_id: int | None = None
    seq: int | None = None  # 讲次
    segment_ids: list[int] = Field(default_factory=list)
    t: int | None = None  # 原话开始的时间（毫秒），用于跳到录音
    verified: bool = True


class PlanComponent(BaseModel):
    name: str
    percent: float = 0
    evidence: PlanEvidence = Field(default_factory=PlanEvidence)


class PlanQuestionType(BaseModel):
    name: str  # 显示的名称（按来源原文，例如「A1 型题」）
    key: str = "other"  # 题型代码，见 QUESTION_TYPES
    count: int = 0
    points_each: float = 0
    total_points: float = 0  # 这种题型合计分值：有题数和每题分值时由代码算出
    share: float = 0  # 占全部题型分值的比例（0–1），由代码算出
    note: str = ""
    evidence: PlanEvidence = Field(default_factory=PlanEvidence)


class PlanRule(BaseModel):
    text: str
    question_type: str = ""  # 适用的题型名称；空表示所有题型
    evidence: PlanEvidence = Field(default_factory=PlanEvidence)


class PlanScope(BaseModel):
    kind: Literal["include", "exclude", "focus"] = "include"
    text: str
    evidence: PlanEvidence = Field(default_factory=PlanEvidence)


class ExamPlan(BaseModel):
    exam_form: str = ""
    duration_minutes: int = 0
    total_points: float = 0  # 来源里说的试卷总分（可能为 0：没说）
    components: list[PlanComponent] = Field(default_factory=list)
    question_types: list[PlanQuestionType] = Field(default_factory=list)
    answer_rules: list[PlanRule] = Field(default_factory=list)
    scope: list[PlanScope] = Field(default_factory=list)
    not_mentioned: list[str] = Field(default_factory=list)

    def computed(self) -> ExamPlan:
        """补上各题型的题型代码、合计分值和占比（全部由代码算）。"""
        types = []
        for item in self.question_types:
            total = item.count * item.points_each if item.count > 0 and item.points_each > 0 else item.total_points
            types.append(item.model_copy(update={"key": normalize_type(item.name) if item.key == "other"
                                                 else item.key, "total_points": round(max(total, 0), 2)}))
        base = sum(item.total_points for item in types) or self.total_points
        types = [item.model_copy(update={"share": round(item.total_points / base, 4) if base > 0 else 0})
                 for item in types]
        return self.model_copy(update={"question_types": types})

    @property
    def types_total(self) -> float:
        return sum(item.total_points for item in self.question_types)

    def type_names(self) -> list[str]:
        return [item.name for item in self.question_types]

    def match_type(self, name: str) -> PlanQuestionType | None:
        """模型或笔记写的题型 → 方案里的题型：先比名称，再比题型代码。"""
        text = name.strip()
        for item in self.question_types:
            if item.name == text:
                return item
        key = normalize_type(text)
        if key == "other":
            return None
        return next((item for item in self.question_types if item.key == key), None)

    def fingerprint(self) -> str:
        """方案内容的指纹（不含出处）：确认时与上次确认的比较，内容没变就不重新整理笔记。"""
        content = self.model_dump(exclude={"components": {"__all__": {"evidence"}},
                                           "question_types": {"__all__": {"evidence"}},
                                           "answer_rules": {"__all__": {"evidence"}},
                                           "scope": {"__all__": {"evidence"}}})
        return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def is_empty(self) -> bool:
        return not (self.exam_form or self.duration_minutes or self.total_points or self.components
                    or self.question_types or self.answer_rules or self.scope)


@dataclass(frozen=True)
class StoredPlan:
    course_id: int
    source_lecture_id: int | None
    user_text: str
    plan: ExamPlan | None  # 还没分析过时为空
    confirmed_fingerprint: str
    model: str
    created_at: str
    updated_at: str
    confirmed_at: str


class ExamPlanStore:
    def __init__(self, library: Library) -> None:
        self.library = library
        self.conn = library.conn

    def get(self, course: Course) -> StoredPlan | None:
        row = self.conn.execute("SELECT * FROM exam_plan WHERE course_id = ?", (course.id,)).fetchone()
        if row is None:
            return None
        plan = ExamPlan.model_validate_json(row["result_json"]) if row["result_json"] else None
        return StoredPlan(row["course_id"], row["source_lecture_id"], row["user_text"], plan,
                          row["confirmed_fingerprint"], row["model"], row["created_at"], row["updated_at"],
                          row["confirmed_at"])

    def active(self, course: Course) -> ExamPlan | None:
        """后续各步骤使用的方案：只有已确认的才算。"""
        if course.exam_plan != "confirmed":
            return None
        stored = self.get(course)
        return stored.plan if stored else None

    def save_inputs(self, course: Course, source_lecture_id: int | None, user_text: str) -> None:
        """记下分析用的来源讲和用户提供的考试说明（分析前先保存，分析失败也不丢）。"""
        now = _now()
        self.conn.execute(
            "INSERT INTO exam_plan (course_id, source_lecture_id, user_text, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?) ON CONFLICT(course_id) DO UPDATE SET source_lecture_id = excluded.source_lecture_id,"
            " user_text = excluded.user_text, updated_at = excluded.updated_at",
            (course.id, source_lecture_id, clean_user_text(user_text), now, now),
        )
        self.conn.commit()

    def save_draft(self, course: Course, plan: ExamPlan, model: str) -> Course:
        """保存分析结果，方案回到待确认。"""
        self._write(course, plan, model=model)
        return self.library.update_course(course.id, exam_plan="draft")

    def save_edit(self, course: Course, plan: ExamPlan) -> None:
        """保存用户在表格里的修改（状态不变，确认另外进行）。"""
        self._write(course, plan)

    def confirm(self, course: Course) -> tuple[Course, bool]:
        """确认方案，返回（课程，内容是否和上次确认的不同）。"""
        stored = self.get(course)
        if stored is None or stored.plan is None:
            raise LibraryError(f"「{course.name}」还没有考试方案：先分析或填写")
        fingerprint = stored.plan.fingerprint()
        changed = fingerprint != stored.confirmed_fingerprint
        self.conn.execute("UPDATE exam_plan SET confirmed_fingerprint = ?, confirmed_at = ? WHERE course_id = ?",
                          (fingerprint, _now(), course.id))
        self.conn.commit()
        return self.library.update_course(course.id, exam_plan="confirmed"), changed

    def skip(self, course: Course) -> Course:
        return self.library.update_course(course.id, exam_plan="skipped")

    def _write(self, course: Course, plan: ExamPlan, model: str | None = None) -> None:
        now = _now()
        data = plan.computed().model_dump_json()
        self.conn.execute(
            "INSERT INTO exam_plan (course_id, result_json, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(course_id) DO UPDATE SET result_json = excluded.result_json,"
            " model = CASE WHEN ? IS NULL THEN exam_plan.model ELSE excluded.model END,"
            " updated_at = excluded.updated_at",
            (course.id, data, model or "", now, now, model),
        )
        self.conn.commit()


def clean_user_text(text: str, limit: int = 6000) -> str:
    text = text.replace("\r\n", "\n").strip()
    if len(text) > limit:
        raise LibraryError(f"考试说明最多 {limit} 字（现在 {len(text)} 字）")
    return text


# ---------- 给模型看的文字 ----------


def _points(value: float) -> str:
    return f"{value:g}"


def type_line(item: PlanQuestionType) -> str:
    parts = [item.name]
    if item.count and item.points_each:
        parts.append(f"{item.count} 题 × {_points(item.points_each)} 分 = {_points(item.total_points)} 分")
    elif item.total_points:
        parts.append(f"共 {_points(item.total_points)} 分" + (f"（{item.count} 题）" if item.count else ""))
    elif item.count:
        parts.append(f"{item.count} 题（分值未提及）")
    if item.share:
        parts.append(f"占 {item.share:.0%}")
    if item.note:
        parts.append(item.note)
    return "，".join(parts)


def plan_text(plan: ExamPlan) -> str:
    """方案的文字说明（提示词、导出共用）。"""
    lines: list[str] = []
    head = [plan.exam_form or "",
            f"考试 {plan.duration_minutes} 分钟" if plan.duration_minutes else "",
            f"卷面 {_points(plan.total_points)} 分" if plan.total_points else ""]
    if any(head):
        lines.append("- 考核方式：" + "；".join(x for x in head if x))
    if plan.components:
        lines.append("- 成绩构成：" + "、".join(
            f"{c.name} {_points(c.percent)}%" if c.percent else c.name for c in plan.components))
    if plan.question_types:
        lines.append("- 题型与分值：")
        lines += [f"  - {type_line(item)}" for item in plan.question_types]
    if plan.answer_rules:
        lines.append("- 答题要求：")
        lines += [f"  - {'［' + rule.question_type + '］' if rule.question_type else ''}{rule.text}"
                  for rule in plan.answer_rules]
    if plan.scope:
        lines.append("- 考试范围：")
        lines += [f"  - {SCOPE_LABELS[item.kind]}：{item.text}" for item in plan.scope]
    if plan.not_mentioned:
        lines.append("- 老师没有提到：" + "、".join(plan.not_mentioned))
    return "\n".join(lines)


# 各步骤怎么用考试方案（附在方案后面）
_PLAN_USES = {
    "extract": (
        "按考试方案整理笔记：\n"
        "- 每个知识点的 question_types 填它适合出成的题型，只能从上面方案的题型名称里选（可以多选，都不适合时为空列表）；\n"
        "- 方案里分值占比高的题型对应的内容记得更细，例如病例分析、论述题占比高时，把临床表现、机制、用药依据记完整；\n"
        "- 方案说不考的内容照常可以记录，但不要标为考点信号；老师在课上重申的范围仍按 scope_include / scope_exclude 记录。\n"
    ),
    "review": (
        "按考试方案编写复习要点：每个知识点按它适合的题型（输入里「适合题型」）和上面的答题要求来写——"
        "名词解释写成一两句可以直接作答的定义，简答题写成分点作答的要点，论述题写成有总述、分述、小结的提纲，"
        "病例分析写出诊断或用药依据、治疗要点；选择题、填空题等侧重容易混淆的数值、对比和易错点。"
        "没有标注题型的知识点照常编写。\n"
    ),
    "report": (
        "按考试方案判断：方案里的每种题型都要给出候选知识点（选择题等小题按题数给出足够的候选，名词解释、简答、论述、"
        "病例分析等大题按题数多给一些候选）；question_types 只用方案里的题型名称；方案说不考的内容不要选。\n"
    ),
    "qa": "回答「某道题怎么答」「这个知识点会怎么考」一类问题时，按方案里的题型和答题要求组织答案。\n",
}


def plan_block(plan: ExamPlan | None, use: str = "") -> str:
    """附加在整理笔记、总复习、考前报告、问答提示词里的考试方案，use 为步骤（附上这一步怎么用方案）；
    没有方案时为空字符串，提示词与没有这个功能时完全一样。"""
    if plan is None or plan.is_empty:
        return ""
    return ("\n本课程的考试方案（已由用户确认，后续分析以此为准）：\n" + plan_text(plan) + "\n"
            + _PLAN_USES.get(use, ""))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
