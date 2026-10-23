"""模型输出的结构化格式。

为兼容各家的严格结构化输出：字段全部必填、不设默认值、不用数值范围约束
（范围在校验器里处理），枚举只用字符串。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class PageDigest(BaseModel):
    page_no: int = Field(description="页码，与输入中的页码一致")
    title: str = Field(description="本页标题；没有标题时概括一个短标题")
    content_md: str = Field(description="本页内容的 Markdown，公式用 LaTeX（行内 $...$，独立 $$...$$）")
    terms: list[str] = Field(description="本页出现的专业术语（中英文），用作语音识别热词")


class PageDigestBatch(BaseModel):
    pages: list[PageDigest]


# ---------- 课件匹配 ----------


class PageRange(BaseModel):
    file: int = Field(description="文件序号（F 后面的数字）")
    start: int = Field(description="起始页码")
    end: int = Field(description="结束页码（含）；只有一页时与 start 相同")


class SlideMatch(BaseModel):
    ranges: list[PageRange] = Field(description="这一讲讲到的页面区间；完全对不上时为空列表")
    note: str = Field(description="一句话说明匹配依据，或对不上的原因")


# ---------- 资料拆题与知识点对应 ----------


class ItemOut(BaseModel):
    label: str = Field(description="资料上的原题号或条目编号，如「一、3」；没有编号时为空字符串")
    item_type: str = Field(description="题型或条目类型，取值见提示词")
    content_md: str = Field(description="题干或条目内容，照录原文（选择题包括全部选项），公式用 LaTeX")
    score: float = Field(description="分值；资料上没写时填 0")
    answer_md: str = Field(description="资料上给出的答案或评分要点；没有时为空字符串")
    pages: list[int] = Field(description="所在页码")


class ItemBatch(BaseModel):
    continuation_md: str = Field(description="本批开头是上一批最后一条的后半部分时，填这部分内容；否则为空字符串")
    items: list[ItemOut]


class ItemMapping(BaseModel):
    item: int = Field(description="题目编号（M 后面的数字）")
    kps: list[str] = Field(description="考查的课程知识点编号，如 K3，最多 3 个；都不相关时为空列表")


class ItemMappingBatch(BaseModel):
    mappings: list[ItemMapping]


# ---------- 考前报告 ----------


class RankedKP(BaseModel):
    kp: str = Field(description="课程知识点编号，如 K3")
    reason: str = Field(description="一两句理由，只依据输入里的证据")
    question_types: list[str] = Field(description="预测的题型，如 名词解释、简答题")
    evidence: list[str] = Field(description="支持理由的证据编号，如 Q3、M2、P5、E1")


class KPRanking(BaseModel):
    ranking: list[RankedKP] = Field(description="最可能考的知识点，按可能性从高到低")


class QARetrieval(BaseModel):
    knowledge_points: list[str] = Field(description="和问题相关的知识点编号，如 K12，最相关的在前，最多 8 个；没有相关的就留空")
    lectures: list[int] = Field(description="问题针对某几讲的整体内容时（如「第 3 讲讲了什么」）填讲次，否则留空")


class QAAnswer(BaseModel):
    found: bool = Field(description="输入的课程内容里是否有回答这个问题所需的依据")
    answer_md: str = Field(description="依据课程内容的回答（Markdown），每处依据后用方括号写证据编号，如 [N2]、[T1]；"
                                       "课程内容里没有答案时直接说明没有")
    supplement_md: str = Field(description="课程内容之外的补充（通用知识或推断），没有就留空；这部分不写证据编号")


SignalType = Literal[
    "exam_hint",  # 明确提示考试
    "scope_include",  # 范围声明：要考
    "scope_exclude",  # 范围声明：不考
    "emphasis",  # 强调
    "common_mistake",  # 易错提醒
    "worked_example",  # 例题
    "derivation",  # 推导
    "homework",  # 作业
    "recap",  # 课前回顾
    "off_slide",  # 课件外补充
    "skipped_slide",  # 被跳过的课件页
]


class Signal(BaseModel):
    type: SignalType
    knowledge_point: str = Field(description="涉及的知识点名称")
    strength: int = Field(description="信号强度，只能取 1、2、3：1 一般，2 明显，3 非常明确")
    quote: str = Field(description="老师原话，必须逐字摘自转写中的连续片段；skipped_slide 可以为空字符串")
    meaning: str = Field(description="原话不是笔记语言时，用笔记语言写出它的意思；已经是笔记语言或为空时填空字符串")
    segment_ids: list[int] = Field(description="原话所在的转写句子编号（S 后面的数字）")
    slide_pages: list[int] = Field(description="相关课件页码；没有则为空列表")

    @model_validator(mode="before")
    @classmethod
    def _without_meaning(cls, data: object) -> object:
        """M4 之前保存的笔记没有 meaning：读出时当作空。给模型的输出结构里它仍是必填项。"""
        if isinstance(data, dict) and "meaning" not in data:
            return {**data, "meaning": ""}
        return data

    @field_validator("strength")
    @classmethod
    def _clamp_strength(cls, value: int) -> int:
        return min(max(value, 1), 3)


class KnowledgePointNote(BaseModel):
    name: str
    summary_md: str = Field(description="要点，Markdown，公式用 LaTeX")
    slide_pages: list[int]
    segment_ids: list[int]
    minutes: float = Field(description="大约讲了多少分钟，可根据句子时间戳估算")
    question_types: list[str] = Field(
        description="按考试方案，这个知识点适合出成哪些题型（只能从方案的题型里选）；没有考试方案时输出空列表")

    @model_validator(mode="before")
    @classmethod
    def _without_question_types(cls, data: object) -> object:
        """v0.2.0 之前保存的笔记没有 question_types：读出时当作空。给模型的输出结构里它仍是必填项。"""
        if isinstance(data, dict) and "question_types" not in data:
            return {**data, "question_types": []}
        return data


class WorkedExample(BaseModel):
    title: str
    problem_md: str
    solution_md: str
    segment_ids: list[int]
    slide_pages: list[int]


class LectureNotes(BaseModel):
    title: str = Field(description="本节课主题")
    outline: list[str] = Field(description="按讲授顺序列出的大纲")
    knowledge_points: list[KnowledgePointNote]
    examples: list[WorkedExample]
    homework: list[str]
    signals: list[Signal]


# ---------- 第二步：跨讲整合 ----------

KPRole = Literal[
    "first",  # 首次讲
    "deepen",  # 在已有基础上深入
    "review",  # 回顾、复习
    "apply",  # 在例题或新情境中应用
]
Relation = Literal[
    "recap",  # 回顾以前的内容
    "builds_on",  # 在以前的知识上继续展开
    "contrast",  # 与以前的内容对比
    "repeat",  # 再次强调以前讲过的重点
]


class KnowledgePointMapping(BaseModel):
    note_kp: int = Field(description="本讲知识点编号（N 后面的数字）")
    target: str = Field(description="对应的已有课程知识点编号，如 K3；是课程里新出现的知识点时填 new")
    name: str = Field(description="课程知识点的规范名称；对应已有知识点时照抄其名称")
    chapter: str = Field(description="所属章节或主题，尽量沿用已有章节名")
    role: KPRole
    aliases: list[str] = Field(description="本讲对这个知识点的其他叫法；没有则为空列表")


class SignalMapping(BaseModel):
    signal: int = Field(description="本讲考点信号编号（G 后面的数字）")
    note_kp: int = Field(description="它属于本讲哪个知识点（N 后面的数字）；都不相关时填 0")


class CrossReference(BaseModel):
    note_kp: int = Field(description="本讲知识点编号（N 后面的数字）")
    target: str = Field(description="被联系的已有课程知识点编号，如 K3")
    lecture_seq: int = Field(description="被联系的内容在第几讲")
    relation: Relation
    explanation: str = Field(description="一句话说明这一联系")
    segment_ids: list[int] = Field(description="本讲中体现这一联系的转写句子编号（S 后面的数字）")


class IntegrationResult(BaseModel):
    knowledge_points: list[KnowledgePointMapping]
    signals: list[SignalMapping]
    references: list[CrossReference]


# ---------- 总复习 ----------


class ReviewEntry(BaseModel):
    kp: str = Field(description="课程知识点编号，如 K3")
    summary_md: str = Field(description="合并各讲内容后的复习要点，Markdown，公式用 LaTeX")
    sources: list[str] = Field(description="要点依据的讲解编号，如 L2N3（第 2 讲第 3 个知识点）")
    emphasis: list[str] = Field(description="引用的老师原话编号，如 Q5；没有则为空列表")
    examples: list[str] = Field(description="相关例题编号，如 E2；没有则为空列表")


class ReviewChapter(BaseModel):
    overview_md: str = Field(description="本章导语：这一章讲什么、复习时抓什么")
    entries: list[ReviewEntry]


# ---------- 考试方案（先于知识点分析：考核方式、题型分值、答题要求、考试范围） ----------

PlanSourceType = Literal["user", "material", "lecture"]
_SOURCE_DESC = "出处：user 用户提供的考试说明，material 考试通知、复习提纲等资料，lecture 课堂转写"
_QUOTE_DESC = ("依据的原文：lecture 时逐字摘自转写的连续片段（不要改写、不要翻译），user、material 时摘自说明或资料原文；"
               "控制在 80 字以内")


class PlanComponentOut(BaseModel):
    name: str = Field(description="成绩组成部分，如 平时成绩、期中考试、期末考试、实验")
    percent: float = Field(description="占总评的百分比，如 70；没有说时填 0")
    source: PlanSourceType = Field(description=_SOURCE_DESC)
    quote: str = Field(description=_QUOTE_DESC)
    segment_ids: list[int] = Field(description="lecture 出处的转写句子编号（S 后面的数字）；其他出处为空列表")


class PlanQuestionTypeOut(BaseModel):
    type: str = Field(description="题型名称，按来源原文写，如 选择题、A1 型题、名词解释、简答题、论述题、病例分析")
    count: int = Field(description="题数；没有说时填 0")
    points_each: float = Field(description="每题分值；没有说时填 0")
    total_points: float = Field(description="这种题型合计多少分；只说了合计时填这里，没有说时填 0")
    note: str = Field(description="补充说明，如「6 题选做 4 题」「中英文均可」；没有时填空字符串")
    source: PlanSourceType = Field(description=_SOURCE_DESC)
    quote: str = Field(description=_QUOTE_DESC)
    segment_ids: list[int] = Field(description="lecture 出处的转写句子编号；其他出处为空列表")


class PlanRuleOut(BaseModel):
    text: str = Field(description="老师规定的答题方向或要求，如「简答题按要点给分，要写出作用机制和临床应用」")
    question_type: str = Field(description="适用的题型名称（与 question_types 里的写法一致）；对所有题型都适用时填空字符串")
    source: PlanSourceType = Field(description=_SOURCE_DESC)
    quote: str = Field(description=_QUOTE_DESC)
    segment_ids: list[int] = Field(description="lecture 出处的转写句子编号；其他出处为空列表")


class PlanScopeOut(BaseModel):
    kind: Literal["include", "exclude", "focus"] = Field(description="include 要考，exclude 不考，focus 重点")
    text: str = Field(description="范围说明，如「第三章全部要考」「药物经济学不考」")
    source: PlanSourceType = Field(description=_SOURCE_DESC)
    quote: str = Field(description=_QUOTE_DESC)
    segment_ids: list[int] = Field(description="lecture 出处的转写句子编号；其他出处为空列表")


class ExamPlanOutput(BaseModel):
    exam_form: str = Field(description="考核方式，如 闭卷笔试、开卷、机考、论文；来源里没有提到时填空字符串")
    duration_minutes: int = Field(description="考试时长（分钟）；没有说时填 0")
    total_points: float = Field(description="试卷总分；没有说时填 0")
    components: list[PlanComponentOut] = Field(description="成绩构成；没有说时为空列表")
    question_types: list[PlanQuestionTypeOut] = Field(description="题型及分值；没有说时为空列表")
    answer_rules: list[PlanRuleOut] = Field(description="老师规定的答题方向和要求；没有说时为空列表")
    scope: list[PlanScopeOut] = Field(description="考试范围、不考的内容、重点；没有说时为空列表")
    not_mentioned: list[str] = Field(
        description="来源里没有提到的项目，从 考核方式、成绩构成、题型、分值、答题要求、考试范围 里选")

