"""把笔记、课程总览渲染成 Markdown（书的导出与实验脚本共用），以及界面上共用的中文标签。"""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING, Any, Sequence

from .asr.base import Segment
from .library import CheckedSignal, Course, Lecture
from .schemas import LectureNotes, Signal
from .timecode import format_ms

if TYPE_CHECKING:  # 只用于类型标注，避免与 knowledge 互相导入
    from .knowledge import HistorySignal, KnowledgePoint, Reference

SIGNAL_LABELS = {
    "exam_hint": "提示考试",
    "scope_include": "考试范围",
    "scope_exclude": "不考",
    "emphasis": "强调",
    "common_mistake": "易错",
    "worked_example": "例题",
    "derivation": "推导",
    "homework": "作业",
    "recap": "回顾",
    "off_slide": "课件外补充",
    "skipped_slide": "跳过的课件",
}
KIND_LABELS = {"normal": "普通课", "exercise": "习题课", "review": "复习课"}
ROLE_LABELS = {"first": "首次讲", "deepen": "深化", "review": "回顾", "apply": "应用"}
RELATION_LABELS = {"recap": "回顾", "builds_on": "承接", "contrast": "对比", "repeat": "再次强调"}


def first_time(segment_ids: Sequence[int], segments: Sequence[Segment]) -> int | None:
    """出处中第一句的开始时间（毫秒）；编号都无效时返回 None。"""
    valid = [i for i in segment_ids if 1 <= i <= len(segments)]
    return segments[min(valid) - 1].start_ms if valid else None


def _time_label(segment_ids: Sequence[int], segments: Sequence[Segment]) -> str:
    start = first_time(segment_ids, segments)
    return format_ms(start) if start is not None else "-"


def pages_label(pages: Sequence[int], labels: Sequence[str] = ()) -> str:
    """P3（ch2-分类与复制策略 第 5 页）；labels[i - 1] 是 Pi 在课程资料里的出处。"""
    return "、".join(f"P{p}（{labels[p - 1]}）" if 0 < p <= len(labels) else f"P{p}" for p in pages) or "无"


def _skipped_pages(signal: Any, labels: Sequence[str]) -> str:
    """没有原话的"跳过的课件"信号，显示对应的课件页。"""
    return f"课件 {pages_label(signal.slide_pages, labels)}" if signal.slide_pages else ""


def table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_notes(
    notes: LectureNotes,
    segments: Sequence[Segment],
    checked: Sequence[CheckedSignal] | None = None,
    heading: str | None = None,
    references: Sequence[Reference] | None = None,
    page_labels: Sequence[str] = (),
) -> str:
    """渲染笔记。传入 checked 时，考点信号表附带原话校验结果；传入 references 时附「与前几讲的联系」；
    page_labels 是课件页 P1、P2… 的出处。"""
    lines = [f"# {heading or notes.title}", "", "## 大纲", *(f"- {item}" for item in notes.outline), ""]
    if references:
        lines += ["## 与前几讲的联系", *render_references(references, notes, segments), ""]
    lines.append("## 知识点")
    for kp in notes.knowledge_points:
        lines += [
            f"### {kp.name}（约 {kp.minutes:g} 分钟 · [{_time_label(kp.segment_ids, segments)}]"
            f" · 课件 {pages_label(kp.slide_pages, page_labels)}）",
            kp.summary_md,
            "",
        ]
    lines.append("## 例题与推导")
    for example in notes.examples:
        lines += [
            f"### {example.title}（[{_time_label(example.segment_ids, segments)}]）",
            example.problem_md,
            "",
            "**解答**",
            "",
            example.solution_md,
            "",
        ]
    lines += ["## 作业", *(f"- {item}" for item in notes.homework or ["（无）"]), "", "## 考点信号", ""]
    lines += _signal_table(notes, segments, checked, page_labels)
    return "\n".join(lines) + "\n"


def render_references(references: Sequence[Reference], notes: LectureNotes, segments: Sequence[Segment]) -> list[str]:
    lines = []
    for ref in references:
        name = notes.knowledge_points[ref.note_index - 1].name if ref.note_index <= len(notes.knowledge_points) else ""
        lines.append(
            f"- 【{RELATION_LABELS.get(ref.relation, ref.relation)}】[{_time_label(ref.segment_ids, segments)}] {name}"
            f" ← 第{ref.target_seq}讲「{ref.kp_name}」：{ref.explanation}"
        )
    return lines


def signal_stats(history: Sequence[HistorySignal]) -> dict[int, dict[str, dict[str, int]]]:
    """各课程知识点的考点信号统计：{K 编号: {信号类型: {"count": 条数, "strength": 强度之和}}}。"""
    stats: dict[int, dict[str, dict[str, int]]] = {}
    for signal in history:
        entry = stats.setdefault(signal.key, {}).setdefault(signal.type, {"count": 0, "strength": 0})
        entry["count"] += 1
        entry["strength"] += signal.strength
    return stats


def render_overview(
    course: Course, lectures: Sequence[Lecture], tree: Sequence[KnowledgePoint], stats: dict[int, dict[str, Any]]
) -> str:
    """课程总览：各讲目录，以及按章节分组的知识点体系（出现的讲次、角色、信号统计）。"""
    lines = [f"# {course.name} · 课程总览", "", "## 目录"]
    lines += [f"- 第{lecture.seq}讲：{lecture.title or '（待处理）'}" for lecture in lectures]
    lines += ["", f"## 知识点体系（{len(tree)} 个）"]
    for chapter, points in group_by_chapter(tree):
        lines += ["", f"### {chapter}"]
        for kp in points:
            where = "、".join(f"第{a.seq}讲{ROLE_LABELS.get(a.role, a.role)}（{a.minutes:g} 分钟）" for a in kp.appearances)
            shown = zh_aliases(kp.aliases)
            aliases = f"（又称：{'、'.join(shown)}）" if shown else ""
            counts = "、".join(f"{SIGNAL_LABELS.get(t, t)} {v['count']}" for t, v in stats.get(kp.key, {}).items())
            lines.append(f"- K{kp.key} {kp.name}{aliases}：{where}" + (f" · 信号：{counts}" if counts else ""))
    return "\n".join(lines) + "\n"


_HANGUL = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")
_HANZI = re.compile(r"[\u4e00-\u9fff]")


def quote_html(quote: str, meaning: str = "") -> str:
    """老师原话：有中文意思时先显示意思，原文收起、点开可核对；没有意思（原话本来就是中文）时显示原话。
    分析结果全中文（M4 修订）；网页和导出的 Markdown 共用（Markdown 里可以直接写 HTML）。"""
    if not meaning:
        return f"「{html.escape(quote, quote=False)}」"
    return (f'<span class="quote-meaning">{html.escape(meaning, quote=False)}</span>'
            f'<details class="quote-original"><summary>原文</summary>{html.escape(quote, quote=False)}</details>')


def zh_aliases(aliases: Sequence[str]) -> list[str]:
    """要显示的别名：只要中文的。外语原词只在后台用于匹配外语出的考题和问答检索，不显示（M4 修订）。"""
    return [alias for alias in aliases if _HANZI.search(alias) and not _HANGUL.search(alias)]


def group_by_chapter(tree: Sequence[KnowledgePoint]) -> list[tuple[str, list[KnowledgePoint]]]:
    """按章节分组，章节按首次出现的先后排列；没有章节的归入「未分章节」。"""
    groups: dict[str, list[KnowledgePoint]] = {}
    for kp in tree:
        groups.setdefault(kp.chapter or "未分章节", []).append(kp)
    return list(groups.items())


def _quote_cell(signal: Signal, page_labels: Sequence[str]) -> str:
    """原话（外语授课时先写中文意思，原文收起）；跳过课件页的信号没有原话，写页码。"""
    if not signal.quote:
        return _skipped_pages(signal, page_labels)
    return quote_html(signal.quote, signal.meaning)


def _signal_table(
    notes: LectureNotes, segments: Sequence[Segment], checked: Sequence[CheckedSignal] | None,
    page_labels: Sequence[str] = (),
) -> list[str]:
    if checked is None:
        rows = [(s, s.segment_ids, None) for s in notes.signals]
    else:
        rows = [(c.signal, c.segment_ids, c.valid) for c in checked]
    header = "| 类型 | 知识点 | 强度 | 时间 | 原话 |" + (" 校验 |" if checked is not None else "")
    lines = [header, "| --- " * (6 if checked is not None else 5) + "|"]
    for signal, ids, valid in rows:
        row = (
            f"| {SIGNAL_LABELS.get(signal.type, signal.type)} | {table_cell(signal.knowledge_point)}"
            f" | {signal.strength}"
            f" | {_time_label(ids, segments)} | {table_cell(_quote_cell(signal, page_labels))} |"
        )
        if valid is not None:
            row += " 通过 |" if valid else " 未找到原话 |"
        lines.append(row)
    return lines
