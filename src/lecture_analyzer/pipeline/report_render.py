"""考前报告导出为 Markdown（网页用同一份 JSON 渲染）；evidence_text 也给问答的出处链接作悬停提示。"""

from __future__ import annotations

from typing import Any

from ..timecode import format_ms

Lines = list[str]


def evidence_text(view: dict[str, Any]) -> str:
    """一条证据的文字：原话附讲次和时间，题目附出处和题型，课件页附页码，例题附讲次；
    问答另有笔记要点（N）和转写片段（T）。"""
    kind = view["id"][0]
    if kind == "Q":  # 外语原话写中文意思（分析结果全中文）；原话本来就是中文时写原话
        return f"「{view.get('meaning') or view['quote']}」（{view['label']}，{_lecture(view)}）"
    if kind == "M":
        content = " ".join(view["content_md"].split())
        return f"「{view['title']}」{view['label']}（{view['type']}）：{content[:160]}"
    if kind == "P":
        return f"课件 {view['label']}" + (f"：{view['title']}" if view.get("title") else "")
    if kind == "N":
        return f"笔记「{view['name']}」（{_lecture(view)}）"
    if kind == "T":  # 转写是原文（可能是外语），提示里只写位置
        return f"转写片段（{_lecture(view)}）"
    return f"例题「{view['title']}」（{_lecture(view)}）"


def _lecture(view: dict[str, Any]) -> str:
    when = f" {format_ms(view['t'])}" if view.get("t") is not None else ""
    return f"第{view['seq']}讲{when}"


def render_report_markdown(title: str, result: dict[str, Any]) -> str:
    """按可能考到的程度列出考点：名称、星级、可能题型、知识点要点（M4 修订：只给结论，不给推断过程）。"""
    lines = [f"# {result['course']} · {title}", "", *_meta(result)]
    if result.get("version", 1) < 2:
        lines += ["", "> 这是旧版报告，只保留了考点名称；重新生成可得到知识点要点。", ""]
        return "\n".join(lines + [f"{row['rank']}. {row['name']}" for row in result.get("priorities", [])]) + "\n"
    if result.get("groups"):  # version 3：按考试方案的题型分组
        lines += _plan_lines(result)
        by_rank = {point["rank"]: point for point in result["points"]}
        for group in result["groups"]:
            row = next((t for t in result["plan_table"] if t["name"] == group["type"]), None)
            lines += ["", f"## {group['type']}" + (f"（{type_brief(row)}）" if row and type_brief(row) else "")]
            if not group["ranks"]:
                lines += ["", "（这种题型没有排进前列的考点）"]
            for rank in group["ranks"]:
                lines += _point_lines(by_rank[rank], "###")
    else:
        for point in result["points"]:
            lines += _point_lines(point, "##")
    if result["excluded"]:
        lines += ["", "## 不考的知识点", "", *(f"- {item['name']}" for item in result["excluded"])]
    return "\n".join(lines) + "\n"


def _point_lines(point: dict[str, Any], heading: str) -> list[str]:
    stars = "★" * point["priority"] if point["priority"] else ""
    lines = ["", f"{heading} {point['rank']}. {point['name']} {stars}".rstrip()]
    if point["types"]:
        lines += ["", f"可能题型：{'、'.join(point['types'])}"]
    return lines + ["", point["summary_md"]]


def type_brief(row: dict[str, Any]) -> str:
    """题型分值的简短说明，如「5 题 × 4 分 = 20 分，占 20%」。"""
    parts = []
    if row["count"] and row["points_each"]:
        parts.append(f"{row['count']} 题 × {row['points_each']:g} 分 = {row['total_points']:g} 分")
    elif row["total_points"]:
        parts.append(f"共 {row['total_points']:g} 分")
    elif row["count"]:
        parts.append(f"{row['count']} 题")
    if row["share"]:
        parts.append(f"占 {row['share']:.0%}")
    return "，".join(parts)


def _plan_lines(result: dict[str, Any]) -> list[str]:
    lines = ["", "## 题型与分值（考试方案）" + (f"：{result['plan_form']}" if result.get("plan_form") else ""), "",
             "| 题型 | 分值 | 说明 |", "| --- | --- | --- |"]
    lines += [f"| {row['name']} | {type_brief(row) or '未提及'} | {row['note']} |" for row in result["plan_table"]]
    if result.get("answer_rules"):
        lines += ["", "**答题要求**", ""]
        lines += [f"- {'［' + r['type'] + '］' if r['type'] else ''}{r['text']}" for r in result["answer_rules"]]
    return lines


def _meta(result: dict[str, Any]) -> list[str]:
    models = "、".join(f"{m['profile']}（{m['model'] or '失败'}）" for m in result["models"])
    scope = result["scope"]
    lectures = "第 " + "、".join(map(str, scope["lectures"])) + " 讲" if scope["lectures"] else "全部讲"
    materials = "、".join(scope["materials"]) if scope["materials"] else "全部资料"
    lines = [f"生成时间：{result['generated_at'].replace('T', ' ')}｜模型：{models}｜范围：{lectures}；{materials}",
             "", f"按可能考到的程度排列。{PRIORITY_LEGEND}"]
    return lines + [f"> 注意：{warning}" for warning in result.get("warnings", [])]


PRIORITY_LEGEND = ("星级与课程页、总复习相同：★★★ 老师提示考试或明确划重点，或出现在往年真题、模拟题、复习提纲里；"
                   "★★ 强调、易错或多次讲到；★ 一般。")
