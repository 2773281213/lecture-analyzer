"""模型输出渲染成 HTML：Markdown（禁用原始 HTML，公式交给浏览器端的 KaTeX），以及问答回答里的出处编号。"""

from __future__ import annotations

import re
from typing import Any

from markdown_it import MarkdownIt
from markupsafe import Markup, escape

from ..pipeline.report_render import evidence_text

_MARKDOWN = MarkdownIt("commonmark", {"html": False, "breaks": True}).enable("table")
_MATH = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$", re.S)
_OPEN, _CLOSE = chr(0xE000), chr(0xE001)  # 私用区字符包住公式编号，不会与正文冲突
_PLACEHOLDER = re.compile(_OPEN + r"(\d+)" + _CLOSE)
# CommonMark 规定 `)**` 后紧跟文字时不算加粗结束，中文、韩文里很常见（如 **장(소장·대장)**인），渲染后补上
_LEFTOVER_BOLD = re.compile(r"\*\*([^*<>\n]+?)\*\*")
_TAG = re.compile(r"(<[^>]*>)")
_CITATION = re.compile(r"\[([QMPENT]\d+)\]")


def render_markdown(text: str | None) -> Markup:
    """Markdown 转 HTML；公式先替换成占位符，避免其中的 * 和 _ 被当成强调，渲染后再转义放回。"""
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"{_OPEN}{len(spans) - 1}{_CLOSE}"

    html = _LEFTOVER_BOLD.sub(r"<strong>\1</strong>", _MARKDOWN.render(_MATH.sub(stash, text or "")))
    return Markup(_PLACEHOLDER.sub(lambda m: str(escape(spans[int(m.group(1))])), html))


def evidence_url(view: dict[str, Any]) -> str:
    """出处的链接：原话、转写、笔记、例题跳到录音的时间点；课件页是页面图片；资料题目打开资料页。"""
    kind = view["id"][0]
    if kind == "P":
        return f"/materials/{view['material_id']}/pages/{view['page_no']}.png"
    if kind == "M":
        return f"/materials/{view['material_id']}"
    time = f"?t={view['t']}" if view.get("t") is not None else ""
    return f"/lectures/{view['lecture_id']}{time}"


def cite_links(html: Markup, evidence: dict[str, dict[str, Any]]) -> Markup:
    """把回答里的 [N1]、[T2]… 换成出处链接。只替换文字部分，不动标签和属性（markdown-it 的输出里
    文字中的 < > 都已转义，按标签切分是可靠的）；不在 evidence 里的编号原样保留。"""

    def link(match: re.Match[str]) -> str:
        view = evidence.get(match.group(1))
        if view is None:
            return match.group(0)
        css = "cite page-link" if view["id"][0] == "P" else "cite"
        return (f'<a class="{css}" href="{escape(evidence_url(view))}" title="{escape(evidence_text(view))}">'
                f"{escape(view['id'])}</a>")

    parts = _TAG.split(str(html))
    return Markup("".join(part if index % 2 else _CITATION.sub(link, part) for index, part in enumerate(parts)))
