"""模型输出渲染成 HTML：Markdown（禁用原始 HTML，公式交给浏览器端的 KaTeX），以及问答回答里的出处编号。"""

from __future__ import annotations

import re
import secrets
from typing import Any

from markdown_it import MarkdownIt
from markupsafe import Markup, escape

from ..pipeline.report_render import evidence_text
from ..textnorm import MATH, split_code

_MARKDOWN = MarkdownIt("commonmark", {"html": False, "breaks": True}).enable("table")
_OPEN, _CLOSE = chr(0xE000), chr(0xE001)  # 私用区字符
# 占位符是私用区字符包住「每次调用随机的 nonce + 公式编号」。正文里自带同样的字符（包括 &#xE000; 这类
# 实体经 markdown-it 解码出来的）没有这次调用的 nonce，不会被当成占位符替换。
_PLACEHOLDER = re.compile(_OPEN + r"([0-9a-f]{8})(\d+)" + _CLOSE)
# CommonMark 规定 `)**` 后紧跟文字时不算加粗结束，中文、韩文里很常见（如 **장(소장·대장)**인），渲染后补上
_LEFTOVER_BOLD = re.compile(r"\*\*([^*<>\n]+?)\*\*")
_TAG = re.compile(r"(<[^>]*>)")
_CITATION = re.compile(r"\[([QMPENTF]\d+)\]")


def _fix_leftover_bold(html: str) -> str:
    """给渲染结果补上加粗；只看代码之外的文字，标签属性（href、title）和代码内容都不动。"""
    parts = _TAG.split(html)
    depth = 0
    for index, part in enumerate(parts):
        if index % 2:
            if part.startswith("<code"):
                depth += 1
            elif part.startswith("</code"):
                depth = max(0, depth - 1)
        elif depth == 0:
            parts[index] = _LEFTOVER_BOLD.sub(r"<strong>\1</strong>", part)
    return "".join(parts)


def render_markdown(text: str | None) -> Markup:
    """Markdown 转 HTML；公式先替换成占位符，避免其中的 * 和 _ 被当成强调，渲染后再转义放回。
    代码段（围栏代码块、行内代码）原样保留，其中的 $ 不当公式、** 也不补加粗。
    已知限制：链接地址里的 $（如 [a](https://e.com/$x$)）仍会被占位符破坏；转义的 \\$ 仍不被识别。"""
    spans: list[str] = []
    nonce = secrets.token_hex(4)

    def stash(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"{_OPEN}{nonce}{len(spans) - 1}{_CLOSE}"  # 不含冒号等行内终止符，公式多时渲染不会变慢

    def restore(match: re.Match[str]) -> str:
        index = int(match.group(2))
        if match.group(1) == nonce and index < len(spans):
            return str(escape(spans[index]))
        return match.group(0)

    source = text or ""
    pieces = [part if is_code else MATH.sub(stash, part) for part, is_code in split_code(source)]
    # 先补加粗、后恢复公式：公式里的 ** 不会被当成加粗
    html = _fix_leftover_bold(_MARKDOWN.render("".join(pieces)))
    return Markup(_PLACEHOLDER.sub(restore, html))


def kp_url(course_id: int, key: int) -> str:
    """知识点的具体内容：总复习里这个知识点的条目（要点、老师原话、例题、相关资料；没生成总复习时是各讲笔记原文）。
    折叠区里的条目打开页面时自动展开（app.js 的 revealHash）。"""
    return f"/courses/{course_id}/review#k-{key}"


def evidence_url(view: dict[str, Any]) -> str:
    """出处的链接：原话、转写、笔记、例题跳到录音的时间点；课件页是页面图片；资料题目打开资料页。"""
    kind = view["id"][0]
    if kind == "P":
        return f"/materials/{view['material_id']}/pages/{view['page_no']}.png"
    if kind == "M":
        return f"/materials/{view['material_id']}"
    if kind == "F":
        return f"/courses/{view['course_id']}/qa/attachments/{view['token']}"
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
