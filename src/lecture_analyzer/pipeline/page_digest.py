"""页面识别：把课件/资料页面整理成结构化内容（课件与资料共用）。"""

from __future__ import annotations

from typing import Literal, Sequence

from ..documents import DocPage
from ..llm import ImagePart, LLMProvider, Part, TextPart, Usage
from ..llm.base import Effort
from ..prompts import load_prompt
from ..schemas import PageDigest, PageDigestBatch

ImagePolicy = Literal["auto", "always", "never"]

_SPARSE_TEXT_CHARS = 80  # 文字少于这个数的页面，多半是图片页或公式页
_MATH_HINTS = set("=∑∫√±≤≥≠≈∞∂∇∈∉⊂∀∃→⇒αβγδεθλμπρσφω^")


def needs_image(page: DocPage, policy: ImagePolicy) -> bool:
    """决定是否把页面图片发给模型：图片按 token 计费，纯文字页只发文字。"""
    if page.image_png is None or policy == "never":
        return False
    if policy == "always":
        return True
    return page.has_images or len(page.text) < _SPARSE_TEXT_CHARS or _looks_mathy(page.text)


def _looks_mathy(text: str) -> bool:
    return sum(ch in _MATH_HINTS for ch in text) >= 3


def digest_pages(
    provider: LLMProvider,
    pages: Sequence[DocPage],
    *,
    subject: str = "医学",
    policy: ImagePolicy = "auto",
    batch_size: int = 6,
    effort: Effort | None = None,
) -> tuple[list[PageDigest], Usage]:
    """分批识别页面，返回与输入页一一对应的结果和总用量。"""
    system = load_prompt("page_digest", subject=subject)
    digests: list[PageDigest] = []
    usage = Usage()
    for start in range(0, len(pages), batch_size):
        batch = pages[start : start + batch_size]
        result = provider.generate_structured(
            _batch_parts(batch, policy, provider.profile.vision), PageDigestBatch, system=system, effort=effort
        )
        usage = usage + result.usage
        digests.extend(_align(batch, result.output.pages))
    return digests, usage


def _batch_parts(batch: Sequence[DocPage], policy: ImagePolicy, vision: bool) -> list[Part]:
    parts: list[Part] = []
    for page in batch:
        parts.append(TextPart(f"=== 第 {page.page_no} 页 ===\n抽取文字：\n{page.text or '（无）'}"))
        if vision and needs_image(page, policy):
            parts.append(ImagePart(page.image_png))  # type: ignore[arg-type]
    first, last = batch[0].page_no, batch[-1].page_no
    parts.append(TextPart(f"请输出以上 {len(batch)} 页（第 {first}–{last} 页）的整理结果。"))
    return parts


def _align(batch: Sequence[DocPage], outputs: Sequence[PageDigest]) -> list[PageDigest]:
    """按页码对齐模型输出；漏掉的页用抽取文字兜底，避免页码错位。"""
    by_page = {digest.page_no: digest for digest in outputs}
    aligned: list[PageDigest] = []
    for page in batch:
        digest = by_page.get(page.page_no)
        if digest is None:
            digest = PageDigest(page_no=page.page_no, title="", content_md=page.text or "（未识别）", terms=[])
        aligned.append(digest)
    return aligned


def collect_terms(digests: Sequence[PageDigest]) -> list[str]:
    """汇总术语表（去重并保持首次出现的顺序），用作语音识别热词。"""
    seen: dict[str, None] = {}
    for digest in digests:
        for term in digest.terms:
            term = term.strip()
            if term:
                seen.setdefault(term, None)
    return list(seen)
