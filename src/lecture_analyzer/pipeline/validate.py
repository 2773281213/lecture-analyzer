"""原话校验：考点信号引用的原话必须真的出现在转写里，防止模型编造。

三级匹配（都在去掉标点空白、统一全角半角之后进行）：
1. window：在引用的句子范围内（前后各放宽 1 句）做子串匹配。模型常把相邻几句短句拼成一条引用，
   所以匹配的是范围内各句拼接后的文本；
2. fuzzy：同一范围内做模糊匹配（partial_ratio ≥ 85），容忍个别字不同；
3. global：在全文中查找，找到就把出处改成原话实际所在的句子。
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Sequence

from rapidfuzz import fuzz

from ..asr.base import Segment
from ..library import CheckedSignal
from ..schemas import Signal
from ..textnorm import normalize_zh

FUZZY_THRESHOLD = 85
MIN_FUZZY_CHARS = 4  # 太短的引用做模糊匹配没有意义


class TranscriptIndex:
    """归一化后的全文和每句在其中的起点，用于按句子编号取范围、按位置反查句子。"""

    def __init__(self, segments: Sequence[Segment]) -> None:
        self.starts: list[int] = []
        parts: list[str] = []
        offset = 0
        for segment in segments:
            normalized = normalize_zh(segment.text)
            self.starts.append(offset)
            parts.append(normalized)
            offset += len(normalized)
        self.text = "".join(parts)
        self.count = len(segments)

    def window(self, segment_ids: Sequence[int]) -> str:
        """引用句子（编号从 1 开始）前后各放宽 1 句的拼接文本；编号全部无效时返回空串。"""
        valid = [i for i in segment_ids if 1 <= i <= self.count]
        if not valid:
            return ""
        first, last = max(min(valid) - 1, 1), min(max(valid) + 1, self.count)
        end = self.starts[last] if last < self.count else len(self.text)
        return self.text[self.starts[first - 1] : end]

    def locate(self, quote: str) -> list[int] | None:
        """在全文中查找，返回原话覆盖的句子编号。"""
        position = self.text.find(quote)
        if position < 0:
            return None
        first = bisect_right(self.starts, position)
        last = bisect_right(self.starts, position + len(quote) - 1)
        return list(range(first, last + 1))


def validate_signals(signals: Sequence[Signal], segments: Sequence[Segment]) -> list[CheckedSignal]:
    index = TranscriptIndex(segments)
    return [check_signal(signal, index) for signal in signals]


def check_signal(signal: Signal, index: TranscriptIndex) -> CheckedSignal:
    quote = normalize_zh(signal.quote)
    if not quote:
        # 只有"跳过的课件页"允许不带原话，此时以课件页码为证据
        valid = signal.type == "skipped_slide" and bool(signal.slide_pages)
        return CheckedSignal(signal, valid, "slide" if valid else "", signal.segment_ids)
    window = index.window(signal.segment_ids)
    if window and quote in window:
        return CheckedSignal(signal, True, "window", signal.segment_ids)
    if window and len(quote) >= MIN_FUZZY_CHARS and fuzz.partial_ratio(quote, window) >= FUZZY_THRESHOLD:
        return CheckedSignal(signal, True, "fuzzy", signal.segment_ids)
    located = index.locate(quote)
    if located is not None:
        return CheckedSignal(signal, True, "global", located)
    return CheckedSignal(signal, False, "", signal.segment_ids)


def pass_rate(checked: Sequence[CheckedSignal]) -> float:
    return sum(c.valid for c in checked) / len(checked) if checked else 1.0
