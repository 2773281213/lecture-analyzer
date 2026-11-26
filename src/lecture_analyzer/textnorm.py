"""中文文本归一化与字错率（CER）：转写评估和原话校验共用；以及分析结果里外语旁注和引用编号的清理。"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Collection

from rapidfuzz.distance import Levenshtein

# 去掉的字符类别：标点（P）、空白分隔符（Z）、控制字符（C）
_DROP_CATEGORIES = ("P", "Z", "C")


def normalize_zh(text: str) -> str:
    """全角转半角、英文转小写，并去掉标点和空白。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith(_DROP_CATEGORIES))


# 代码段切分由 split_code 负责：先用 _FENCE_OPEN / _FENCE_CLOSE 线性逐行扫描围栏代码块
# （列表里缩进的也算；反引号围栏的信息串不含反引号，结束围栏同字符、长度不短于开栏且行尾仅有空白），
# 非围栏段再用 _INLINE_CODE 切出行内代码（内容不含反引号，两端反引号串长度相等）。
# 不要把代码和公式合进同一个 alternation，否则正文里孤立的 $ 会吞进后续行内代码里的 $，把代码切碎。
_FENCE_OPEN = re.compile(r"^[ \t]*+(?:(`{3,}+)[^`\n]*|(~{3,}+)[^\n]*)\r?\n$")
_FENCE_CLOSE = re.compile(r"^[ \t]*+(`{3,}+|~{3,}+)[ \t]*+\r?\n?$")
_INLINE_CODE = re.compile(r"(?<!`)(`++)[^`\n]+?\1(?!`)")
MATH = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$", re.S)

# 括号旁注：（……）或 (……)，不跨行、不嵌套；紧跟在 ] 后面的是 Markdown 链接地址，不算
_ASIDE = re.compile(r"(?<!\])[ \t]?(?<!\])[（(]([^（）()\n]{1,80})[）)]")
_HANGUL = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")
_HANZI = re.compile(r"[\u4e00-\u9fff]")
_LATIN_ONLY = re.compile(r"[A-Za-z][A-Za-z\s.,'’/&-]*")
_WORD = re.compile(r"[A-Za-z]+")
_ITEM = re.compile(r"[^、，,；;]+")
_SEPARATORS = "、，,；;"
# 条目开头引出外语说法的词（「称 eclipse」「英文：virus」），去掉它再判断是不是只有外语
_NAMING = re.compile(r"^(?:又称|也称|简称|俗称|称为|称|即|英文|英语|韩文|韩语|原文|原词)[:：\s]*")
# 引用输入编号的条目：「Q3」「见 Q6」「见例题 E2」「L2N3」，后面可以带句号
_CITATION = re.compile(r"(?:参见|详见|见例题|见)?\s*([KQE]\s*\d+|L\s*\d+\s*N\s*\d+)\s*[。.]?")


def strip_foreign_asides(text: str) -> str:
    """去掉括号里只有外语的旁注（分析结果全中文，M4 修订）：全是韩文的；英文单词或短语；中外文混排的括号
    去掉其中只有外语的条目（如「（又称痘疮、마마）」→「（又称痘疮）」）。
    DNA、mRNA、HDV、AmpC 这类缩写，单个字母的选项，带数字、公式、链接或中文的括号都保留。"""
    return _strip_asides(text, _foreign_item)


def strip_citations(text: str, labels: Collection[str]) -> str:
    """去掉括号里引用输入编号的旁注：页面不显示 K、L、Q、E 这些编号（「（见 Q6、Q7）」整个去掉，
    「（Q7，重点）」→「（重点）」）。只认 labels 里有的编号，「雌二醇（E2）」这类写法在没有这个编号时保留。"""
    def cited(item: str) -> bool:
        match = _CITATION.fullmatch(item)
        return match is not None and re.sub(r"\s+", "", match.group(1)).upper() in labels

    return _strip_asides(text, cited)


def _split_pattern(pattern: re.Pattern[str], text: str) -> list[tuple[str, bool]]:
    pieces: list[tuple[str, bool]] = []
    last = 0
    for match in pattern.finditer(text):
        if match.start() > last:
            pieces.append((text[last:match.start()], False))
        pieces.append((match.group(0), True))
        last = match.end()
    if last < len(text):
        pieces.append((text[last:], False))
    return pieces


def _split_lines_keepends(text: str) -> list[str]:
    """严格按 \\n 切行并保留行尾 \\n（不用 str.splitlines，避免把 \\u2028 等字符误当换行）。"""
    parts = text.split("\n")
    lines = [f"{part}\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def split_code(text: str) -> list[tuple[str, bool]]:
    """按 Markdown 代码段（围栏代码块与行内代码）切分文本，返回 (片段, 是否代码段) 列表；顺序拼接等于原文。"""
    if not text:
        return []
    if "```" not in text and "~~~" not in text:
        return _split_pattern(_INLINE_CODE, text) if "`" in text else [(text, False)]

    lines = _split_lines_keepends(text)
    count = len(lines)
    # 预先逆序统计每行之后反引号/波浪号结束围栏的最大长度：无闭合的开栏行可 O(1) 跳过，整体保持线性时间
    closes: list[tuple[str, int] | None] = [None] * count
    max_after_bt = [0] * count
    max_after_tilde = [0] * count
    best_bt = 0
    best_tilde = 0
    for idx in range(count - 1, -1, -1):
        max_after_bt[idx] = best_bt
        max_after_tilde[idx] = best_tilde
        close_match = _FENCE_CLOSE.match(lines[idx])
        if close_match is not None:
            fence = close_match.group(1)
            char, length = fence[0], len(fence)
            closes[idx] = (char, length)
            if char == "`":
                if length > best_bt:
                    best_bt = length
            elif length > best_tilde:
                best_tilde = length

    if best_bt == 0 and best_tilde == 0:
        return _split_pattern(_INLINE_CODE, text) if "`" in text else [(text, False)]

    pieces: list[tuple[str, bool]] = []
    plain: list[str] = []

    def flush_plain() -> None:
        if not plain:
            return
        chunk = "".join(plain)
        plain.clear()
        if "`" in chunk:
            pieces.extend(_split_pattern(_INLINE_CODE, chunk))
        else:
            pieces.append((chunk, False))

    idx = 0
    while idx < count:
        line = lines[idx]
        open_match = _FENCE_OPEN.match(line)
        if open_match is not None:
            fence = open_match.group(1) or open_match.group(2)
            char, length = fence[0], len(fence)
            max_after = max_after_bt[idx] if char == "`" else max_after_tilde[idx]
            if max_after >= length:
                end = idx + 1
                while end < count:
                    close = closes[end]
                    if close is not None and close[0] == char and close[1] >= length:
                        break
                    end += 1
                if end < count:
                    flush_plain()
                    pieces.append(("".join(lines[idx : end + 1]), True))
                    idx = end + 1
                    continue
        plain.append(line)
        idx += 1

    flush_plain()
    return pieces


def _strip_asides(text: str, drop: Callable[[str], bool]) -> str:
    # 代码段原样保留，非代码片段里再跳过公式，只清理普通文字中的括号旁注
    pieces: list[str] = []
    for part, is_code in split_code(text):
        if is_code:
            pieces.append(part)
            continue
        for chunk, is_math in _split_pattern(MATH, part):
            pieces.append(chunk if is_math else _ASIDE.sub(lambda match: _strip_aside(match, drop), chunk))
    return "".join(pieces)


def _strip_aside(match: re.Match[str], drop: Callable[[str], bool]) -> str:
    """整个括号该去掉就去掉；否则按「、，；」拆开，去掉该去的条目，其余照原来的分隔符连起来。"""
    inner = match.group(1).strip()
    if drop(inner):
        return ""
    items = [item.strip() for item in _ITEM.findall(inner)]
    kept = [item for item in items if not drop(item)]
    if len(kept) == len(items):
        return match.group(0)
    separator = next((ch for ch in inner if ch in _SEPARATORS), "、")
    return match.group(0).replace(match.group(1), separator.join(kept)) if kept else ""


def _foreign_item(text: str) -> bool:
    return _foreign_aside(_NAMING.sub("", text))


def _foreign_aside(inner: str) -> bool:
    if _HANGUL.search(inner):
        return not _HANZI.search(inner)
    if not _LATIN_ONLY.fullmatch(inner):
        return False
    # 有普通单词（3 个字母以上、大写字母少于 2 个）才算外语旁注；全是缩写的保留
    return any(len(word) >= 3 and sum(ch.isupper() for ch in word) < 2 for word in _WORD.findall(inner))


def cer(reference: str, hypothesis: str) -> float:
    """字错率 = 编辑距离 / 参考文本字数（均在归一化后计算）。"""
    ref = normalize_zh(reference)
    hyp = normalize_zh(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)
