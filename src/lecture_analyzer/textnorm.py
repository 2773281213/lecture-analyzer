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


# 围栏代码块（列表里缩进的也算；反引号围栏的信息串不能含反引号、结束行除空白外不能有其他字符）|
# 行内代码（内容不含反引号，两端反引号串长度相等）。
# 必须用占有量词，否则一长串反引号会二次方回溯；也不要把代码和公式合进同一个 alternation，
# 否则正文里孤立的 $ 会吞进后续行内代码里的 $，把代码切碎。
CODE = re.compile(
    r"^[ \t]*(`{3,}+)[^`\n]*\r?\n.*?^[ \t]*\1[ \t]*\r?$"
    r"|^[ \t]*(~{3,}+)[^\n]*\r?\n.*?^[ \t]*\2[ \t]*\r?$"
    r"|(?<!`)(`++)[^`\n]+?\3(?!`)",
    re.S | re.M,
)
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


def split_code(text: str) -> list[tuple[str, bool]]:
    """按 Markdown 代码段（围栏代码块与行内代码）切分文本，返回 (片段, 是否代码段) 列表；顺序拼接等于原文。"""
    return _split_pattern(CODE, text)


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
