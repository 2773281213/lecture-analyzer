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


# 括号旁注：（……）或 (……)，不跨行、不嵌套；紧跟在 ] 后面的是 Markdown 链接地址，不算
_ASIDE = re.compile(r"(?<!\])[ \t]?(?<!\])[（(]([^（）()\n]{1,80})[）)]")
_MATH_OR_CODE = re.compile(r"(\$\$.+?\$\$|\$[^$\n]+?\$|`[^`\n]+`)", re.S)  # 公式和代码里的括号不动
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


def _strip_asides(text: str, drop: Callable[[str], bool]) -> str:
    parts = _MATH_OR_CODE.split(text)  # 奇数位置是公式或代码
    return "".join(part if index % 2 else _ASIDE.sub(lambda match: _strip_aside(match, drop), part)
                   for index, part in enumerate(parts))


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
