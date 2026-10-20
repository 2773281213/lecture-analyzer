"""课程设置：学科、授课语言（可多种）、现场翻译。提示词、转写方案和规则都按它调整，不写死某种授课语言。

语言用 ISO 639 代码（zh、ko、en、ja……），名称表在 LANGUAGE_NAMES，表里没有的代码照样可用，
界面上直接显示代码。旧写法 "ko+zh"（老师讲韩语、现场翻译成中文）会自动换算。
笔记和分析结果一律用中文（M4 修订：分析结果和知识点分析必须全中文），不再是课程选项。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator, model_validator

LANGUAGE_NAMES = {
    "zh": "中文", "yue": "粤语", "ko": "韩语", "ja": "日语", "en": "英语", "fr": "法语", "de": "德语",
    "es": "西班牙语", "pt": "葡萄牙语", "it": "意大利语", "ru": "俄语", "ar": "阿拉伯语", "vi": "越南语",
    "th": "泰语", "id": "印尼语", "ms": "马来语", "tr": "土耳其语", "hi": "印地语", "nl": "荷兰语",
    "pl": "波兰语", "uk": "乌克兰语", "fa": "波斯语", "he": "希伯来语", "mn": "蒙古语",
}
_CODE = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})?$")
NOTE_LANGUAGE = "zh"  # 笔记、课程记忆、总复习、考前报告、问答都用中文
# 老师提到以前内容时的常见说法，供跨讲整合找"回顾线索"；没有词表的语言照常运行，只是少这一路线索
RECAP_CUES = {
    "zh": ["上次", "上节", "上一节", "上周", "上星期", "昨天", "前天", "前面讲", "之前讲", "以前讲", "讲过", "学过"],
    "ko": ["지난번", "지난 시간", "지난시간", "지난주", "지난 주", "어제", "어저께", "저번", "배웠던", "배웠죠", "배운",
           "말했던", "얘기했던", "복습"],
    "en": ["last time", "last lecture", "last class", "last week", "yesterday", "previously", "we discussed",
           "we covered", "we learned", "as i mentioned", "recall that"],
    "ja": ["前回", "先週", "昨日", "前に説明", "前に話した", "習った", "復習"],
}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def recap_pattern(languages: list[str]) -> re.Pattern[str] | None:
    """这些语言的回顾说法合成一个正则（不区分大小写）；都没有词表时返回 None。"""
    cues = [cue for code in languages for cue in RECAP_CUES.get(code.split("-")[0], [])]
    return re.compile("|".join(re.escape(cue) for cue in cues), re.IGNORECASE) if cues else None


def parse_languages(text: str) -> list[str]:
    """"ko, zh" → ["ko", "zh"]；去重、转小写，并校验格式。"""
    codes: list[str] = []
    for raw in re.split(r"[,，、\s]+", text.strip()):
        code = raw.strip().lower()
        if not code:
            continue
        if not _CODE.match(code):
            raise ValueError(f"语言代码格式不对：{raw}（用 ISO 639 代码，如 zh、ko、en、ja）")
        if code not in codes:
            codes.append(code)
    return codes


class CourseSettings(BaseModel):
    subject: str = "医学"
    teaching: str = "zh"  # 授课语言，多种用逗号分隔，如 "ko,en"（老师混用几种语言）
    interpreted_to: str = ""  # 现场翻译成的语言；空表示没有现场翻译
    note_language: str = NOTE_LANGUAGE  # 固定为中文；书库里旧的其他取值读出时也按中文

    @model_validator(mode="before")
    @classmethod
    def _legacy_teaching(cls, data: Any) -> Any:
        """旧写法 "ko+zh" 表示老师讲韩语、现场翻译成中文；写了这种形式时以其中的翻译语言为准。"""
        if isinstance(data, dict) and "+" in str(data.get("teaching", "")):
            source, _, target = str(data["teaching"]).partition("+")
            data = {**data, "teaching": source, "interpreted_to": target}
        return data

    @field_validator("teaching")
    @classmethod
    def _check_teaching(cls, value: str) -> str:
        codes = parse_languages(value)
        if not codes:
            raise ValueError("至少要有一种授课语言")
        return ",".join(codes)

    @field_validator("interpreted_to")
    @classmethod
    def _check_single(cls, value: str) -> str:
        codes = parse_languages(value)
        if len(codes) > 1:
            raise ValueError("现场翻译只能选一种语言")
        return codes[0] if codes else ""

    @field_validator("note_language", mode="before")
    @classmethod
    def _always_chinese(cls, value: Any) -> str:
        return NOTE_LANGUAGE

    @property
    def languages(self) -> list[str]:
        """授课语言。"""
        return self.teaching.split(",")

    @property
    def spoken_languages(self) -> list[str]:
        """录音里会出现的全部语言：授课语言加上翻译语言。"""
        extra = [self.interpreted_to] if self.interpreted_to and self.interpreted_to not in self.languages else []
        return [*self.languages, *extra]

    @property
    def interpreted(self) -> bool:
        return bool(self.interpreted_to)

    @property
    def asr_language(self) -> str | None:
        """本地转写强制的语种：录音里只有一种语言时强制，否则自动识别。"""
        spoken = self.spoken_languages
        return spoken[0] if len(spoken) == 1 else None

    @property
    def teaching_description(self) -> str:
        names = "、".join(language_name(code) for code in self.languages)
        if self.interpreted:
            return f"老师用{names}授课，现场有人翻译成{language_name(self.interpreted_to)}"
        return f"{names}授课"

    @property
    def note_language_name(self) -> str:
        return language_name(self.note_language)

    @property
    def foreign_languages(self) -> list[str]:
        """中文以外的授课语言（如韩语授课时为 ["ko"]）：内容用原文理解，用中文总结。"""
        return [code for code in self.languages if code.split("-")[0] != NOTE_LANGUAGE]

    @property
    def term_rule(self) -> str:
        """提示词里知识点名称和术语的写法：一律用中文（M4 修订：分析结果和知识点分析必须全中文）。
        外语原词只放进知识点别名，供后台对照外语考题，不显示。"""
        rule = ("知识点名称和正文一律用中文书写，不要在括号里附任何外语原词——韩语、英语、拉丁语都不附，老师讲课时"
                "直接说的英语术语也一样（例如写「真菌病毒」，不写「真菌病毒（mycovirus）」）；只有 DNA、RNA、PCR、ATP "
                "这类通用的缩写和符号可以直接使用；人名、学名用通用的中文译名，没有通用译名的人名可以保留原文。")
        if not self.foreign_languages:
            return rule
        source = "、".join(language_name(code) for code in self.foreign_languages)
        return f"{rule}老师用{source}讲的术语要译成准确的中文。"

    @property
    def html_lang(self) -> str:
        """网页上笔记正文的 lang 属性。"""
        return "zh-CN"


class SettingsError(ValueError):
    """课程设置取值不对；消息是给使用者看的中文说明。"""


def build_settings(**fields: Any) -> CourseSettings:
    """创建课程设置；取值不对时抛出 SettingsError，只保留中文说明，不带 pydantic 的英文细节。"""
    try:
        return CourseSettings(**fields)
    except ValidationError as exc:
        messages = [str(error["msg"]).removeprefix("Value error, ") for error in exc.errors()]
        raise SettingsError("；".join(messages)) from None
