"""主模型拒答时自动换下一个档案重跑同一次请求。

Anthropic 有服务端的拒答回退（请求里带 fallbacks，由服务端换模型），但只在直连官方接口时可用：走中转站时
这个参数会被拒绝，所以中转站用户没有这一层。这里在客户端补上同样的效果，对任何接入方式都有效。

配置（config.toml）：

    [models.claude.options]
    fallback = "gpt"        # 被拒答时改用「gpt」这个档案；可以再往下配，最多三层

只有「模型自己拒绝作答」（RefusalError）才换模型；「输入被上游内容审核过滤」（ContentBlockedError）不换——
那是内容本身被拦，换模型多半也一样，转写另有「把音频切小再试」的处理。
"""

from __future__ import annotations

from typing import Any, Sequence, TypeVar

from pydantic import BaseModel

from .base import Effort, LLMProvider, LLMResult, Part, RefusalError

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class FallbackProvider(LLMProvider):
    """先问主模型；被拒答就换备用档案重跑，用量按两家之和记账，结果里带实际作答的档案。"""

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self.primary = primary
        self.fallback = fallback
        super().__init__(primary.profile)  # 对外仍是主档案：能力检查、档案名、用量记账都按主档案算

    def generate_text(
        self,
        parts: Sequence[Part],
        *,
        system: str | None = None,
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult[str]:
        try:
            return self.primary.generate_text(parts, system=system, effort=effort, max_tokens=max_tokens)
        except RefusalError as refused:
            result = self.fallback.generate_text(parts, system=system, effort=effort, max_tokens=max_tokens)
            return _merged(result, refused)

    def generate_structured(
        self,
        parts: Sequence[Part],
        schema: type[SchemaT],
        *,
        system: str | None = None,
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult[SchemaT]:
        try:
            return self.primary.generate_structured(
                parts, schema, system=system, effort=effort, max_tokens=max_tokens
            )
        except RefusalError as refused:
            result = self.fallback.generate_structured(
                parts, schema, system=system, effort=effort, max_tokens=max_tokens
            )
            return _merged(result, refused)

    def list_remote_models(self) -> list[str]:
        return self.primary.list_remote_models()

    def _complete(
        self,
        parts: Sequence[Part],
        *,
        system: str | None,
        schema: type[BaseModel] | None,
        effort: Effort | None,
        max_tokens: int,
    ) -> tuple[str, Any]:
        """只为满足基类的抽象方法：轮换发生在下面的 generate_* 里（那里能带上结构校验和用量合计）。"""
        try:
            return self.primary._complete(parts, system=system, schema=schema, effort=effort, max_tokens=max_tokens)
        except RefusalError as refused:
            text, usage = self.fallback._complete(parts, system=system, schema=schema, effort=effort,
                                                  max_tokens=max_tokens)
            return text, usage + refused.usage


def _merged(result: LLMResult[Any], refused: RefusalError) -> LLMResult[Any]:
    """备用档案的结果，加上被拒答那次已经消耗的输入。"""
    return LLMResult(result.output, result.usage + refused.usage, result.model, result.used_profile, True)
