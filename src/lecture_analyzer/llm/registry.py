"""适配器注册表：把模型档案（config.toml）变成可调用的 LLMProvider 实例。

新增一种接入方式：在 _ADAPTERS 里加一行"类型名 → 模块:类名"。
同一接入方式下的新模型：只改 config.toml，不用动代码。
"""

from __future__ import annotations

import importlib

from .base import LLMError, LLMProvider, ModelProfile

# 延迟导入：只有真正用到某家模型时才加载它的 SDK
_ADAPTERS: dict[str, str] = {
    "anthropic": "lecture_analyzer.llm.anthropic_adapter:AnthropicProvider",
    "gemini": "lecture_analyzer.llm.gemini_adapter:GeminiProvider",
    "openai_compat": "lecture_analyzer.llm.openai_compat:OpenAICompatProvider",
}


def adapter_types() -> list[str]:
    return list(_ADAPTERS)


def create_provider(profile: ModelProfile) -> LLMProvider:
    target = _ADAPTERS.get(profile.adapter)
    if target is None:
        raise LLMError(f"未知的适配器类型：{profile.adapter}（可选：{', '.join(_ADAPTERS)}）")
    module_name, class_name = target.split(":")
    provider_cls = getattr(importlib.import_module(module_name), class_name)
    return provider_cls(profile)
