"""Claude 适配器：使用 Anthropic 官方 Python SDK。"""

from __future__ import annotations

import base64
from typing import Any, Sequence

import anthropic
import httpx2
from pydantic import BaseModel

from .base import Effort, ImagePart, LLMError, LLMProvider, ModelProfile, Part, TextPart, Usage, retry_transient

# 服务端拒答回退（Anthropic 推荐默认开启）：请求被安全分类器拒绝时，由服务端换模型重跑。
# 只在直连官方接口时启用；也可在模型档案里设置 options.refusal_fallback = false 手动关闭。
# 流式读到一半连接断开时，SDK 有时直接抛出 httpx2（SDK 用的 HTTP 库）的传输错误（实测 RemoteProtocolError），
# 一并按连接中断重试；anthropic 1.x 起不再用 httpx，按 httpx 的异常类判断会漏掉（M4 验收实测）
FALLBACK_BETA = "server-side-fallback-2026-07-01"
OFFICIAL_HOST = "api.anthropic.com"
TRANSIENT_ERRORS = (anthropic.APIConnectionError, httpx2.TransportError)


class AnthropicProvider(LLMProvider):
    def __init__(self, profile: ModelProfile, client: Any | None = None) -> None:
        super().__init__(profile)
        self._client = client or anthropic.Anthropic(
            api_key=profile.required_api_key(),
            base_url=profile.resolved_base_url(),
            timeout=anthropic.Timeout(profile.timeout_s, read=profile.stream_idle_s),  # 一律流式
            max_retries=0,  # 重试只在 retry_transient 里做一层
        )

    def _complete(
        self,
        parts: Sequence[Part],
        *,
        system: str | None,
        schema: type[BaseModel] | None,
        effort: Effort | None,
        max_tokens: int,
    ) -> tuple[str, Usage]:
        kwargs: dict[str, Any] = {
            "model": self.profile.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [_to_block(p) for p in parts]}],
        }
        if system:
            kwargs["system"] = system
        output_config: dict[str, Any] = {}
        if effort and self.profile.supports_effort:
            output_config["effort"] = effort
        if schema is not None:
            # 用原始 schema 形式而非 output_format，SDK 不做解析，统一交给基类校验和重试
            output_config["format"] = {"type": "json_schema", "schema": anthropic.transform_schema(schema)}
        if output_config:
            kwargs["output_config"] = output_config

        if self._use_refusal_fallback():
            api = self._client.beta.messages
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        else:
            api = self._client.messages

        # 输入输出都可能很长，统一走流式，避免 HTTP 超时；读到一半断开时整体重发
        try:
            message = retry_transient(lambda: _final_message(api, kwargs), TRANSIENT_ERRORS, self.profile.name)
        except anthropic.APIError as exc:
            raise LLMError(f"模型档案「{self.profile.name}」请求失败：{type(exc).__name__}: {str(exc)[:300]}") from exc

        usage = _to_usage(message.usage)
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) or "未知"
            raise LLMError(f"Claude 拒绝了这次请求（类别：{category}）", usage)
        if message.stop_reason == "max_tokens":
            raise LLMError(
                f"输出达到 max_tokens（{max_tokens}）被截断，请调大模型档案「{self.profile.name}」的 max_output",
                usage,
            )
        text = "".join(block.text for block in message.content if block.type == "text")
        return text, usage

    def list_remote_models(self) -> list[str]:
        return [model.id for model in self._client.models.list()]

    def _use_refusal_fallback(self) -> bool:
        """拒答回退是 Anthropic 官方接口的功能；走中转站等其他地址时自动关闭（实测中转站会拒绝该参数）。"""
        if not self.profile.options.get("refusal_fallback", True):
            return False
        base_url = self.profile.resolved_base_url()
        return base_url is None or OFFICIAL_HOST in base_url


def _final_message(api: Any, kwargs: dict[str, Any]) -> Any:
    with api.stream(**kwargs) as stream:
        return stream.get_final_message()


def _to_block(part: Part) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": part.mime_type,
                "data": base64.standard_b64encode(part.data).decode("ascii"),
            },
        }
    raise LLMError(f"Claude 适配器不支持 {type(part).__name__}")


def _to_usage(raw: Any) -> Usage:
    cache_read = getattr(raw, "cache_read_input_tokens", None) or 0
    cache_write = getattr(raw, "cache_creation_input_tokens", None) or 0
    return Usage(
        input_tokens=raw.input_tokens + cache_read + cache_write,
        output_tokens=raw.output_tokens,
        cache_read_tokens=cache_read,
    )
