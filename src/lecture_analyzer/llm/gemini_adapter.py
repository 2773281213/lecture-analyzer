"""Gemini 适配器：使用 Google 官方 google-genai SDK。"""

from __future__ import annotations

from typing import Any, Sequence

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from .base import (
    AudioPart,
    Effort,
    EmptyResponseError,
    ImagePart,
    LLMError,
    LLMProvider,
    ModelProfile,
    Part,
    TextPart,
    Usage,
)

# 统一的 effort 档位映射到 Gemini 的思考级别（仅在档案声明 supports_effort 时发送）
_THINKING_LEVEL = {
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
    "xhigh": types.ThinkingLevel.HIGH,
    "max": types.ThinkingLevel.HIGH,
}

_NORMAL_FINISH = {None, types.FinishReason.STOP, types.FinishReason.FINISH_REASON_UNSPECIFIED}
# 这些结束原因表示内容被审核拦截：同一输入重试通常仍会被拦，交给调用方换个办法（如切小再试）
_BLOCKED_FINISH = {
    types.FinishReason.SAFETY,
    types.FinishReason.RECITATION,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.SPII,
}


class GeminiProvider(LLMProvider):
    def __init__(self, profile: ModelProfile, client: Any | None = None) -> None:
        super().__init__(profile)
        http_options = types.HttpOptions(
            base_url=profile.resolved_base_url(),
            timeout=int(profile.timeout_s * 1000),
        )
        self._client = client or genai.Client(api_key=profile.required_api_key(), http_options=http_options)

    def _complete(
        self,
        parts: Sequence[Part],
        *,
        system: str | None,
        schema: type[BaseModel] | None,
        effort: Effort | None,
        max_tokens: int,
    ) -> tuple[str, Usage]:
        config: dict[str, Any] = {"max_output_tokens": max_tokens}
        if system:
            config["system_instruction"] = system
        if schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = schema
        if effort and self.profile.supports_effort:
            config["thinking_config"] = types.ThinkingConfig(thinking_level=_THINKING_LEVEL[effort])

        try:
            response = self._client.models.generate_content(
                model=self.profile.model,
                contents=[_to_part(p) for p in parts],
                config=types.GenerateContentConfig(**config),
            )
        except errors.APIError as exc:
            raise LLMError(f"模型档案「{self.profile.name}」请求失败：{type(exc).__name__}: {str(exc)[:300]}") from exc
        usage = _to_usage(response.usage_metadata)
        if not response.candidates:
            feedback = getattr(response, "prompt_feedback", None)
            reason = getattr(feedback, "block_reason", None) or "未知"
            raise EmptyResponseError(f"Gemini 没有返回结果（拦截原因：{reason}）", usage)
        finish = response.candidates[0].finish_reason
        if finish == types.FinishReason.MAX_TOKENS:
            raise LLMError(
                f"输出达到 max_output_tokens（{max_tokens}）被截断，请调大模型档案「{self.profile.name}」的 max_output",
                usage,
            )
        if finish in _BLOCKED_FINISH:
            raise EmptyResponseError(f"Gemini 拦截了这次输出（finish_reason：{finish}）", usage)
        if finish not in _NORMAL_FINISH:
            raise LLMError(f"Gemini 未正常结束（finish_reason：{finish}）", usage)
        return response.text or "", usage

    def list_remote_models(self) -> list[str]:
        names = []
        for model in self._client.models.list():
            if "generateContent" in (model.supported_actions or []):
                names.append((model.name or "").removeprefix("models/"))
        return names


def _to_part(part: Part) -> types.Part:
    if isinstance(part, TextPart):
        return types.Part.from_text(text=part.text)
    if isinstance(part, (ImagePart, AudioPart)):
        return types.Part.from_bytes(data=part.data, mime_type=part.mime_type)
    raise LLMError(f"Gemini 适配器不支持 {type(part).__name__}")


def _to_usage(meta: Any) -> Usage:
    if meta is None:
        return Usage()
    output = (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
    return Usage(
        input_tokens=meta.prompt_token_count or 0,
        output_tokens=output,
        cache_read_tokens=meta.cached_content_token_count or 0,
    )
