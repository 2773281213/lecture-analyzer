"""OpenAI 兼容适配器：GPT，以及提供 OpenAI 兼容接口的国产模型（DeepSeek、通义千问、Kimi、GLM 等）。

各家对结构化输出的支持不同，由模型档案的 structured 字段决定请求方式：
- json_schema：严格 JSON Schema（GPT 等支持的模型），由服务端按 schema 约束输出
- json_object：只保证输出合法 JSON（如 DeepSeek、多数通义千问模型），schema 写进系统提示词
- prompt：不发送 response_format，schema 只写在提示词里
三种方式的结果都由基类统一做 Pydantic 校验（兼容 ```json 代码块和前后多余文字），不合格时重试一次。
不用 SDK 的 parse / stream 辅助方法：它们按 schema 严格解析，输出不是纯 JSON 时整次失败，
流式时在第一个字符就报错，拿不到实际返回了什么。

要求了 JSON（json_schema、json_object）却连一个 JSON 对象都没有时，这段文字不是模型按格式生成的结果，
而是中转站代替结果返回的说明，按拦截处理（EmptyResponseError）。M4 验收实测：Gemini 内容过滤拦截
生物学内容时，中转站返回一段英文说明，finish_reason 为 stop，没有用量。

档案选项 options.stream = true 时改用流式请求：服务端边生成边返回，长输出不会因为中转站
（例如 Cloudflare 的 100–120 秒等待上限）迟迟收不到响应而断开（524）。数据块由这里自己累积，
用量通过 stream_options.include_usage 取得；多久收不到任何数据算卡住见 ModelProfile.stream_idle_s。
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any, Sequence

import httpx2
import openai
from openai.lib._parsing._completions import type_to_response_format_param  # 与 SDK 的 parse 辅助方法相同的请求格式
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
    retry_transient,
    schema_instruction,
)

# 连接错误（含超时）；流式读到一半断开时 SDK 可能直接抛出 httpx2（SDK 用的 HTTP 库）的传输错误
TRANSIENT_ERRORS = (openai.APIConnectionError, httpx2.TransportError)
# 统一的 effort 档位映射到 OpenAI 的 reasoning_effort（仅在档案声明 supports_effort 时发送）
_REASONING_EFFORT = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


class OpenAICompatProvider(LLMProvider):
    def __init__(self, profile: ModelProfile, client: Any | None = None) -> None:
        super().__init__(profile)
        if client is None:
            read = profile.stream_idle_s if profile.options.get("stream", False) else profile.timeout_s
            client = openai.OpenAI(api_key=profile.required_api_key(), base_url=profile.resolved_base_url(),
                                   timeout=openai.Timeout(profile.timeout_s, read=read),
                                   max_retries=0)  # 重试只在 retry_transient 里做一层
        self._client = client

    def _complete(
        self,
        parts: Sequence[Part],
        *,
        system: str | None,
        schema: type[BaseModel] | None,
        effort: Effort | None,
        max_tokens: int,
    ) -> tuple[str, Usage]:
        kwargs = self._request_kwargs(parts, system, schema, effort, max_tokens)
        completion = self._request(kwargs)
        if not completion.choices:  # 中转站或上游偶发返回空结果（已实测），重试一次
            completion = self._request(kwargs)

        usage = _to_usage(completion)
        if not completion.choices:
            detail = getattr(completion, "model_extra", None) or {}
            raise EmptyResponseError(
                f"模型档案「{self.profile.name}」没有返回任何结果（附加信息：{str(detail)[:300]}）", usage
            )
        content = self._content(completion.choices[0], usage, max_tokens)
        if "response_format" in kwargs and "{" not in content:
            raise EmptyResponseError(
                f"模型档案「{self.profile.name}」没有按要求返回 JSON，多半是上游的内容审核拦截了这次输入"
                f"（返回内容：{content[:200]}）", usage
            )
        return content, usage

    def _request_kwargs(
        self,
        parts: Sequence[Part],
        system: str | None,
        schema: type[BaseModel] | None,
        effort: Effort | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        """组装请求参数；只有服务端会约束输出格式（json_schema、json_object）时才带 response_format。"""
        mode = self.profile.structured
        strict_schema = schema is not None and mode in ("json_schema", "native")
        system_text = system or ""
        if schema is not None and not strict_schema:
            system_text = f"{system_text}\n\n{schema_instruction(schema)}".strip()

        messages: list[dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": _to_content(parts)})

        # 新的 OpenAI 推理模型要求用 max_completion_tokens，多数兼容接口仍用 max_tokens
        token_param = self.profile.options.get("max_tokens_param", "max_tokens")
        kwargs: dict[str, Any] = {"model": self.profile.model, "messages": messages, token_param: max_tokens}
        if effort and self.profile.supports_effort:
            kwargs["reasoning_effort"] = _REASONING_EFFORT[effort]

        if schema is not None and mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        elif strict_schema:
            kwargs["response_format"] = type_to_response_format_param(schema)
        return kwargs

    def _content(self, choice: Any, usage: Usage, max_tokens: int) -> str:
        """检查结束原因和拒绝，返回正文。"""
        if choice.finish_reason == "length":
            raise LLMError(self._truncated_message(max_tokens), usage)
        if choice.finish_reason == "content_filter":
            raise LLMError(f"模型档案「{self.profile.name}」的输出被内容过滤拦截", usage)
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise LLMError(f"模型拒绝了这次请求：{refusal}", usage)
        return choice.message.content or ""

    def list_remote_models(self) -> list[str]:
        return [model.id for model in self._client.models.list()]

    def _request(self, kwargs: dict[str, Any]) -> Any:
        """发一次请求，临时错误（连接错误、超时、服务端暂时不可用）等一会儿整体重发，
        其他接口错误转成 LLMError（截断、内容过滤等结束原因由 _content 判断）。"""
        if self.profile.options.get("stream", False):
            send = lambda: self._stream(kwargs)  # noqa: E731
        else:
            send = lambda: self._client.chat.completions.create(**kwargs)  # noqa: E731
        try:
            return retry_transient(send, TRANSIENT_ERRORS, self.profile.name)
        except openai.APIError as exc:
            raise LLMError(f"模型档案「{self.profile.name}」请求失败：{type(exc).__name__}: {str(exc)[:300]}") from exc

    def _stream(self, kwargs: dict[str, Any]) -> Any:
        """流式请求，读完后返回与非流式相同形状的结果（choices[0].message、finish_reason、usage）。"""
        options = {**kwargs, "stream_options": {"include_usage": True}}
        with self._client.chat.completions.create(stream=True, **options) as stream:
            return _accumulate(stream)

    def _truncated_message(self, max_tokens: int) -> str:
        return f"输出达到上限（{max_tokens} token）被截断，请调大模型档案「{self.profile.name}」的 max_output"


def _accumulate(chunks: Any) -> Any:
    """把流式返回的数据块拼成完整结果：只取第一个候选；用量在最后一个数据块里。"""
    choices, usage = _first_choices(chunks)
    if not choices:
        return SimpleNamespace(choices=[], usage=usage)
    finish_reason = next((choice.finish_reason for choice in reversed(choices) if choice.finish_reason), None)
    message = SimpleNamespace(content=_join(choices, "content"), refusal=_join(choices, "refusal") or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=usage)


def _first_choices(chunks: Any) -> tuple[list[Any], Any]:
    choices: list[Any] = []
    usage = None
    for chunk in chunks:
        usage = getattr(chunk, "usage", None) or usage
        choices += [choice for choice in chunk.choices or [] if choice.index == 0]
    return choices, usage


def _join(choices: Sequence[Any], name: str) -> str:
    return "".join(getattr(choice.delta, name, None) or "" for choice in choices)


def _to_content(parts: Sequence[Part]) -> str | list[dict[str, Any]]:
    """纯文本时拼成字符串（部分兼容接口不接受数组格式）；带图片时用多模态数组。"""
    if all(isinstance(p, TextPart) for p in parts):
        return "\n\n".join(p.text for p in parts)  # type: ignore[union-attr]
    content: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            data = base64.standard_b64encode(part.data).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{part.mime_type};base64,{data}"}})
        elif isinstance(part, AudioPart):
            content.append({"type": "input_audio", "input_audio": {
                "data": base64.standard_b64encode(part.data).decode("ascii"),
                "format": _audio_format(part.mime_type),
            }})
        else:
            raise LLMError(f"OpenAI 兼容适配器不支持 {type(part).__name__}")
    return content


_AUDIO_FORMATS = {"audio/mp3": "mp3", "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav"}


def _audio_format(mime_type: str) -> str:
    """OpenAI 格式的音频输入只接受 mp3 和 wav。"""
    try:
        return _AUDIO_FORMATS[mime_type]
    except KeyError:
        raise LLMError(f"OpenAI 兼容接口不支持 {mime_type} 音频，请转成 mp3 或 wav") from None


def _to_usage(completion: Any) -> Usage:
    raw = getattr(completion, "usage", None)
    if raw is None:
        return Usage()
    details = getattr(raw, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None)
    if cached is None:
        cached = getattr(raw, "prompt_cache_hit_tokens", None)  # DeepSeek 的字段名
    return Usage(
        input_tokens=raw.prompt_tokens or 0,
        output_tokens=raw.completion_tokens or 0,
        cache_read_tokens=cached or 0,
    )
