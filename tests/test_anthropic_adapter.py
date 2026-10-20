from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from lecture_analyzer.llm import ImagePart, LLMError, TextPart, Usage
from lecture_analyzer.llm.anthropic_adapter import FALLBACK_BETA, AnthropicProvider

from .conftest import make_profile


class Answer(BaseModel):
    answer: str


def _message(stop_reason="end_turn", text='{"answer": "x"}'):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber"),
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=20, cache_read_input_tokens=50, cache_creation_input_tokens=10
        ),
    )


class StubStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class StubMessages:
    def __init__(self, message):
        self.message = message
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return StubStream(self.message)


def _provider(message=None, **profile_overrides):
    message = message or _message()
    client = SimpleNamespace(messages=StubMessages(message), beta=SimpleNamespace(messages=StubMessages(message)))
    values = {"adapter": "anthropic", "model": "claude-opus-5-5", "vision": True, "supports_effort": True}
    values.update(profile_overrides)
    return AnthropicProvider(make_profile(**values), client=client), client


def test_refusal_fallback_enabled_by_default():
    provider, client = _provider()
    provider.generate_text([TextPart("你好")])
    kwargs = client.beta.messages.calls[0]
    assert kwargs["betas"] == [FALLBACK_BETA] and kwargs["fallbacks"] == "default"
    assert client.messages.calls == []


def test_refusal_fallback_can_be_disabled():
    provider, client = _provider(options={"refusal_fallback": False})
    provider.generate_text([TextPart("你好")])
    assert "betas" not in client.messages.calls[0]


def test_refusal_fallback_skipped_for_relay():
    provider, client = _provider(base_url="https://relay.example/")
    provider.generate_text([TextPart("你好")])
    assert "fallbacks" not in client.messages.calls[0] and client.beta.messages.calls == []


def test_schema_and_effort_go_into_output_config():
    provider, client = _provider()
    result = provider.generate_structured([TextPart("问题")], Answer, system="系统", effort="medium")
    kwargs = client.beta.messages.calls[0]
    assert kwargs["system"] == "系统"
    assert kwargs["output_config"]["effort"] == "medium"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert "answer" in kwargs["output_config"]["format"]["schema"]["properties"]
    assert result.output.answer == "x"


def test_usage_includes_cached_input():
    provider, _ = _provider()
    assert provider.generate_text([TextPart("你好")]).usage == Usage(160, 20, 50)


def test_image_becomes_base64_block():
    provider, client = _provider()
    provider.generate_text([ImagePart(b"abc", "image/png")])
    block = client.beta.messages.calls[0]["messages"][0]["content"][0]
    assert block["source"] == {"type": "base64", "media_type": "image/png", "data": "YWJj"}


@pytest.mark.parametrize(("stop_reason", "pattern"), [("refusal", "拒绝"), ("max_tokens", "截断")])
def test_abnormal_stop_raises(stop_reason, pattern):
    provider, _ = _provider(_message(stop_reason=stop_reason))
    with pytest.raises(LLMError, match=pattern) as info:
        provider.generate_text([TextPart("你好")])
    assert info.value.usage.output_tokens == 20


def test_stream_interrupted_is_retried_and_api_errors_are_reported(monkeypatch):
    import anthropic
    import httpx

    from lecture_analyzer.llm import base

    monkeypatch.setattr(base, "RETRY_DELAYS", (0, 0))
    request = httpx.Request("POST", "https://relay.example/v1/messages")

    class Flaky(StubMessages):
        def __init__(self, message, failures):
            super().__init__(message)
            self.failures = failures

        def stream(self, **kwargs):
            self.calls.append(kwargs)
            if self.failures:
                raise self.failures.pop(0)
            return StubStream(self.message)

    dropped = anthropic.APIConnectionError(request=request)
    import httpx2

    # SDK 用 httpx2 发请求（M4 验收：按 httpx 的异常类判断漏掉了这个断线）
    cut_off = httpx2.RemoteProtocolError("peer closed connection without sending complete message body")
    flaky = Flaky(_message(), [dropped, cut_off])
    client = SimpleNamespace(messages=flaky, beta=SimpleNamespace(messages=flaky))
    provider = AnthropicProvider(make_profile(adapter="anthropic", model="claude-opus-5-5", base_url="https://relay.example"),
                                 client=client)
    assert provider.generate_structured([TextPart("问题")], Answer).output.answer == "x" and len(flaky.calls) == 3

    overloaded = anthropic.OverloadedError("Overloaded", response=httpx.Response(529, request=request), body=None)
    flaky.failures = [overloaded]  # 服务端过载：等一会儿重发
    flaky.calls.clear()
    assert provider.generate_text([TextPart("问题")]).output and len(flaky.calls) == 2

    invalid = anthropic.BadRequestError("Bad request", response=httpx.Response(400, request=request), body=None)
    flaky.failures = [invalid]
    flaky.calls.clear()
    with pytest.raises(LLMError, match="请求失败：BadRequestError"):
        provider.generate_text([TextPart("问题")])
    assert len(flaky.calls) == 1
