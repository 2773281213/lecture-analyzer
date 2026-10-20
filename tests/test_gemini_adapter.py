from types import SimpleNamespace

import pytest
from google.genai import types
from pydantic import BaseModel

from lecture_analyzer.llm import AudioPart, EmptyResponseError, LLMError, TextPart, Usage
from lecture_analyzer.llm.gemini_adapter import GeminiProvider

from .conftest import make_profile


class Answer(BaseModel):
    answer: str


def _response(finish=types.FinishReason.STOP, candidates=True):
    usage = SimpleNamespace(
        prompt_token_count=100, candidates_token_count=10, thoughts_token_count=5, cached_content_token_count=20
    )
    return SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=finish)] if candidates else [],
        text='{"answer": "x"}',
        usage_metadata=usage,
        prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
    )


def _provider(response=None, **profile_overrides):
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return response or _response()

    values = {"adapter": "gemini", "model": "gemini-test", "audio": True, "vision": True}
    values.update(profile_overrides)
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return GeminiProvider(make_profile(**values), client=client), calls


def test_structured_request_config():
    provider, calls = _provider()
    result = provider.generate_structured([TextPart("问题")], Answer, system="系统", effort="high")
    config = calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is Answer
    assert config.system_instruction == "系统"
    assert config.thinking_config is None  # 档案未声明 supports_effort
    assert result.output.answer == "x"


def test_thinking_level_sent_when_supported():
    provider, calls = _provider(supports_effort=True)
    provider.generate_text([TextPart("问题")], effort="medium")
    assert calls[0]["config"].thinking_config.thinking_level == types.ThinkingLevel.MEDIUM


def test_audio_part_is_inline_bytes():
    provider, calls = _provider()
    provider.generate_text([AudioPart(b"flac-bytes", "audio/flac")])
    part = calls[0]["contents"][0]
    assert part.inline_data.mime_type == "audio/flac" and part.inline_data.data == b"flac-bytes"


def test_usage_counts_thoughts_as_output():
    provider, _ = _provider()
    assert provider.generate_text([TextPart("问题")]).usage == Usage(100, 15, 20)


@pytest.mark.parametrize(
    ("response", "error", "pattern"),
    [
        (_response(finish=types.FinishReason.MAX_TOKENS), LLMError, "截断"),
        (_response(finish=types.FinishReason.SAFETY), EmptyResponseError, "拦截了这次输出"),
        (_response(finish=types.FinishReason.PROHIBITED_CONTENT), EmptyResponseError, "拦截了这次输出"),
        (_response(finish=types.FinishReason.OTHER), LLMError, "未正常结束"),
        (_response(candidates=False), EmptyResponseError, "没有返回结果"),
    ],
)
def test_abnormal_finish_raises(response, error, pattern):
    provider, _ = _provider(response)
    with pytest.raises(error, match=pattern):
        provider.generate_text([TextPart("问题")])
