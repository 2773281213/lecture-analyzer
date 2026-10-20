import pytest
from pydantic import BaseModel

from lecture_analyzer.llm import AudioPart, ImagePart, LLMError, StructuredOutputError, TextPart, Usage
from lecture_analyzer.llm.base import extract_json_text, schema_instruction

from .conftest import FakeProvider, make_profile


class Answer(BaseModel):
    answer: str
    score: int


def test_extract_json_text_strips_code_fence_and_noise():
    text = '好的，结果如下：\n```json\n{"answer": "x", "score": 1}\n```\n以上。'
    assert extract_json_text(text) == '{"answer": "x", "score": 1}'


def test_extract_json_text_without_json_raises():
    with pytest.raises(ValueError):
        extract_json_text("没有任何结构")


def test_schema_instruction_mentions_json_and_fields():
    text = schema_instruction(Answer)
    assert "JSON" in text and "answer" in text and "score" in text


def test_generate_structured_success():
    provider = FakeProvider(['{"answer": "x", "score": 2}'])
    result = provider.generate_structured([TextPart("问题")], Answer, system="系统", effort="low")
    assert result.output == Answer(answer="x", score=2)
    assert result.usage == Usage(10, 5)
    assert provider.calls[0]["schema"] is Answer
    assert provider.calls[0]["max_tokens"] == provider.profile.max_output


def test_generate_structured_retries_once_with_error_hint():
    provider = FakeProvider(['{"answer": "x"}', '{"answer": "x", "score": 3}'])
    result = provider.generate_structured([TextPart("问题")], Answer)
    assert result.output.score == 3
    assert result.usage == Usage(20, 10)
    retry_parts = provider.calls[1]["parts"]
    assert len(retry_parts) == 2 and "没有通过格式校验" in retry_parts[-1].text


def test_generate_structured_gives_up_after_two_failures():
    provider = FakeProvider(["不是 JSON", '{"answer": 1}'])
    with pytest.raises(StructuredOutputError) as info:
        provider.generate_structured([TextPart("问题")], Answer)
    assert info.value.usage == Usage(20, 10)


def test_validation_error_raised_inside_adapter_is_retried():
    provider = FakeProvider([ValueError("SDK 自带校验失败"), '{"answer": "y", "score": 1}'])
    assert provider.generate_structured([TextPart("问题")], Answer).output.answer == "y"


def test_llm_error_is_not_retried():
    provider = FakeProvider([LLMError("输出被截断"), '{"answer": "y", "score": 1}'])
    with pytest.raises(LLMError, match="截断"):
        provider.generate_structured([TextPart("问题")], Answer)
    assert len(provider.calls) == 1


def test_image_rejected_without_vision():
    provider = FakeProvider(["{}"], make_profile(vision=False))
    with pytest.raises(LLMError, match="不支持图片"):
        provider.generate_text([ImagePart(b"png")])


def test_audio_rejected_without_audio():
    provider = FakeProvider(["{}"], make_profile(audio=False))
    with pytest.raises(LLMError, match="不支持音频"):
        provider.generate_text([AudioPart(b"flac")])


def test_unconfigured_profile_rejected():
    with pytest.raises(LLMError, match="没有填写 model"):
        FakeProvider([], make_profile(model=""))


def test_estimate_cost_counts_cache_reads_at_cache_price():
    profile = make_profile(price_in=4.0, price_out=20.0, price_cache_read=0.2)
    cost = profile.estimate_cost(Usage(input_tokens=1_000_000, output_tokens=100_000, cache_read_tokens=500_000))
    assert cost == pytest.approx(500_000 * 4.0 / 1e6 + 500_000 * 0.2 / 1e6 + 100_000 * 20.0 / 1e6)


def test_estimate_cost_without_prices_is_none():
    assert make_profile().estimate_cost(Usage(100, 100)) is None


def test_base_url_from_env_overrides_profile(monkeypatch):
    profile = make_profile(base_url="https://official.example/v1", base_url_env="LEC_TEST_BASE_URL")
    monkeypatch.setenv("LEC_TEST_BASE_URL", " https://relay.example/v1 ")
    assert profile.resolved_base_url() == "https://relay.example/v1"
    monkeypatch.setenv("LEC_TEST_BASE_URL", "")  # 变量留空时退回档案里的地址
    assert profile.resolved_base_url() == "https://official.example/v1"
    assert make_profile().resolved_base_url() is None  # 都没有则直连官方
