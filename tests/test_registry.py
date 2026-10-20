import pytest

from lecture_analyzer.llm import LLMError, ModelProfile, adapter_types, create_provider
from lecture_analyzer.llm.anthropic_adapter import AnthropicProvider
from lecture_analyzer.llm.gemini_adapter import GeminiProvider
from lecture_analyzer.llm.openai_compat import OpenAICompatProvider

from .conftest import make_profile


@pytest.mark.parametrize(
    ("adapter", "expected"),
    [("anthropic", AnthropicProvider), ("gemini", GeminiProvider), ("openai_compat", OpenAICompatProvider)],
)
def test_create_provider_by_adapter_type(monkeypatch, adapter, expected):
    monkeypatch.setenv("LEC_TEST_KEY", "test-key")
    profile = make_profile(adapter=adapter, api_key_env="LEC_TEST_KEY", base_url="https://example.invalid/v1")
    assert isinstance(create_provider(profile), expected)


@pytest.mark.parametrize("adapter", ["anthropic", "openai_compat"])
def test_adapters_use_base_url_from_env(monkeypatch, adapter):
    monkeypatch.setenv("LEC_TEST_KEY", "test-key")
    monkeypatch.setenv("LEC_TEST_BASE_URL", "https://relay.example/v1")
    profile = make_profile(adapter=adapter, api_key_env="LEC_TEST_KEY", base_url_env="LEC_TEST_BASE_URL")
    assert "relay.example" in str(create_provider(profile)._client.base_url)


def test_all_adapter_types_registered():
    assert set(adapter_types()) == {"anthropic", "gemini", "openai_compat"}


def test_unknown_adapter_type_rejected():
    profile = ModelProfile.model_construct(name="x", adapter="unknown", model="m")
    with pytest.raises(LLMError, match="未知的适配器类型"):
        create_provider(profile)
