"""拒答自动轮换：主模型拒绝作答时改用备用档案重跑同一次请求。"""

import pytest
from pydantic import BaseModel

from lecture_analyzer.llm import EmptyResponseError, FallbackProvider, RefusalError, TextPart, Usage
from lecture_analyzer.llm import registry

from .conftest import FakeProvider, make_profile


class Answer(BaseModel):
    answer: str


def _providers(*primary_responses, **backup_responses):
    primary = FakeProvider(list(primary_responses), make_profile(name="claude", model="claude-x"))
    backup = FakeProvider(list(backup_responses.get("responses", ['{"answer": "备用"}'])), make_profile(name="gpt"))
    return FallbackProvider(primary, backup), primary, backup


def test_refusal_switches_to_fallback_and_sums_usage():
    provider, primary, backup = _providers(RefusalError("Claude 拒绝了这次请求（类别：bio）", Usage(100, 0), "bio"))
    result = provider.generate_structured([TextPart("问题")], Answer, system="系统", effort="medium")
    assert result.output.answer == "备用"
    assert result.usage == Usage(110, 5)  # 被拒答那次已经消耗的 100 也计入
    assert result.used_profile.name == "gpt" and result.model == "fake-model"
    assert len(primary.calls) == 1 and len(backup.calls) == 1
    # 换模型重跑的是同一次请求：输入、系统提示词、schema 都一样
    assert backup.calls[0]["parts"] == primary.calls[0]["parts"]
    assert backup.calls[0]["system"] == "系统" and backup.calls[0]["schema"] is Answer


def test_normal_answer_does_not_touch_fallback():
    provider, primary, backup = _providers('{"answer": "主模型"}')
    result = provider.generate_structured([TextPart("问题")], Answer)
    assert result.output.answer == "主模型"
    assert result.used_profile.name == "claude" and result.usage == Usage(10, 5)
    assert len(backup.calls) == 0


def test_generate_text_also_rotates():
    provider, _, backup = _providers(RefusalError("拒绝", Usage(7, 0)), responses=["备用的正文"])
    result = provider.generate_text([TextPart("问题")])
    assert result.output == "备用的正文" and result.usage == Usage(17, 5)
    assert result.used_profile.name == "gpt" and len(backup.calls) == 1


def test_both_refuse_reports_the_last_refusal():
    provider, _, backup = _providers(RefusalError("主模型拒绝", Usage(5, 0)), responses=[RefusalError("备用也拒绝")])
    with pytest.raises(RefusalError, match="备用也拒绝"):
        provider.generate_structured([TextPart("问题")], Answer)


def test_input_blocked_is_not_rotated():
    """输入被上游内容审核过滤（ContentBlockedError）不换模型：转写另有「切小再试」的处理，不能被抢走。"""
    from lecture_analyzer.llm import ContentBlockedError

    provider, _, backup = _providers(ContentBlockedError("被过滤"))
    with pytest.raises(ContentBlockedError):
        provider.generate_structured([TextPart("问题")], Answer)
    assert len(backup.calls) == 0


def test_refusal_error_is_an_empty_response_error():
    """转写的「被拦截就切小」按 EmptyResponseError 判断，拒答必须仍然属于这一类。"""
    assert issubclass(RefusalError, EmptyResponseError)


def test_registry_wraps_provider_when_fallback_configured(monkeypatch):
    monkeypatch.setattr(registry, "_build", lambda profile: FakeProvider([], profile))
    backup = make_profile(name="gpt", model="gpt-x")
    primary = make_profile(name="claude", model="claude-x", fallback_profile=backup)
    provider = registry.create_provider(primary)
    assert isinstance(provider, FallbackProvider)
    assert provider.profile.name == "claude"  # 对外仍是主档案
    assert provider.fallback.profile.name == "gpt"


def test_registry_returns_plain_provider_without_fallback(monkeypatch):
    monkeypatch.setattr(registry, "_build", lambda profile: FakeProvider([], profile))
    provider = registry.create_provider(make_profile())
    assert not isinstance(provider, FallbackProvider)
