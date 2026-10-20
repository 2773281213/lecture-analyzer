from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from lecture_analyzer.llm import (
    AudioPart, EmptyResponseError, ImagePart, LLMError, StructuredOutputError, TextPart, Usage,
)
from lecture_analyzer.llm.openai_compat import OpenAICompatProvider

from .conftest import make_profile


class Answer(BaseModel):
    answer: str


def _completion(content='{"answer": "x"}', finish="stop", usage=None):
    usage = usage or SimpleNamespace(
        prompt_tokens=100, completion_tokens=20, prompt_tokens_details=SimpleNamespace(cached_tokens=30)
    )
    message = SimpleNamespace(content=content, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=message)], usage=usage)


class StubCompletions:
    """按顺序返回预设响应；只给一个时每次都返回它。"""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def _next(self):
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return self._next()


def _provider(response=None, *more_responses, **profile_overrides):
    completions = StubCompletions(response or _completion(), *more_responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAICompatProvider(make_profile(**profile_overrides), client=client), completions


def test_json_object_mode_puts_schema_in_system_prompt():
    provider, completions = _provider(structured="json_object")
    result = provider.generate_structured([TextPart("问题")], Answer, system="你是助教")
    method, kwargs = completions.calls[0]
    assert method == "create"
    assert kwargs["response_format"] == {"type": "json_object"}
    system = kwargs["messages"][0]
    assert system["role"] == "system" and system["content"].startswith("你是助教") and "JSON Schema" in system["content"]
    assert kwargs["messages"][1]["content"] == "问题"  # 纯文本拼成字符串
    assert result.output.answer == "x"


def test_json_schema_mode_sends_strict_schema_without_schema_prompt():
    provider, completions = _provider(_completion('```json\n{"answer": "x"}\n```'), structured="json_schema")
    result = provider.generate_structured([TextPart("问题")], Answer)
    method, kwargs = completions.calls[0]
    assert method == "create"
    response_format = kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Answer" and response_format["json_schema"]["strict"] is True
    assert all(m["role"] != "system" for m in kwargs["messages"])
    assert result.output.answer == "x"  # 代码块由基类剥掉，不会因为 SDK 的严格解析整次失败


BLOCK_NOTICE = "This request was blocked by Gemini's filters. Please try rephrasing your prompt."


@pytest.mark.parametrize("structured", ["json_schema", "json_object"])
def test_block_notice_instead_of_json_counts_as_blocked(structured):
    # M4 验收实测：中转站用一段说明代替结果（finish_reason 为 stop，没有用量），按拦截处理，不再带修正提示重试
    provider, completions = _provider(_completion(BLOCK_NOTICE), structured=structured)
    with pytest.raises(EmptyResponseError, match="内容审核拦截.*blocked by Gemini"):
        provider.generate_structured([TextPart("问题")], Answer)
    assert len(completions.calls) == 1


def test_prose_in_prompt_mode_still_gets_repair_retry():
    provider, completions = _provider(_completion("好的，下面是答案"), structured="prompt")
    with pytest.raises(StructuredOutputError):
        provider.generate_structured([TextPart("问题")], Answer)
    assert len(completions.calls) == 2
    provider, _ = _provider(_completion(BLOCK_NOTICE))
    assert provider.generate_text([TextPart("问题")]).output == BLOCK_NOTICE  # 纯文本请求原样返回


def test_prompt_mode_sends_no_response_format():
    provider, completions = _provider(structured="prompt")
    provider.generate_structured([TextPart("问题")], Answer)
    _, kwargs = completions.calls[0]
    assert "response_format" not in kwargs
    assert "JSON Schema" in kwargs["messages"][0]["content"]


def test_image_parts_become_data_uri():
    provider, completions = _provider(vision=True)
    provider.generate_text([TextPart("看图"), ImagePart(b"abc", "image/png")])
    content = completions.calls[0][1]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "看图"}
    assert content[1]["image_url"]["url"] == "data:image/png;base64,YWJj"


def test_audio_parts_become_input_audio():
    provider, completions = _provider(audio=True)
    provider.generate_text([AudioPart(b"abc", "audio/mp3"), TextPart("转写")])
    content = completions.calls[0][1]["messages"][-1]["content"]
    assert content[0] == {"type": "input_audio", "input_audio": {"data": "YWJj", "format": "mp3"}}


def test_unsupported_audio_format_rejected():
    provider, _ = _provider(audio=True)
    with pytest.raises(LLMError, match="mp3 或 wav"):
        provider.generate_text([AudioPart(b"abc", "audio/flac")])


def test_empty_choices_are_retried_once():
    empty = SimpleNamespace(choices=[], usage=None, model_extra={})
    provider, completions = _provider(empty, _completion(content="恢复了"))
    assert provider.generate_text([TextPart("问题")]).output == "恢复了"
    assert len(completions.calls) == 2


def test_empty_choices_twice_raise_llm_error():
    empty = SimpleNamespace(choices=[], usage=None, model_extra={"error": "blocked"})
    provider, completions = _provider(empty)
    with pytest.raises(LLMError, match="没有返回任何结果.*blocked"):
        provider.generate_text([TextPart("问题")])
    assert len(completions.calls) == 2


def test_length_finish_reason_raises():
    provider, _ = _provider(_completion(finish="length"))
    with pytest.raises(LLMError, match="截断"):
        provider.generate_text([TextPart("问题")])


def test_token_param_and_effort_follow_profile():
    provider, completions = _provider(supports_effort=True, options={"max_tokens_param": "max_completion_tokens"})
    provider.generate_text([TextPart("问题")], effort="xhigh", max_tokens=123)
    _, kwargs = completions.calls[0]
    assert kwargs["max_completion_tokens"] == 123 and "max_tokens" not in kwargs
    assert kwargs["reasoning_effort"] == "high"


def test_effort_not_sent_when_unsupported():
    provider, completions = _provider()
    provider.generate_text([TextPart("问题")], effort="high")
    assert "reasoning_effort" not in completions.calls[0][1]


def test_usage_maps_openai_and_deepseek_cache_fields():
    provider, _ = _provider()
    assert provider.generate_text([TextPart("问题")]).usage == Usage(100, 20, 30)
    deepseek_usage = SimpleNamespace(prompt_tokens=50, completion_tokens=5, prompt_cache_hit_tokens=40)
    provider, _ = _provider(_completion(usage=deepseek_usage))
    assert provider.generate_text([TextPart("问题")]).usage == Usage(50, 5, 40)


@pytest.mark.parametrize("adapter", ["openai_compat", "anthropic", "gemini"])
def test_missing_api_key_is_reported(monkeypatch, adapter):
    # M4 安装包验收：新书库没有填密钥时，任务页只显示 Anthropic SDK 的原始报错，看不出该去哪里填
    from lecture_analyzer.llm import create_provider

    monkeypatch.delenv("LEC_TEST_MISSING_KEY", raising=False)
    with pytest.raises(LLMError, match="缺少密钥：请在设置页填写（命令行：在 .env 里设置 LEC_TEST_MISSING_KEY）"):
        create_provider(make_profile(adapter=adapter, api_key_env="LEC_TEST_MISSING_KEY"))


# ---------- 流式请求（options.stream = true） ----------


class _Stream:
    """模拟 SDK 返回的流：可以 with，可以迭代。"""

    def __init__(self, items):
        self.items = list(items)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for item in self.items:
            if isinstance(item, Exception):
                raise item
            yield item


def _chunk(content=None, finish=None, usage=None, index=0):
    choices = [] if content is None and finish is None else [
        SimpleNamespace(index=index, delta=SimpleNamespace(content=content, refusal=None), finish_reason=finish)
    ]
    return SimpleNamespace(choices=choices, usage=usage)


def _streaming_provider(*responses, **profile_overrides):
    completions = StubCompletions(*responses)
    profile = make_profile(options={"stream": True}, **profile_overrides)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAICompatProvider(profile, client=client), completions


def test_streaming_accumulates_chunks_and_usage():
    usage = SimpleNamespace(prompt_tokens=50, completion_tokens=7, prompt_tokens_details=None)
    chunks = [_chunk('{"ans'), _chunk("其他候选", index=1), _chunk('wer": "流式"}', finish="stop"), _chunk(usage=usage)]
    provider, completions = _streaming_provider(_Stream(chunks), structured="json_object")
    result = provider.generate_structured([TextPart("问题")], Answer)
    method, kwargs = completions.calls[0]
    assert method == "create" and kwargs["stream"] is True and kwargs["stream_options"] == {"include_usage": True}
    assert result.output.answer == "流式" and result.usage == Usage(50, 7, 0)


def test_streaming_strict_schema_tolerates_code_fence():
    # M4 验收实测：模型在 JSON 前面套上 ```json，SDK 的 stream 辅助方法逐块解析时在第一个字符就报错
    chunks = [_chunk('```json\n{"ans'), _chunk('wer": "y"}\n```', finish="stop")]
    provider, completions = _streaming_provider(_Stream(chunks), structured="json_schema")
    assert provider.generate_structured([TextPart("问题")], Answer).output.answer == "y"
    method, kwargs = completions.calls[0]
    assert method == "create" and kwargs["stream"] is True
    assert kwargs["response_format"]["json_schema"]["name"] == "Answer"


def test_streaming_truncation_and_empty_stream():
    provider, _ = _streaming_provider(_Stream([_chunk('{"answer": "很长', finish="length")]))
    with pytest.raises(LLMError, match="截断"):
        provider.generate_text([TextPart("问题")])
    provider, completions = _streaming_provider(_Stream([_chunk(usage=None)]))
    with pytest.raises(LLMError, match="没有返回任何结果"):
        provider.generate_text([TextPart("问题")])
    assert len(completions.calls) == 2  # 空结果重试过一次


def test_streaming_block_notice_counts_as_blocked():
    chunks = [_chunk(BLOCK_NOTICE), _chunk(finish="stop")]  # 实测的数据块：没有用量
    provider, completions = _streaming_provider(_Stream(chunks), structured="json_schema")
    with pytest.raises(EmptyResponseError, match="blocked by Gemini"):
        provider.generate_structured([TextPart("问题")], Answer)
    assert len(completions.calls) == 1


def test_streaming_strict_schema_truncation_and_empty_stream():
    provider, _ = _streaming_provider(_Stream([_chunk('{"answer": "很长', finish="length")]), structured="json_schema")
    with pytest.raises(LLMError, match="截断"):
        provider.generate_structured([TextPart("问题")], Answer)
    provider, _ = _streaming_provider(_Stream([]), structured="json_schema")  # 一个数据块都没有
    with pytest.raises(LLMError, match="没有返回任何结果"):
        provider.generate_structured([TextPart("问题")], Answer)


def test_stream_interrupted_is_retried_then_reported(monkeypatch):
    import httpx
    import openai

    from lecture_analyzer.llm import base

    monkeypatch.setattr(base, "RETRY_DELAYS", (0, 0))
    dropped = openai.APIConnectionError(request=httpx.Request("POST", "https://relay.example/v1"))
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=2, prompt_tokens_details=None)
    good = _Stream([_chunk("恢复了", finish="stop"), _chunk(usage=usage)])
    provider, completions = _streaming_provider(_Stream([dropped]), good)
    assert provider.generate_text([TextPart("问题")]).output == "恢复了" and len(completions.calls) == 2

    provider, completions = _streaming_provider(_Stream([dropped]))
    with pytest.raises(LLMError, match="暂时不可用.*重试 2 次仍失败"):
        provider.generate_text([TextPart("问题")])
    assert len(completions.calls) == 3

    import httpx2

    # SDK 用 httpx2 发请求，读到一半断开时直接抛出它的传输错误（M4 验收：按 httpx 的异常类判断会漏掉）
    cut_off = httpx2.RemoteProtocolError("peer closed connection without sending complete message body")
    provider, completions = _streaming_provider(_Stream([_chunk("前半"), cut_off]), good)
    assert provider.generate_text([TextPart("问题")]).output == "恢复了" and len(completions.calls) == 2


def test_client_retries_only_in_one_layer(monkeypatch):
    # M4 验收：SDK 重试 2 次、外层再重试 3 次，每次等满 600 秒，一个转写片段卡了一小时
    monkeypatch.setenv("LEC_TEST_KEY", "test-key")
    streaming = OpenAICompatProvider(make_profile(api_key_env="LEC_TEST_KEY", options={"stream": True}))
    assert streaming._client.max_retries == 0
    assert (streaming._client.timeout.read, streaming._client.timeout.connect) == (300.0, 600.0)
    plain = OpenAICompatProvider(make_profile(api_key_env="LEC_TEST_KEY", options={"stream_idle_s": 90}))
    assert plain._client.timeout.read == 600.0  # 非流式请求要等完整结果
    assert make_profile(options={"stream_idle_s": 90}).stream_idle_s == 90

    import httpx
    import openai

    dropped = openai.APIConnectionError(request=httpx.Request("POST", "https://relay.example/v1"))
    completions = _Failing([dropped], _completion(content="恢复了"))  # 非流式的连接错误也在这一层重试
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = OpenAICompatProvider(make_profile(), client=client)
    assert provider.generate_text([TextPart("问题")]).output == "恢复了" and len(completions.calls) == 2


class _Failing(StubCompletions):
    """前几次请求抛出预设的接口错误，之后正常返回。"""

    def __init__(self, errors, response):
        super().__init__(response)
        self.errors = list(errors)

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        if self.errors:
            raise self.errors.pop(0)
        return self._next()


def _status_error(status, cls_name):
    import httpx
    import openai

    response = httpx.Response(status, request=httpx.Request("POST", "https://relay.example/v1"))
    return getattr(openai, cls_name)(f"HTTP {status}", response=response, body=None)


@pytest.mark.parametrize("stream", [False, True])
def test_busy_server_is_retried_then_reported(stream):
    # M4 验收实测：中转站号池容量不足返回 503，SDK 只快速重试两次就放弃
    busy = _status_error(503, "InternalServerError")
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=2, prompt_tokens_details=None)
    good = _Stream([_chunk("恢复了", finish="stop"), _chunk(usage=usage)]) if stream else _completion(content="恢复了")
    completions = _Failing([busy, _status_error(429, "RateLimitError")], good)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = OpenAICompatProvider(make_profile(options={"stream": stream}), client=client)
    assert provider.generate_text([TextPart("问题")]).output == "恢复了" and len(completions.calls) == 3

    completions.errors = [busy] * 4
    with pytest.raises(LLMError, match="暂时不可用.*重试 3 次仍失败.*InternalServerError"):
        provider.generate_text([TextPart("问题")])


def test_other_api_errors_are_reported_without_retry():
    completions = _Failing([_status_error(400, "BadRequestError")], _completion())
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = OpenAICompatProvider(make_profile(), client=client)
    with pytest.raises(LLMError, match="请求失败：BadRequestError"):
        provider.generate_text([TextPart("问题")])
    assert len(completions.calls) == 1
