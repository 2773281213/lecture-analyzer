"""统一的模型接口：流程代码只依赖这里的类型，不直接接触任何厂商 SDK。

新增一种接入方式时，继承 LLMProvider 并实现 `_complete`，再到 registry.py 注册即可；
同一接入方式下的新模型（如新的国产模型）只需在 config.toml 里加一个模型档案。
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, Sequence, TypeVar

from pydantic import BaseModel

AdapterType = Literal["anthropic", "gemini", "openai_compat"]
StructuredMode = Literal["native", "json_schema", "json_object", "prompt"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    mime_type: str = "image/png"


@dataclass(frozen=True)
class AudioPart:
    data: bytes
    mime_type: str = "audio/mp3"


Part = TextPart | ImagePart | AudioPart


@dataclass
class Usage:
    """一次调用的 token 用量。input_tokens 含缓存命中部分，cache_read_tokens 是其中的子集。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
        )


class ModelProfile(BaseModel):
    """模型档案：一个可调用模型的连接信息与能力声明，来自 config.toml 的 [models.<名称>]。"""

    name: str = ""
    adapter: AdapterType
    model: str = ""
    model_env: str = ""  # 从该环境变量（.env）读取型号，优先于 model；中转站共用一个地址时靠它区分模型
    api_key_env: str = ""
    base_url: str | None = None
    base_url_env: str = ""  # 从该环境变量（.env）读取接口地址，优先于 base_url；用于中转站
    vision: bool = False
    audio: bool = False
    structured: StructuredMode = "prompt"
    supports_effort: bool = False
    max_context: int = 128_000
    max_output: int = 16_000
    price_in: float = 0.0  # 美元 / 百万输入 token
    price_out: float = 0.0  # 美元 / 百万输出 token
    price_cache_read: float = 0.0  # 美元 / 百万缓存命中 token
    tokens_per_char: float = 1.0  # 中文字符到 token 的换算比例，M0 实测后填写
    timeout_s: float = 600.0
    options: dict[str, Any] = {}  # 适配器专属选项

    @property
    def is_configured(self) -> bool:
        return bool(self.model.strip())

    def api_key(self) -> str | None:
        """密钥：Windows 凭据管理器里保存的优先，没有时用环境变量（含 .env）。"""
        if not self.api_key_env:
            return None
        from ..credentials import get_secret  # 延迟导入：只在真正需要密钥时加载 keyring

        return get_secret(self.api_key_env)

    @property
    def stream_idle_s(self) -> float:
        """流式请求多久收不到任何数据就当作连接卡住（作为读取超时）。"""
        return float(self.options.get("stream_idle_s", min(self.timeout_s, STREAM_IDLE_SECONDS)))

    def required_api_key(self) -> str:
        """取密钥，没有时说明去哪里填（各家 SDK 自己的报错只说缺凭据，看不出该去哪里填）。"""
        key = self.api_key()
        if key:
            return key
        where = f"命令行：在 .env 里设置 {self.api_key_env}" if self.api_key_env else "config.toml 里这个档案没有设置 api_key_env"
        raise LLMError(f"模型档案「{self.name}」缺少密钥：请在设置页填写（{where}）")

    def resolved_base_url(self) -> str | None:
        """实际使用的接口地址：环境变量里设置了就用它，否则用档案里的 base_url，都没有则直连官方。"""
        if self.base_url_env:
            value = os.environ.get(self.base_url_env, "").strip()
            if value:
                return value
        return self.base_url or None

    def estimate_cost(self, usage: Usage) -> float | None:
        """按档案里的单价估算费用（美元）；没填单价时返回 None。"""
        if not (self.price_in or self.price_out):
            return None
        uncached = max(usage.input_tokens - usage.cache_read_tokens, 0)
        cache_price = self.price_cache_read or self.price_in
        return (
            uncached * self.price_in
            + usage.cache_read_tokens * cache_price
            + usage.output_tokens * self.price_out
        ) / 1_000_000


OutT = TypeVar("OutT")
SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass
class LLMResult(Generic[OutT]):
    output: OutT
    usage: Usage
    model: str


class LLMError(RuntimeError):
    """模型调用失败：拒答、输出被截断、能力不支持等。usage 记录失败前已消耗的用量。"""

    def __init__(self, message: str, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage or Usage()


class StructuredOutputError(LLMError):
    """结构化输出连续两次没有通过校验。"""


class EmptyResponseError(LLMError):
    """模型没有返回可用的结果（空结果，或中转站代替结果返回的拦截说明），多半是上游的内容审核拦截了这次输入
    （同一输入重试通常仍被拦截）。"""


RETRY_DELAYS = (5.0, 20.0, 60.0)  # 临时错误重试前等待的秒数（共重试三次）
# 服务端暂时不可用：请求超时、冲突、限流、内部错误、网关错误、容量不足、过载（与 SDK 自带重试的范围相同）。
# 中转站号池容量不足（503 No capacity available）往往要等几十秒才恢复（M4 验收实测），SDK 的快速重试不够
BUSY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
# 流式请求多久收不到任何数据就当作连接卡住（档案选项 stream_idle_s 可调）。M4 验收：中转站接受了请求却一直
# 不返回，每次都等满 600 秒超时，加上 SDK 和外层的重试，一个转写片段卡了一小时
STREAM_IDLE_SECONDS = 300.0
CallT = TypeVar("CallT")


def retry_transient(call: Callable[[], CallT], connection_errors: tuple[type[BaseException], ...], name: str) -> CallT:
    """临时错误等一会儿重发，重试几次仍失败时转成 LLMError；其他异常照常抛出。

    临时错误指 connection_errors 里的连接错误（含超时、流式读到一半断开），以及状态码在 BUSY_STATUS 里的接口错误
    （各家 SDK 的接口错误都带 status_code）。重试只在这里做一层：适配器创建客户端时关掉 SDK 自带的重试，
    否则两层相乘，卡住的请求要等很久才报错。
    """
    delays = list(RETRY_DELAYS)
    while True:
        try:
            return call()
        except Exception as exc:
            if not (isinstance(exc, connection_errors) or getattr(exc, "status_code", None) in BUSY_STATUS):
                raise
            if not delays:
                raise LLMError(
                    f"模型档案「{name}」暂时不可用（连接中断或服务端繁忙），重试 {len(RETRY_DELAYS)} 次仍失败，"
                    f"请稍后重新提交：{type(exc).__name__}: {str(exc)[:300]}"
                ) from exc
            time.sleep(delays.pop(0))


def schema_instruction(schema: type[BaseModel]) -> str:
    """给不支持原生结构化输出的模型用：把 JSON Schema 写进提示词。"""
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    return (
        "只输出一个 JSON 对象，不要输出解释、前后缀或代码块标记。"
        f"该 JSON 必须符合下面的 JSON Schema：\n{schema_json}"
    )


def extract_json_text(text: str) -> str:
    """取出模型输出中的 JSON 对象文本，兼容 ```json 代码块和前后多余文字。"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("输出中找不到 JSON 对象")
    return text[start : end + 1]


def parse_structured(text: str, schema: type[SchemaT]) -> SchemaT:
    return schema.model_validate_json(extract_json_text(text))


def _repair_hint(error: Exception) -> str:
    detail = str(error)
    if len(detail) > 1500:
        detail = detail[:1500] + "……"
    return (
        f"上一次的输出没有通过格式校验，错误如下：\n{detail}\n"
        "请重新完成同样的任务，只输出完整、合法、符合要求结构的 JSON。"
    )


class LLMProvider(ABC):
    """所有模型适配器的基类。子类只需实现 `_complete`，校验与重试在这里统一处理。"""

    def __init__(self, profile: ModelProfile) -> None:
        if not profile.is_configured:
            raise LLMError(f"模型档案「{profile.name}」没有填写 model")
        self.profile = profile

    @abstractmethod
    def _complete(
        self,
        parts: Sequence[Part],
        *,
        system: str | None,
        schema: type[BaseModel] | None,
        effort: Effort | None,
        max_tokens: int,
    ) -> tuple[str, Usage]:
        """调用一次模型，返回原始文本和用量；schema 不为空时，输出应是符合该结构的 JSON。"""

    def list_remote_models(self) -> list[str]:
        """列出服务端可用的模型名，方便填写配置。"""
        raise NotImplementedError(f"{type(self).__name__} 不支持列出模型")

    def generate_text(
        self,
        parts: Sequence[Part],
        *,
        system: str | None = None,
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult[str]:
        self._check_parts(parts)
        text, usage = self._complete(
            parts, system=system, schema=None, effort=effort,
            max_tokens=max_tokens or self.profile.max_output,
        )
        return LLMResult(text, usage, self.profile.model)

    def generate_structured(
        self,
        parts: Sequence[Part],
        schema: type[SchemaT],
        *,
        system: str | None = None,
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult[SchemaT]:
        """要求模型输出符合 schema 的 JSON；校验失败时附上错误信息重试一次。"""
        self._check_parts(parts)
        limit = max_tokens or self.profile.max_output
        usage = Usage()
        attempt_parts: Sequence[Part] = parts
        failure: ValueError | None = None
        for _ in range(2):
            try:
                text, call_usage = self._complete(
                    attempt_parts, system=system, schema=schema, effort=effort, max_tokens=limit
                )
                usage = usage + call_usage
                return LLMResult(parse_structured(text, schema), usage, self.profile.model)
            except ValueError as error:  # 含 pydantic.ValidationError
                failure = error
                attempt_parts = [*parts, TextPart(_repair_hint(error))]
        raise StructuredOutputError(
            f"模型档案「{self.profile.name}」两次输出都不符合 {schema.__name__}：{failure}", usage
        ) from failure

    def _check_parts(self, parts: Sequence[Part]) -> None:
        if not self.profile.vision and any(isinstance(p, ImagePart) for p in parts):
            raise LLMError(f"模型档案「{self.profile.name}」不支持图片输入（vision = false）")
        if not self.profile.audio and any(isinstance(p, AudioPart) for p in parts):
            raise LLMError(f"模型档案「{self.profile.name}」不支持音频输入（audio = false）")
