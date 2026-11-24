"""测试模型档案能否连通：发一个很短的请求（命令行 `lec models --ping` 和设置页的「测试连接」共用）。"""

from __future__ import annotations

import time

from .base import ModelProfile, TextPart
from .registry import create_provider


def short_error(exc: Exception, limit: int = 200) -> str:
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return text if len(text) <= limit else text[:limit] + "……"


def ping(profile: ModelProfile) -> tuple[bool, str]:
    """返回（是否可用，说明）；说明里只有耗时、用量或错误信息，不含密钥。"""
    started = time.perf_counter()
    # 只测试当前档案本身，不触发拒答轮换到备用档案
    direct = profile.model_copy(update={"fallback_profile": None})
    try:
        result = create_provider(direct).generate_text(
            [TextPart("请只回复：OK")],
            effort="low" if direct.supports_effort else None,
            max_tokens=min(direct.max_output, 2048),
        )
    except Exception as exc:  # noqa: BLE001 —— 各家 SDK 的异常类型不同，统一转成说明
        return False, short_error(exc)
    usage = result.usage
    return True, f"{time.perf_counter() - started:.1f}s，输入 {usage.input_tokens} / 输出 {usage.output_tokens} token"
