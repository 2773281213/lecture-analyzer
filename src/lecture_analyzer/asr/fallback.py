"""大模型转写缺失片段的本地补转。

[asr] fallback = "auto" 时，只有中文的课用 Paraformer，其他语言用 Qwen3；也可以写成具体方案，或 "none" 表示不补转。
补转只是尽力而为：方案写错、没装本地依赖、显存不足等情况下保留缺口并在处理记录里说明，不中断整个流程。
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from ..config import AppConfig, ConfigError
from ..course import CourseSettings
from ..media import extract_audio
from ..timecode import format_ms
from . import create_backend, local_spec
from .base import ASRBackend, Segment

if TYPE_CHECKING:
    from ..llm import ModelProfile, Usage


def fill_skipped(
    cfg: AppConfig, settings: CourseSettings, wav: Path, skipped: Sequence[tuple[int, int]], work_dir: Path,
    on_usage: Callable[[ModelProfile, Usage], None] | None = None,
) -> tuple[list[Segment], str]:
    """用本地模型补转缺失的区间（毫秒），返回（补出的句子，处理说明）。"""
    ranges = "、".join(f"{format_ms(start)}–{format_ms(end)}" for start, end in skipped)
    if cfg.asr.fallback == "none":
        return [], f"{ranges} 被上游拦截，未补转"
    spec = local_spec(settings) if cfg.asr.fallback == "auto" else cfg.asr.fallback
    try:
        with closing(create_backend(cfg, spec, work_dir=work_dir, settings=settings)) as backend:
            backend.on_usage = on_usage  # 大模型补转（llm:档案）逐次记账；本地模型没有请求，不会回调
            filled = [seg for span in skipped for seg in _transcribe_span(backend, wav, span, work_dir)]
    except (RuntimeError, ConfigError) as exc:  # ASRError、MediaError、显存不足，以及方案里写错的档案名；代码缺陷照常抛出
        return [], f"{ranges} 被上游拦截，本地补转失败：{exc}"
    return filled, f"{ranges} 被上游拦截，已用 {backend.name} 补转 {len(filled)} 句"


def _transcribe_span(backend: ASRBackend, wav: Path, span: tuple[int, int], work_dir: Path) -> list[Segment]:
    """截出一段音频转写，再把时间换算回整节课。"""
    start, end = span
    piece = extract_audio(
        wav, work_dir / f"gap_{start}.wav", start_seconds=start / 1000, max_seconds=(end - start) / 1000
    )
    return [replace(s, start_ms=s.start_ms + start, end_ms=s.end_ms + start) for s in backend.transcribe(piece)]
