"""语音识别接口：所有后端都返回统一的、带时间戳的句子列表。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class Segment:
    start_ms: int
    end_ms: int
    text: str
    lang: str = ""  # 语种代码（zh / ko …）；不识别语种的后端留空


class ASRError(RuntimeError):
    """语音识别失败。"""


class ASRBackend(ABC):
    name: str = ""
    progress: Callable[[str], None] | None = None  # 转写进度（如"第 3/18 片"），由调用方设置

    def notify(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def warm_up(self) -> None:
        """预先加载模型等耗时资源（可选）；计时对比时先调用，避免把加载时间算进转写时间。"""

    def close(self) -> None:
        """释放模型占用的显存（可选）；同一进程里依次运行多个后端时调用，避免显存不够。"""

    @abstractmethod
    def transcribe(self, wav: Path, hotwords: Sequence[str] = ()) -> list[Segment]:
        """转写 16kHz 单声道 wav，返回按时间排序的句子。hotwords 为课程术语，用于提高识别率。"""


def free_gpu_memory() -> None:
    """回收已释放模型的显存；没装 torch 时什么也不做。"""
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_segments(segments: Sequence[Segment], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(seg) for seg in segments]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_segments(path: Path) -> list[Segment]:
    return [Segment(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def segments_text(
    segments: Sequence[Segment], start_ms: int = 0, end_ms: int | None = None, lang: str | None = None
) -> str:
    """取出时间窗口内（按句子起点判断）、指定语种的文字，用于和人工校对稿比较。"""
    return "".join(
        seg.text
        for seg in segments
        if seg.start_ms >= start_ms
        and (end_ms is None or seg.start_ms < end_ms)
        and (lang is None or seg.lang == lang)
    )
