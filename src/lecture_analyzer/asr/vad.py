"""语音活动检测（VAD）：把长音频切成说话片段，交给不输出时间戳的识别模型逐段识别。

片段边界就是停顿处。在"老师讲韩语、现场翻译"的课上，老师和翻译之间有停顿，
所以一个片段基本只含一种语言、一个说话人。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .funasr_backend import MAX_SEGMENT_MS, VAD_MODEL, load_automodel

MIN_SEGMENT_MS = 400  # 更短的片段多半是咳嗽、"嗯"之类，识别出来也是噪声


class VadSegmenter:
    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self._model: Any = None

    def segment(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """返回说话片段的 [(开始毫秒, 结束毫秒)]，已去掉过短的片段。"""
        results = self._load().generate(input=audio)
        spans = results[0].get("value", []) if results else []
        return [(int(start), int(end)) for start, end in spans if end - start >= MIN_SEGMENT_MS]

    def warm_up(self) -> None:
        self._load()

    def close(self) -> None:
        self._model = None

    def _load(self) -> Any:
        if self._model is None:
            self._model = load_automodel(
                model=VAD_MODEL, max_single_segment_time=MAX_SEGMENT_MS, device=self.device, disable_update=True
            )
        return self._model
