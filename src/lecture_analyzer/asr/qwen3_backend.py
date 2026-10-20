"""Qwen3-ASR（transformers 原生支持）：VAD 切句后逐句识别，同时得到每句的语种。

适合韩语、中韩混合以及"老师讲韩语 + 现场翻译"的课。模型支持 52 种语言和方言，
术语表通过 prompt 作为上下文传入，提高专业词的识别率。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..media import SAMPLE_RATE
from .base import ASRBackend, ASRError, Segment, free_gpu_memory
from .funasr_backend import read_wav_float32
from .models import QWEN3_REPOS
from .vad import VadSegmenter

MAX_NEW_TOKENS = 448  # 单段最长 30 秒，够用
MAX_PROMPT_TERMS = 100
_SAMPLES_PER_MS = SAMPLE_RATE // 1000


class Qwen3ASRBackend(ASRBackend):
    def __init__(
        self,
        size: str = "1.7b",
        device: str = "cuda:0",
        language: str | None = None,
        batch_size: int = 4,
        vad: VadSegmenter | None = None,
        hub: str = "modelscope",
    ) -> None:
        if size not in QWEN3_REPOS:
            raise ASRError(f"未知的 Qwen3-ASR 型号：{size}（可选：{', '.join(QWEN3_REPOS)}）")
        self.model_id = QWEN3_REPOS[size]
        self.name = f"qwen3:{size}"
        self.device = device
        self.language = forced_language(language)  # None 表示自动识别语种
        self.batch_size = batch_size
        self.vad = vad or VadSegmenter(device)
        self.hub = hub
        self._loaded: tuple[Any, Any] | None = None  # (processor, model)

    def transcribe(self, wav: Path, hotwords: Sequence[str] = ()) -> list[Segment]:
        audio = read_wav_float32(wav)
        spans = self.vad.segment(audio)
        prompt = "、".join(term.strip() for term in hotwords[:MAX_PROMPT_TERMS] if term.strip()) or None
        segments: list[Segment] = []
        for start in range(0, len(spans), self.batch_size):
            batch = spans[start : start + self.batch_size]
            clips = [audio[begin * _SAMPLES_PER_MS : end * _SAMPLES_PER_MS] for begin, end in batch]
            for (begin, end), item in zip(batch, self._recognize(clips, prompt)):
                text = item.get("transcription", "").strip()
                if text:
                    segments.append(Segment(begin, end, text, self._lang_code(item.get("language"))))
        return segments

    def warm_up(self) -> None:
        self._load()
        self.vad.warm_up()

    def close(self) -> None:
        self._loaded = None
        self.vad.close()
        free_gpu_memory()

    def _recognize(self, clips: list[Any], prompt: str | None) -> list[dict[str, Any]]:
        import torch

        processor, model = self._load()
        inputs = processor.apply_transcription_request(
            audio=clips, language=self.language, prompt=[prompt] * len(clips) if prompt else None
        ).to(model.device, model.dtype)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        return [processor.decode(row, return_format="parsed") for row in new_tokens]

    def _lang_code(self, language_name: str | None) -> str:
        if self.language:  # 强制语种时模型不再输出语种
            return self.language
        if not language_name:
            return ""
        return _NAME_TO_CODE.get(language_name, language_name.lower())

    def _load(self) -> tuple[Any, Any]:
        if self._loaded is None:
            try:
                import torch
                from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
            except ImportError as exc:
                raise ASRError("没有安装本地语音识别依赖，请先运行：uv sync --extra asr") from exc
            model_dir = resolve_model_dir(self.model_id, self.hub)
            processor = AutoProcessor.from_pretrained(model_dir)
            # 不用 device_map（那需要额外安装 accelerate）：先在内存里加载，再整体搬到显卡
            model = Qwen3ASRForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16)
            model = model.to(self.device).eval()
            self._loaded = (processor, model)
        return self._loaded


def resolve_model_dir(model_id: str, hub: str) -> str:
    """返回可以直接加载的模型位置。

    modelscope：先下载到本地缓存，返回本地目录。官方 Qwen 仓库与 HuggingFace 上的权重一致，在国内更稳定。
    huggingface：直接返回模型名，由 transformers 自行下载。
    """
    if hub == "huggingface":
        return model_id
    if hub != "modelscope":
        raise ASRError(f"未知的模型来源：{hub}（可选：modelscope、huggingface）")
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise ASRError("没有安装本地语音识别依赖，请先运行：uv sync --extra asr") from exc
    return snapshot_download(model_id)


def _language_table() -> dict[str, str]:
    """Qwen3-ASR 能强制的语种：代码 → 名称（模型识别结果里输出的是名称）。"""
    try:
        from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME
    except ImportError:
        return {"zh": "Chinese", "ko": "Korean", "en": "English"}
    return dict(LANGUAGE_CODE_TO_NAME)


_CODE_TO_NAME = _language_table()
_NAME_TO_CODE = {name: code for code, name in _CODE_TO_NAME.items()}


def forced_language(code: str | None) -> str | None:
    """课程语言换成 Qwen3-ASR 可以强制的语种（zh-tw 取 zh）；不支持的语言改为自动识别。"""
    base = code.split("-")[0] if code else None
    return base if base in _CODE_TO_NAME else None
