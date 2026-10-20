"""语音识别：统一接口 + 本地 FunASR / Qwen3-ASR / 大模型转写几种后端。

转写方案写成 "类型[:参数]"；"auto" 按课程的授课语言选本地模型（resolve_spec），任何语言的课都能转写。
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import AppConfig
from ..course import CourseSettings, language_name
from .base import ASRBackend, ASRError, Segment, load_segments, save_segments, segments_text

# 设置页的默认转写方案（[asr] backend）
ASR_BACKENDS = {
    "auto": "自动（按课程的授课语言选本地模型：只有中文用 Paraformer，其他语言用 Qwen3-ASR）",
    "llm": "大模型转写（推荐，不占显卡）", "funasr": "本地 FunASR", "qwen3": "本地 Qwen3-ASR（多语种）",
}
# 只认部分语言的本地方案；Qwen3-ASR（52 种语言）和大模型转写不限语言
LOCAL_LANGUAGES = {
    "paraformer": ("本地 Paraformer", {"zh"}),
    "sensevoice": ("本地 SenseVoice", {"zh", "en", "ja", "ko", "yue"}),
}


def local_spec(settings: CourseSettings) -> str:
    """只有中文时用 Paraformer（支持热词），其他语言或多语混合用 Qwen3（支持 52 种语言）。"""
    return "funasr:paraformer" if settings.spoken_languages == ["zh"] else "qwen3"


def resolve_spec(cfg: AppConfig, spec: str | None, settings: CourseSettings) -> str:
    """这门课实际用的转写方案：没指定时用 [asr] 设置，"auto" 按授课语言选本地模型；
    方案不支持课程录音里的语言时报错（而不是转写出错乱的结果）。"""
    spec = spec or cfg.asr.backend
    if spec == "auto":
        return local_spec(settings)
    problem = language_problem(cfg, spec, settings)
    if problem:
        raise ASRError(problem)
    return spec


def language_problem(cfg: AppConfig, spec: str | None, settings: CourseSettings) -> str:
    """方案不支持课程录音里的语言时，返回给使用者看的说明；支持时返回空字符串。"""
    kind, _, arg = (spec or cfg.asr.backend).partition(":")
    model = (arg or cfg.asr.funasr_preset) if kind == "funasr" else ""
    if model not in LOCAL_LANGUAGES:
        return ""
    label, supported = LOCAL_LANGUAGES[model]
    missing = [code for code in settings.spoken_languages if code.split("-")[0] not in supported]
    if not missing:
        return ""
    names = "、".join(language_name(code) for code in missing)
    return f"{label} 不支持{names}：请把这门课的转写方案改成「自动」、本地 Qwen3-ASR 或大模型转写"


def create_backend(
    cfg: AppConfig, spec: str | None = None, *, work_dir: Path, settings: CourseSettings | None = None
) -> ASRBackend:
    """按 "类型[:参数]" 创建后端，例如 funasr:paraformer、funasr:sensevoice、qwen3:1.7b、llm:gemini。

    spec 为空时使用 config.toml 的 [asr] 设置。后端模块延迟导入，没装本地依赖时也能用大模型转写。
    settings 为课程设置：录音里只有一种语言时强制该语种，多种语言（含现场翻译）时逐句自动识别；
    不传时一律自动识别。
    """
    if (spec or cfg.asr.backend) == "auto":
        spec = local_spec(settings or CourseSettings())
    kind, _, arg = (spec or cfg.asr.backend).partition(":")
    language = settings.asr_language if settings is not None else None
    if kind in ("funasr", "qwen3"):
        use_model_dir(cfg)
    if kind == "funasr":
        from .funasr_backend import FunASRBackend

        return FunASRBackend(arg or cfg.asr.funasr_preset, device=resolve_device(cfg.asr.device),
                             language=language or "auto")
    if kind == "qwen3":
        from .qwen3_backend import Qwen3ASRBackend

        return Qwen3ASRBackend(
            arg or cfg.asr.qwen3_size, device=resolve_device(cfg.asr.device), language=language,
            hub=cfg.asr.model_hub,
        )
    if kind == "llm":
        from ..llm import create_provider
        from .llm_backend import LLMTranscriber

        profile = cfg.profile(arg or cfg.asr.llm_profile)
        alternates = [
            create_provider(profile.model_copy(update={"model": model}))
            for model in cfg.asr.llm_alternates if model and model != profile.model
        ]
        return LLMTranscriber(
            create_provider(profile), work_dir, chunk_minutes=cfg.asr.chunk_minutes, settings=settings,
            alternates=alternates,
        )
    raise ASRError(f"未知的转写后端：{spec}（可选：auto、funasr[:预设]、qwen3[:型号]、llm[:模型档案]）")


def resolve_device(device: str) -> str:
    """auto：有可用的 NVIDIA 显卡就用 cuda:0，否则用 CPU（很慢，界面会提示改用大模型转写）。"""
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def use_model_dir(cfg: AppConfig) -> None:
    """配置了 [asr] model_dir 时，本地模型（ModelScope、Hugging Face）都下载到这里。"""
    if cfg.model_dir is None:
        return
    os.environ["MODELSCOPE_CACHE"] = str(cfg.model_dir)
    os.environ["HF_HOME"] = str(cfg.model_dir / "huggingface")


__all__ = [
    "ASRBackend",
    "ASRError",
    "Segment",
    "create_backend",
    "language_problem",
    "local_spec",
    "resolve_spec",
    "load_segments",
    "save_segments",
    "segments_text",
]
