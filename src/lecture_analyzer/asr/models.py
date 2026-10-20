"""本地语音识别模型：各转写方案要用哪些模型、是否已下载、下载（显示进度）。

模型从 ModelScope 下载到 [asr] model_dir（不填时用 ModelScope 默认的缓存目录）。下载位置与 FunASR、
Qwen3-ASR 自己下载时完全相同，所以下载过的不会重复下载，也可以把 model_dir 指向已有的 ModelScope 缓存。
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

from ..config import AppConfig
from . import use_model_dir
from .base import ASRError

# FunASR 预设里的模型简称 → ModelScope 上的模型（与 funasr/download/name_maps_from_hub.py 一致）
FUNASR_REPOS = {
    "paraformer-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
}
# Qwen3-ASR 型号 → ModelScope 上的模型（qwen3_backend.py 也用这张表；放在这里，查下载状态时不用导入 transformers）
QWEN3_REPOS = {
    "1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
    "0.6b": "Qwen/Qwen3-ASR-0.6B-hf",
}
# 大约的下载大小（MB），只用来估算进度
SIZES_MB = {
    "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch": 953,
    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch": 4,
    "iic/punc_ct-transformer_cn-en-common-vocab471067-large": 1160,
    "iic/SenseVoiceSmall": 897,
    "Qwen/Qwen3-ASR-1.7B-hf": 3900,
    "Qwen/Qwen3-ASR-0.6B-hf": 1500,
}
REVISION = "master"  # FunASR 下载时用的版本，缓存目录是 snapshots/master
PROGRESS_SECONDS = 3.0

Report = Callable[[str], None]


def required_repos(cfg: AppConfig, spec: str | None = None) -> list[str]:
    """转写方案要用的 ModelScope 模型（含语音活动检测、标点）；大模型转写为空。spec 为空时用 [asr] 设置。"""
    kind, _, arg = (spec or cfg.asr.backend).partition(":")
    if kind not in ("funasr", "qwen3"):
        return []
    from .funasr_backend import PRESETS, VAD_MODEL  # 只在用本地方案时导入（需要本地语音识别依赖）

    vad = FUNASR_REPOS[VAD_MODEL]
    if kind == "funasr":
        preset = PRESETS.get(arg or cfg.asr.funasr_preset)
        if preset is None:
            return []
        names = [preset.model, VAD_MODEL, *([preset.punc_model] if preset.punc_model else [])]
        return [FUNASR_REPOS.get(name, name) for name in names]
    model = QWEN3_REPOS.get(arg or cfg.asr.qwen3_size)
    # 模型来源是 Hugging Face 时由 transformers 自己下载，这里只管语音活动检测模型
    return [model, vad] if model and cfg.asr.model_hub == "modelscope" else [vad]


def cache_root(cfg: AppConfig) -> Path:
    """ModelScope 的缓存位置：[asr] model_dir；不填时与 ModelScope 默认一致。"""
    if cfg.model_dir is not None:
        return cfg.model_dir
    env = os.environ.get("MODELSCOPE_CACHE")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "modelscope"


def repo_dir(root: Path, repo: str) -> Path:
    return root / "models" / repo.replace("/", "--")


def is_downloaded(root: Path, repo: str) -> bool:
    """已完整下载：snapshots 下有模型的说明文件，且没有下载到一半的文件。"""
    for snapshot in (repo_dir(root, repo) / "snapshots").glob("*"):
        described = (snapshot / "configuration.json").is_file() or (snapshot / "config.json").is_file()
        if described and not any(snapshot.rglob("*.incomplete")):
            return True
    return False


def model_status(cfg: AppConfig, spec: str | None = None) -> list[tuple[str, bool]]:
    """（模型，是否已下载），设置页显示用。"""
    root = cache_root(cfg)
    return [(repo, is_downloaded(root, repo)) for repo in required_repos(cfg, spec)]


def download_models(cfg: AppConfig, spec: str | None, report: Report) -> list[str]:
    """下载转写方案要用、还没下载的模型，下载期间报告进度；返回这次下载了哪些。"""
    root = cache_root(cfg)
    missing = [repo for repo in required_repos(cfg, spec) if not is_downloaded(root, repo)]
    if not missing:
        return []
    use_model_dir(cfg)
    for index, repo in enumerate(missing, start=1):
        label = f"下载本地模型 {index}/{len(missing)}（{repo}）"
        report(f"{label}…")
        _download_with_progress(repo, repo_dir(root, repo), SIZES_MB.get(repo, 0),
                                lambda text, label=label: report(f"{label}：{text}"))
    return missing


def _download_with_progress(repo: str, folder: Path, expected_mb: int, report: Report) -> None:
    """在后台线程里下载，当前线程每隔几秒按目录大小报告进度（书库连接不能跨线程使用，所以由当前线程报告）。"""
    errors: list[BaseException] = []

    def download() -> None:
        try:
            _snapshot(repo)
        except BaseException as exc:  # 交给当前线程抛出
            errors.append(exc)

    thread = threading.Thread(target=download, name="lec-model-download", daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(PROGRESS_SECONDS)
        report(_progress_text(folder, expected_mb))
    if errors:
        raise ASRError(f"下载模型 {repo} 失败：{errors[0]}") from errors[0]


def _snapshot(repo: str) -> None:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise ASRError("没有安装本地语音识别依赖，请先运行：uv sync --extra asr") from exc
    snapshot_download(repo, revision=REVISION)


def _progress_text(folder: Path, expected_mb: int) -> str:
    done_mb = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()) / 1e6 if folder.is_dir() else 0
    if not expected_mb:
        return f"已下载 {done_mb:.0f} MB"
    percent = min(99, int(done_mb * 100 / expected_mb))  # 大小只是估计，下载完成前最多显示 99%
    return f"已下载 {done_mb:.0f} / 约 {expected_mb} MB（{percent}%）"
