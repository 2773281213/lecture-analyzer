"""音视频处理：使用 imageio-ffmpeg 自带的 ffmpeg，不依赖系统安装。"""

from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg

SAMPLE_RATE = 16_000  # 语音识别模型的标准采样率


class MediaError(RuntimeError):
    """音视频处理失败。"""


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: list[str]) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise MediaError(f"ffmpeg 执行失败：{result.stderr.strip()[-800:]}")


def extract_audio(
    src: Path,
    dst: Path,
    sample_rate: int = SAMPLE_RATE,
    max_seconds: float | None = None,
    start_seconds: float | None = None,
) -> Path:
    """从视频或音频中抽出单声道 16 bit PCM wav；start/max 用于只截取一段做试验。"""
    if not src.is_file():
        raise MediaError(f"找不到文件：{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    seek = ["-ss", f"{start_seconds:g}"] if start_seconds else []  # 放在 -i 前面：快速定位
    limit = ["-t", f"{max_seconds:g}"] if max_seconds else []
    pcm = ["-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le"]
    run_ffmpeg([*seek, "-i", str(src), *limit, *pcm, str(dst)])
    return dst


def encode_mp3(src: Path, dst: Path, bitrate: str = "64k") -> Path:
    """转成单声道 mp3（10 分钟约 5MB），用于发给云端模型；Gemini 原生接口和 OpenAI 兼容接口都支持。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", str(src), "-ac", "1", "-c:a", "libmp3lame", "-b:a", bitrate, str(dst)])
    return dst


def wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as audio:
        return int(audio.getnframes() * 1000 / audio.getframerate())


def split_wav(path: Path, chunk_seconds: float, out_dir: Path) -> list[tuple[Path, int]]:
    """把 wav 按固定时长切片，返回 [(切片路径, 起始毫秒)]。"""
    if chunk_seconds <= 0:
        raise MediaError("切片时长必须大于 0")
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[Path, int]] = []
    with wave.open(str(path), "rb") as src:
        params = src.getparams()
        rate = src.getframerate()
        frame_bytes = params.sampwidth * params.nchannels
        frames_per_chunk = max(round(chunk_seconds * rate), 1)  # round：避免浮点误差多切出一个极短尾片
        start_frame = 0
        while data := src.readframes(frames_per_chunk):
            chunk_path = out_dir / f"{path.stem}_{len(chunks):03d}.wav"
            with wave.open(str(chunk_path), "wb") as dst:
                dst.setparams(params)
                dst.writeframes(data)
            chunks.append((chunk_path, start_frame * 1000 // rate))
            start_frame += len(data) // frame_bytes
    return chunks


def halve_wav(path: Path, out_dir: Path) -> list[tuple[Path, int]]:
    """按帧对半切成两片（奇数帧时前一片多一帧），返回值同 split_wav。"""
    with wave.open(str(path), "rb") as src:
        half_frames = math.ceil(src.getnframes() / 2)
        rate = src.getframerate()
    return split_wav(path, half_frames / rate, out_dir)
