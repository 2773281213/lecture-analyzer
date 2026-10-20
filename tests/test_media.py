import wave
from pathlib import Path

import pytest

from lecture_analyzer.media import (
    MediaError, encode_mp3, extract_audio, halve_wav, run_ffmpeg, split_wav, wav_duration_ms,
)


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> Path:
    """生成 3 秒的测试视频（彩条 + 440Hz 正弦音），模拟课堂录像。"""
    path = tmp_path_factory.mktemp("media") / "课堂录像.mp4"
    run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=44100",
        "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(path),
    ])
    return path


def test_extract_audio_from_video(sample_video: Path, tmp_path: Path):
    wav = extract_audio(sample_video, tmp_path / "out" / "audio.wav")
    with wave.open(str(wav), "rb") as audio:
        assert audio.getframerate() == 16_000
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
    assert abs(wav_duration_ms(wav) - 3000) < 100


def test_extract_audio_can_limit_duration(sample_video: Path, tmp_path: Path):
    wav = extract_audio(sample_video, tmp_path / "head.wav", max_seconds=1.5)
    assert abs(wav_duration_ms(wav) - 1500) < 100


def test_extract_audio_can_start_midway(sample_video: Path, tmp_path: Path):
    wav = extract_audio(sample_video, tmp_path / "mid.wav", start_seconds=2.0)
    assert abs(wav_duration_ms(wav) - 1000) < 150


def test_split_wav_offsets(sample_video: Path, tmp_path: Path):
    wav = extract_audio(sample_video, tmp_path / "audio.wav")
    chunks = split_wav(wav, 1.0, tmp_path / "chunks")
    assert [offset for _, offset in chunks][:3] == [0, 1000, 2000]
    assert sum(wav_duration_ms(path) for path, _ in chunks) == pytest.approx(wav_duration_ms(wav), abs=5)


def _frames(path: Path) -> int:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes()


@pytest.mark.parametrize("frames", [48_000, 48_001])
def test_halve_wav_gives_exactly_two_pieces(tmp_path: Path, frames: int):
    wav = tmp_path / "odd.wav"
    with wave.open(str(wav), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16_000)
        out.writeframes(b"\0\0" * frames)
    halves = halve_wav(wav, tmp_path / "halves")
    assert [offset for _, offset in halves] == [0, 1500]
    assert sum(_frames(path) for path, _ in halves) == frames


def test_encode_mp3(sample_video: Path, tmp_path: Path):
    wav = extract_audio(sample_video, tmp_path / "audio.wav")
    data = encode_mp3(wav, tmp_path / "audio.mp3").read_bytes()
    assert data[:3] == b"ID3" or data[0] == 0xFF  # ID3 标签或 MPEG 帧同步头
    assert len(data) < (tmp_path / "audio.wav").stat().st_size


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(MediaError, match="找不到文件"):
        extract_audio(tmp_path / "不存在.mp4", tmp_path / "x.wav")


def test_split_rejects_non_positive_chunk(tmp_path: Path):
    with pytest.raises(MediaError):
        split_wav(tmp_path / "x.wav", 0, tmp_path)
