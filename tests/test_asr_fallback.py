from pathlib import Path

import pytest

from lecture_analyzer.asr import fallback
from lecture_analyzer.asr.base import ASRBackend, ASRError, Segment
from lecture_analyzer.config import load_config
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.media import run_ffmpeg


class LocalASR(ASRBackend):
    name = "funasr:paraformer"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.closed = False

    def transcribe(self, wav, hotwords=()):
        if self.error:
            raise self.error
        return [Segment(100, 900, "补转的句子")]

    def close(self):
        self.closed = True


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=5", "-ar", "16000", "-ac", "1", str(path)])
    return path


def test_skipped_ranges_are_filled(example_config, wav, tmp_path, monkeypatch):
    backend = LocalASR()
    specs = []
    monkeypatch.setattr(fallback, "create_backend", lambda cfg, spec, **kw: specs.append(spec) or backend)
    filled, note = fallback.fill_skipped(load_config(example_config), CourseSettings(teaching="zh"), wav,
                                         [(2000, 3000)], tmp_path / "fb")
    assert filled == [Segment(2100, 2900, "补转的句子")]  # 换算回整节课的时间
    assert specs == ["funasr:paraformer"] and backend.closed
    assert "00:02–00:03 被上游拦截，已用 funasr:paraformer 补转 1 句" in note


def test_korean_course_uses_qwen3(example_config, wav, tmp_path, monkeypatch):
    specs = []
    monkeypatch.setattr(fallback, "create_backend", lambda cfg, spec, **kw: specs.append(spec) or LocalASR())
    fallback.fill_skipped(load_config(example_config), CourseSettings(teaching="ko+zh"), wav, [(0, 1000)], tmp_path)
    assert specs == ["qwen3"]


def test_fallback_none_and_failures_keep_gap(example_config, wav, tmp_path, monkeypatch):
    cfg = load_config(example_config)
    cfg.asr.fallback = "none"
    filled, note = fallback.fill_skipped(cfg, CourseSettings(), wav, [(0, 1000)], tmp_path)
    assert filled == [] and "未补转" in note

    cfg.asr.fallback = "whisper"  # 写错的方案：创建后端时就失败
    filled, note = fallback.fill_skipped(cfg, CourseSettings(), wav, [(0, 1000)], tmp_path)
    assert filled == [] and "本地补转失败：未知的转写后端" in note

    cfg.asr.fallback = "auto"
    backend = LocalASR(ASRError("没有安装本地语音识别依赖"))
    monkeypatch.setattr(fallback, "create_backend", lambda *a, **kw: backend)
    filled, note = fallback.fill_skipped(cfg, CourseSettings(), wav, [(0, 1000)], tmp_path)
    assert filled == [] and "本地补转失败：没有安装本地语音识别依赖" in note
    assert backend.closed  # 转写失败也要释放模型
