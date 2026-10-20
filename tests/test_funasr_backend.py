from pathlib import Path

import numpy as np
import pytest

from lecture_analyzer.asr import ASRError, create_backend
from lecture_analyzer.asr.base import Segment
from lecture_analyzer.asr.funasr_backend import (
    FunASRBackend,
    _collect_segments,
    read_wav_float32,
    rebuild_sentences,
    usable_hotwords,
)
from lecture_analyzer.asr.llm_backend import LLMTranscriber
from lecture_analyzer.config import load_config
from lecture_analyzer.media import run_ffmpeg


def test_usable_hotwords_keeps_chinese_terms_only():
    terms = ["加速度", "Laplace 变换", "傅里叶变换", "加速度", "力", "二阶导数2", "  动量  "]
    assert usable_hotwords(terms) == ["加速度", "傅里叶变换", "二阶导数2", "动量"]


def test_read_wav_float32(tmp_path: Path):
    wav = tmp_path / "a.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-ar", "16000", "-ac", "1", str(wav)])
    audio = read_wav_float32(wav)
    assert audio.dtype == np.float32 and audio.shape[0] == 16000 and np.abs(audio).max() <= 1.0


def test_read_wav_rejects_wrong_rate(tmp_path: Path):
    wav = tmp_path / "b.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-ar", "44100", "-ac", "1", str(wav)])
    with pytest.raises(ASRError, match="16kHz"):
        read_wav_float32(wav)


class FakeAutoModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [{"sentence_info": [{"start": 0, "end": 1200, "text": "牛顿第二定律。"}, {"start": 1300, "end": 1500, "text": " "}]}]


def test_transcribe_passes_hotwords_and_collects_sentences(tmp_path: Path):
    wav = tmp_path / "c.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-ar", "16000", "-ac", "1", str(wav)])
    backend = FunASRBackend("paraformer", device="cpu")
    backend._model = FakeAutoModel()
    segments = backend.transcribe(wav, hotwords=["加速度", "Newton"])
    assert segments == [Segment(0, 1200, "牛顿第二定律。")]
    call = backend._model.calls[0]
    assert call["hotword"] == "加速度" and call["sentence_timestamp"] is True


def test_sensevoice_ignores_hotwords_and_passes_language(tmp_path: Path):
    wav = tmp_path / "d.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-ar", "16000", "-ac", "1", str(wav)])
    backend = FunASRBackend("sensevoice", device="cpu", language="ko")
    backend._model = FakeAutoModel()
    backend.transcribe(wav, hotwords=["加速度"])
    call = backend._model.calls[0]
    assert "hotword" not in call and call["language"] == "ko"


def test_paraformer_never_gets_language(tmp_path: Path):
    wav = tmp_path / "e.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-ar", "16000", "-ac", "1", str(wav)])
    backend = FunASRBackend("paraformer", device="cpu", language="ko")
    backend._model = FakeAutoModel()
    backend.transcribe(wav)
    assert "language" not in backend._model.calls[0]


def _chars_with_times(chars: str, step: int = 100):
    return list(chars), [[i * step, i * step + 80] for i in range(len(chars))]


def test_rebuild_sentences_fixes_shifted_punctuation():
    raw = "去看那么量反应曲线这个什么范围呢"
    tokens, times = _chars_with_times(raw)
    text = "去看，那么量反应曲线。这个什么范围呢？"
    assert rebuild_sentences(text, tokens, times) == [
        Segment(0, 880, "去看，那么量反应曲线。"),
        Segment(900, 1580, "这个什么范围呢？"),
    ]


def test_rebuild_sentences_breaks_long_run_on_at_comma():
    raw = "甲" * 55 + "乙" * 5
    tokens, times = _chars_with_times(raw)
    text = "甲" * 55 + "，" + "乙" * 5
    sentences = rebuild_sentences(text, tokens, times)
    assert [len(s.text) for s in sentences] == [56, 5]


def test_rebuild_sentences_handles_english_tokens():
    tokens = ["阿", "司", "匹", "林", "没", "gpcr", "我", "们"]
    times = [[i * 100, i * 100 + 80] for i in range(len(tokens))]
    sentences = rebuild_sentences("阿司匹林没 GPCR 我们。", tokens, times)
    assert sentences == [Segment(0, 780, "阿司匹林没GPCR 我们。")]


@pytest.mark.parametrize(
    ("text", "tokens", "times"),
    [
        ("去看，多了字。", list("去看"), [[0, 1], [1, 2]]),  # 文本里有对不上的字
        ("去看。", list("去看"), [[0, 1]]),  # 时间戳个数不一致
        ("去。", list("去看"), [[0, 1], [1, 2]]),  # 词没有用完
    ],
)
def test_rebuild_sentences_gives_up_on_mismatch(text, tokens, times):
    assert rebuild_sentences(text, tokens, times) is None


def test_collect_segments_prefers_rebuilt_sentences():
    tokens, times = _chars_with_times("所发生的效应")
    good = {"text": "所发生的效应。", "raw_text": " ".join(tokens), "timestamp": times,
            "sentence_info": [{"start": 0, "end": 1, "text": "所发，"}]}
    broken = {"text": "对不上", "raw_text": "完全 不同", "timestamp": [[0, 1], [1, 2]],
              "sentence_info": [{"start": 5, "end": 9, "text": "退回原结果"}]}
    assert _collect_segments([good, broken]) == [Segment(0, 580, "所发生的效应。"), Segment(5, 9, "退回原结果")]


def test_unknown_preset_rejected():
    with pytest.raises(ASRError, match="未知的 FunASR 预设"):
        FunASRBackend("whisper")


def test_create_backend_from_spec(example_config: Path, tmp_path: Path, monkeypatch):
    cfg = load_config(example_config)
    backend = create_backend(cfg, "funasr:sensevoice", work_dir=tmp_path)
    assert isinstance(backend, FunASRBackend) and backend.name == "funasr:sensevoice"
    assert create_backend(cfg, work_dir=tmp_path).name == "funasr:paraformer"  # 默认读 [asr]

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg.models["gemini"].model = "gemini-test"
    cfg.asr.llm_alternates = ["gemini-test", "gemini-alt", ""]  # 与主模型相同的和空的跳过
    llm_backend = create_backend(cfg, "llm:gemini", work_dir=tmp_path)
    assert isinstance(llm_backend, LLMTranscriber) and llm_backend.name == "llm:gemini"
    assert [p.profile.model for p in llm_backend.providers] == ["gemini-test", "gemini-alt"]

    with pytest.raises(ASRError, match="未知的转写后端"):
        create_backend(cfg, "whisper", work_dir=tmp_path)
