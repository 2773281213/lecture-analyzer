from pathlib import Path

import numpy as np
import pytest

from lecture_analyzer.asr import ASRError, create_backend
from lecture_analyzer.asr.base import Segment, load_segments, save_segments, segments_text
from lecture_analyzer.asr.qwen3_backend import Qwen3ASRBackend, resolve_model_dir
from lecture_analyzer.asr.vad import VadSegmenter
from lecture_analyzer.config import load_config
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.media import run_ffmpeg


class FakeVad(VadSegmenter):
    def __init__(self, spans):
        super().__init__("cpu")
        self.spans = spans

    def segment(self, audio):
        return self.spans


class FakeTensor:
    """模拟 generate 的输出：二维 token 序列，支持切片和 shape。"""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key):
        _, cols = key
        return FakeTensor([row[cols] for row in self.rows]) if isinstance(cols, slice) else None

    def __iter__(self):
        return iter(self.rows)


class FakeInputs(dict):
    def to(self, *args):
        return self


class FakeProcessor:
    def __init__(self, outputs):
        self.outputs = outputs
        self.requests = []

    def apply_transcription_request(self, audio, language=None, prompt=None):
        self.requests.append({"clips": [len(clip) for clip in audio], "language": language, "prompt": prompt})
        return FakeInputs(input_ids=FakeTensor([[0, 0]] * len(audio)))

    def decode(self, row, return_format="raw"):
        return self.outputs.pop(0)


class FakeModel:
    device, dtype = "cpu", None

    def generate(self, input_ids, max_new_tokens):
        return FakeTensor([[0, 0, 1] for _ in input_ids.rows])


def _backend(spans, outputs, **kwargs):
    backend = Qwen3ASRBackend("0.6b", device="cpu", vad=FakeVad(spans), batch_size=2, **kwargs)
    processor = FakeProcessor(outputs)
    backend._loaded = (processor, FakeModel())
    return backend, processor


@pytest.fixture
def two_second_wav(tmp_path: Path) -> Path:
    path = tmp_path / "a.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-ar", "16000", "-ac", "1", str(path)])
    return path


def test_segments_keep_vad_times_and_language(two_second_wav: Path):
    outputs = [
        {"language": "Korean", "transcription": "바이러스 명칭을 정할 때"},
        {"language": "Chinese", "transcription": "给病毒命名的时候"},
        {"language": None, "transcription": "  "},  # 空结果丢弃
    ]
    backend, processor = _backend([(0, 600), (700, 1500), (1600, 1900)], outputs)
    segments = backend.transcribe(two_second_wav, hotwords=["바이러스", "病毒", " "])
    assert segments == [
        Segment(0, 600, "바이러스 명칭을 정할 때", "ko"),
        Segment(700, 1500, "给病毒命名的时候", "zh"),
    ]
    assert processor.requests[0]["clips"] == [600 * 16, 800 * 16]  # 毫秒换算成采样点
    assert processor.requests[0]["prompt"] == ["바이러스、病毒"] * 2
    assert len(processor.requests) == 2  # batch_size=2，三个片段分两批


def test_forced_language_is_used_for_all_segments(two_second_wav: Path):
    backend, processor = _backend([(0, 900)], [{"language": None, "transcription": "안녕하세요"}], language="ko")
    assert backend.transcribe(two_second_wav)[0].lang == "ko"
    assert processor.requests[0]["language"] == "ko" and processor.requests[0]["prompt"] is None


def test_unknown_size_rejected():
    with pytest.raises(ASRError, match="未知的 Qwen3-ASR 型号"):
        Qwen3ASRBackend("7b")


def test_resolve_model_dir_by_hub(monkeypatch):
    import modelscope

    monkeypatch.setattr(modelscope, "snapshot_download", lambda model_id: f"D:/cache/{model_id}")
    assert resolve_model_dir("Qwen/Qwen3-ASR-0.6B-hf", "modelscope") == "D:/cache/Qwen/Qwen3-ASR-0.6B-hf"
    assert resolve_model_dir("Qwen/Qwen3-ASR-0.6B-hf", "huggingface") == "Qwen/Qwen3-ASR-0.6B-hf"
    with pytest.raises(ASRError, match="未知的模型来源"):
        resolve_model_dir("x", "github")


def test_vad_filters_short_spans():
    vad = VadSegmenter("cpu")

    class FakeVadModel:
        def generate(self, input):
            return [{"value": [[0, 300], [500, 2000], [2100, 2400], [3000, 3500]]}]

    vad._model = FakeVadModel()
    assert vad.segment(np.zeros(16000, dtype=np.float32)) == [(500, 2000), (3000, 3500)]


def test_create_qwen3_backend(example_config: Path, tmp_path: Path):
    cfg = load_config(example_config)
    small = create_backend(cfg, "qwen3:0.6b", work_dir=tmp_path)
    assert small.name == "qwen3:0.6b" and small.hub == "modelscope"
    assert create_backend(cfg, "qwen3", work_dir=tmp_path).name == "qwen3:1.7b"


@pytest.mark.parametrize(
    ("teaching", "qwen3_language", "sensevoice_language"),
    [
        (None, None, "auto"), ("ko", "ko", "ko"), ("zh", "zh", "zh"), ("ko+zh", None, "auto"), ("ko,en", None, "auto"),
        ("fr", "fr", "auto"),  # SenseVoice 不支持法语：自动识别
        ("zh-tw", "zh", "auto"),  # 地区代码：Qwen3 取语言部分
        ("sw", None, "auto"),  # 两个模型都不支持的语言：自动识别，不报错
    ],
)
def test_course_settings_decide_asr_language(example_config, tmp_path, teaching, qwen3_language, sensevoice_language):
    cfg = load_config(example_config)
    settings = CourseSettings(teaching=teaching) if teaching else None
    assert create_backend(cfg, "qwen3", work_dir=tmp_path, settings=settings).language == qwen3_language
    assert create_backend(cfg, "funasr:sensevoice", work_dir=tmp_path, settings=settings).language == sensevoice_language


def test_segment_lang_roundtrip_and_filter(tmp_path: Path):
    segments = [Segment(0, 1000, "선생님", "ko"), Segment(1000, 2000, "老师", "zh"), Segment(2000, 3000, "无语种")]
    path = tmp_path / "s.json"
    save_segments(segments, path)
    assert load_segments(path) == segments
    assert segments_text(segments, lang="ko") == "선생님"
    old_format = tmp_path / "old.json"  # 早期结果没有 lang 字段，也要能读
    old_format.write_text('[{"start_ms": 0, "end_ms": 10, "text": "旧"}]', encoding="utf-8")
    assert load_segments(old_format) == [Segment(0, 10, "旧", "")]
