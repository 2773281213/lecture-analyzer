import json
from pathlib import Path

import pytest

from lecture_analyzer.asr import llm_backend
from lecture_analyzer.asr.base import ASRError, Segment, load_segments, save_segments, segments_text
from lecture_analyzer.asr.llm_backend import LLMTranscriber
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.llm import AudioPart, EmptyResponseError, TextPart, Usage
from lecture_analyzer.media import run_ffmpeg

from .conftest import FakeProvider, make_profile


@pytest.fixture
def three_second_wav(tmp_path: Path) -> Path:
    path = tmp_path / "lecture.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-ar", "16000", "-ac", "1", str(path)])
    return path


def _chunk(*lines):
    items = [{"start": s, "end": e, "lang": lang, "text": t} for s, e, lang, t in lines]
    return json.dumps({"lines": items}, ensure_ascii=False)


def test_llm_transcriber_offsets_each_chunk(three_second_wav: Path, tmp_path: Path):
    responses = [
        _chunk(("00:00", "00:00.6", "ko", "바이러스"), ("bad", "00:00.9", "ZH", "时间坏了")),
        _chunk(("00:00.2", "00:00.8", "zh", "第二段")),
        _chunk(("00:00", "00:01", "zh", "  ")),  # 空句子会被丢弃
        _chunk(),  # 备用：编码误差可能多出一个极短切片
    ]
    provider = FakeProvider(responses, make_profile(audio=True))
    settings = CourseSettings(subject="病毒学", teaching="ko+zh", note_language="ko")
    transcriber = LLMTranscriber(provider, tmp_path / "work", chunk_minutes=1 / 60, settings=settings)
    progress: list[str] = []
    transcriber.progress = progress.append
    segments = transcriber.transcribe(three_second_wav, hotwords=["加速度", "动量"])
    assert progress[:3] == [f"第 {i}/{len(progress)} 片" for i in (1, 2, 3)]  # 每一片开始时报告进度

    assert segments == [
        Segment(0, 600, "바이러스", "ko"),
        Segment(600, 900, "时间坏了", "zh"),  # 起点无法解析时沿用上一句结束时间；语种转小写
        Segment(1200, 1800, "第二段", "zh"),
    ]
    first_call = provider.calls[0]
    audio = first_call["parts"][0]
    assert isinstance(audio, AudioPart) and audio.mime_type == "audio/mp3" and audio.data[:3] == b"ID3"
    assert isinstance(first_call["parts"][1], TextPart) and "加速度、动量" in first_call["parts"][1].text
    assert "病毒学" in first_call["system"] and "现场有人翻译成中文" in first_call["system"]
    assert transcriber.usage.input_tokens == 10 * len(provider.calls)


def test_cached_chunks_are_not_requested_again(three_second_wav: Path, tmp_path: Path):
    responses = [_chunk(("00:00", "00:01", "zh", "缓存的句子")), _chunk(), _chunk(), _chunk()]
    provider = FakeProvider(responses, make_profile(audio=True))
    transcriber = LLMTranscriber(provider, tmp_path / "work", chunk_minutes=1 / 20)  # 3 秒一段，整段一片
    first = transcriber.transcribe(three_second_wav)
    calls = len(provider.calls)
    assert transcriber.transcribe(three_second_wav) == first == [Segment(0, 1000, "缓存的句子", "zh")]
    assert len(provider.calls) == calls  # 第二次全部命中缓存


def test_blocked_piece_is_split_and_retried(three_second_wav: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(llm_backend, "MIN_PIECE_SECONDS", 1)
    responses = [
        EmptyResponseError("拦截", Usage(15, 0)),  # 整段（3 秒）被拦
        _chunk(("00:00", "00:01", "zh", "前半段")),  # 前半段 1.5 秒
        _chunk(("00:00", "00:01", "zh", "后半段")),  # 后半段 1.5 秒
    ]
    provider = FakeProvider(responses, make_profile(audio=True))
    transcriber = LLMTranscriber(provider, tmp_path / "work", chunk_minutes=1 / 20)
    segments = transcriber.transcribe(three_second_wav)
    assert [s.text for s in segments] == ["前半段", "后半段"]
    assert segments[1].start_ms == 1500 and transcriber.skipped == []
    assert transcriber.usage.input_tokens == 15 + 20  # 被拦截那次消耗的输入也计入


def test_piece_cut_off_twice_is_split_like_a_blocked_one(three_second_wav: Path, tmp_path: Path, monkeypatch):
    # M4 验收实测：上游过滤在生成中途截断输出，带修正提示重试仍在同一处断开
    monkeypatch.setattr(llm_backend, "MIN_PIECE_SECONDS", 1)
    cut_off = '{"lines": [{"start": "00:00", "end": "00:01", "lang": "zh", "text": "前'
    responses = [cut_off, cut_off, _chunk(("00:00", "00:01", "zh", "前半段")), _chunk(("00:00", "00:01", "zh", "后半段"))]
    provider = FakeProvider(responses, make_profile(audio=True))
    transcriber = LLMTranscriber(provider, tmp_path / "work", chunk_minutes=1 / 20)
    assert [s.text for s in transcriber.transcribe(three_second_wav)] == ["前半段", "后半段"]
    assert transcriber.skipped == [] and transcriber.usage.input_tokens == 10 * 4  # 截断的两次也计入


def test_mostly_unclear_results_are_not_accepted(three_second_wav: Path, tmp_path: Path, monkeypatch):
    # M4 验收：备用型号把老师的韩语原话写成「[听不清]」或空句，只留下翻译的中文
    monkeypatch.setattr(llm_backend, "MIN_PIECE_SECONDS", 10)
    unclear = _chunk(("00:00", "00:01", "ko", "[听不清]"), ("00:01", "00:02", "ko", " "),
                     ("00:02", "00:03", "zh", "翻译的中文"))
    primary = FakeProvider([EmptyResponseError("拦截")], make_profile(audio=True))
    alternate = FakeProvider([unclear], make_profile(audio=True, model="alt-model"))
    transcriber = LLMTranscriber(primary, tmp_path / "work", chunk_minutes=1 / 20, alternates=[alternate])
    assert transcriber.transcribe(three_second_wav) == [] and transcriber.skipped == [(0, 3000)]  # 交给本地模型补转
    assert not list((tmp_path / "work").rglob("*.alt-model.json"))  # 不可用的结果不缓存

    lines = llm_backend.TranscriptChunk.model_validate_json
    assert llm_backend.unusable(lines(unclear).lines)
    half = _chunk(("00:00", "00:01", "ko", "[听不清] 바이러스"), ("00:01", "00:02", "ko", "【聽不清】"))
    assert not llm_backend.unusable(lines(half).lines) and not llm_backend.unusable([])  # 整段静音不算


def test_interpreted_piece_without_teacher_language_is_not_accepted(three_second_wav: Path, tmp_path: Path,
                                                                    monkeypatch):
    # M4 验收：备用型号漏掉老师的韩语原话，只留下翻译的中文
    monkeypatch.setattr(llm_backend, "MIN_PIECE_SECONDS", 10)
    lines = llm_backend.TranscriptChunk.model_validate_json
    only_interpreter = _chunk(*[("00:00", "00:01", "zh", f"翻译的第 {n} 句") for n in range(5)])
    settings = CourseSettings(subject="病毒学", teaching="ko", interpreted_to="zh", note_language="zh")
    transcriber = LLMTranscriber(FakeProvider([only_interpreter], make_profile(audio=True)), tmp_path / "work",
                                 chunk_minutes=1 / 20, settings=settings)
    assert transcriber.transcribe(three_second_wav) == [] and transcriber.skipped == [(0, 3000)]
    with_teacher = lines(_chunk(*[("00:00", "00:01", "zh", "翻译")] * 4, ("00:01", "00:02", "ko-KR", "바이러스"))).lines
    assert not llm_backend.unusable(with_teacher, ["ko"])
    assert not llm_backend.unusable(lines(only_interpreter).lines)  # 没有现场翻译的课不检查语种
    assert not llm_backend.unusable(lines(only_interpreter).lines[:4], ["ko"])  # 句子太少：只有翻译也正常


def test_bad_caches_are_ignored(three_second_wav: Path, tmp_path: Path):
    good = _chunk(("00:00", "00:01", "ko", "바이러스"))
    first = LLMTranscriber(FakeProvider([good], make_profile(audio=True)), tmp_path / "work", chunk_minutes=1 / 20)
    first.transcribe(three_second_wav)
    [cache] = (tmp_path / "work").rglob("*.json")
    cache.write_text('{"lines": [{"start": "00:00"', encoding="utf-8")  # 写到一半时任务被取消
    again = FakeProvider([good], make_profile(audio=True))
    second = LLMTranscriber(again, tmp_path / "work", chunk_minutes=1 / 20)
    assert [s.text for s in second.transcribe(three_second_wav)] == ["바이러스"] and len(again.calls) == 1


def test_piece_blocked_at_minimum_length_is_skipped(three_second_wav: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(llm_backend, "MIN_PIECE_SECONDS", 10)
    provider = FakeProvider([EmptyResponseError("拦截")], make_profile(audio=True))
    transcriber = LLMTranscriber(provider, tmp_path / "work", chunk_minutes=1 / 20)
    assert transcriber.transcribe(three_second_wav) == []
    assert transcriber.skipped == [(0, 3000)]


def test_alternate_model_only_for_shortest_pieces(three_second_wav: Path, tmp_path: Path, monkeypatch):
    """备用模型转写长片段会漏内容：长片段被拦截时只对半切，切到最短仍被拦截才换备用模型。"""
    monkeypatch.setattr(llm_backend, "MIN_PIECE_SECONDS", 2)  # 3 秒一片 → 切成两个 1.5 秒的最短片段
    primary = FakeProvider([
        EmptyResponseError("拦截", Usage(15, 0)),  # 整片被拦
        _chunk(("00:00", "00:01", "ko", "주 모델")),  # 前半段主模型成功
        EmptyResponseError("拦截", Usage(5, 0)),  # 后半段主模型被拦
    ], make_profile(audio=True))
    alternate = FakeProvider([_chunk(("00:00", "00:01", "ko", "다른 모델"))], make_profile(audio=True, model="alt-model"))
    transcriber = LLMTranscriber(primary, tmp_path / "work", chunk_minutes=1 / 20, alternates=[alternate])
    segments = transcriber.transcribe(three_second_wav)
    assert [s.text for s in segments] == ["주 모델", "다른 모델"] and segments[1].start_ms == 1500
    assert transcriber.skipped == [] and len(alternate.calls) == 1  # 备用模型只转写了最短的那一片
    assert {model: usage.input_tokens for model, (_, usage) in transcriber.usage_by_model.items()} == {
        "fake-model": 30, "alt-model": 10}
    # 再转写一次：各片的缓存（包括备用模型的）都会被用上，不再请求
    again = LLMTranscriber(FakeProvider([EmptyResponseError("拦截")], make_profile(audio=True)), tmp_path / "work",
                           chunk_minutes=1 / 20, alternates=[FakeProvider([], make_profile(audio=True, model="alt-model"))])
    assert [s.text for s in again.transcribe(three_second_wav)] == ["주 모델", "다른 모델"]


def test_llm_transcriber_requires_audio_capability(tmp_path: Path):
    with pytest.raises(ASRError, match="不支持音频"):
        LLMTranscriber(FakeProvider([], make_profile(audio=False)), tmp_path)


def test_segments_roundtrip_and_window(tmp_path: Path):
    segments = [Segment(0, 1000, "甲"), Segment(5000, 6000, "乙"), Segment(12000, 13000, "丙")]
    path = tmp_path / "out" / "segments.json"
    save_segments(segments, path)
    assert load_segments(path) == segments
    assert segments_text(segments, 1000, 12000) == "乙"
    assert segments_text(segments) == "甲乙丙"
