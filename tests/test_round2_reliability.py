"""第二轮改进（可靠性）：ffmpeg 原子输出、转写与资料计费、锁与任务执行器边界测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lecture_analyzer import llm as llm_module
from lecture_analyzer import media
from lecture_analyzer.asr import fallback as asr_fallback
from lecture_analyzer.asr import llm_backend
from lecture_analyzer.asr.llm_backend import LLMTranscriber, TranscriptChunk
from lecture_analyzer.config import load_config
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.library import Library
from lecture_analyzer.llm import ContentBlockedError, LLMError, ModelProfile, Usage
from lecture_analyzer.llm.base import extract_json_text
from lecture_analyzer.media import MediaError, encode_mp3, extract_audio
from lecture_analyzer.pipeline import run as run_module
from lecture_analyzer.pipeline.run import LectureRunner

from .conftest import FakeProvider, make_profile, make_wav


def _transcript_json(*lines: tuple[str, str, str, str]) -> str:
    items = [{"start": s, "end": e, "lang": lang, "text": text} for s, e, lang, text in lines]
    return json.dumps({"lines": items}, ensure_ascii=False)


# ---------- T6：ffmpeg 原子输出、转写逐次记账、补转容错与 JSON 提取 ----------


def test_ffmpeg_output_is_atomic(tmp_path: Path, monkeypatch):
    wav = make_wav(tmp_path / "src.wav", 2)
    out = tmp_path / "out"

    def fail_midway(args: list[str]) -> None:
        Path(args[-1]).write_bytes(b"partial")
        raise MediaError("模拟进程被强杀")

    monkeypatch.setattr(media, "run_ffmpeg", fail_midway)
    with pytest.raises(MediaError, match="模拟进程被强杀"):
        encode_mp3(wav, out / "a.mp3")
    assert list(out.iterdir()) == []

    with pytest.raises(MediaError, match="模拟进程被强杀"):
        extract_audio(wav, out / "a.wav")
    assert list(out.iterdir()) == []

    monkeypatch.undo()
    mp3 = encode_mp3(wav, out / "a.mp3")
    extracted = extract_audio(wav, out / "a.wav")
    assert mp3.is_file() and extracted.is_file()
    assert sorted(p.name for p in out.iterdir()) == ["a.mp3", "a.wav"]


def test_transcribe_logs_usage_per_request_when_later_part_fails(m1_env, tmp_path: Path, monkeypatch):
    second_wav = make_wav(tmp_path / "药理学3-2.wav", 2)
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        library.add_part(m1_env.lecture_id, second_wav)

    first_ok = _transcript_json(("00:00", "00:01", "zh", "第一节内容"))
    provider = FakeProvider([first_ok, LLMError("中转站不可用")], make_profile(audio=True, model="asr-model"))

    def fake_backend(cfg, spec=None, *, work_dir, settings=None):
        return LLMTranscriber(provider, work_dir, settings=settings)

    monkeypatch.setattr(run_module, "create_backend", fake_backend)
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        lecture = library.lecture_by_id(m1_env.lecture_id)
        with pytest.raises(LLMError, match="中转站不可用"):
            LectureRunner(m1_env.cfg, library, lecture, report=lambda _: None).run()
        summary = library.cost_summary()
        assert len(summary) == 1
        assert summary[0]["step"] == "transcribe" and summary[0]["model"] == "asr-model"
        assert summary[0]["input_tokens"] == 10
        assert library.lecture_by_id(m1_env.lecture_id).step_state("transcribe") == "failed"

    # 0 用量（例如请求阶段被内容审核拦截）不写入空的用量回调
    monkeypatch.setattr(llm_backend, "MIN_PIECE_SECONDS", 10)
    recorded: list[tuple[ModelProfile, Usage]] = []
    blocked_transcriber = LLMTranscriber(
        FakeProvider([ContentBlockedError("blocked")], make_profile(audio=True)),
        tmp_path / "blocked_work",
        chunk_minutes=1 / 20,
    )
    blocked_transcriber.on_usage = lambda prof, usage: recorded.append((prof, usage))
    assert blocked_transcriber.transcribe(second_wav) == []
    assert blocked_transcriber.skipped == [(0, 2000)]
    assert recorded == []


def test_fill_skipped_logs_llm_usage_and_handles_unknown_profile(example_config: Path, tmp_path: Path, monkeypatch):
    wav = make_wav(tmp_path / "audio.wav", 3)
    cfg = load_config(example_config)

    # (a) 方案里写错的档案名（ConfigError）应被兜住并保留缺口
    cfg.asr.fallback = "llm:typo"
    filled, note = asr_fallback.fill_skipped(cfg, CourseSettings(), wav, [(0, 1000)], tmp_path / "fb_typo")
    assert filled == [] and "本地补转失败" in note and "typo" in note

    # (b) 有效的音频档案补转时，应通过 on_usage 回调记录非 0 用量
    audio_profile = make_profile(name="audio_fb", model="fb-audio-model", audio=True)
    cfg.models["audio_fb"] = audio_profile
    cfg.asr.fallback = "llm:audio_fb"
    chunk = _transcript_json(("00:00", "00:00.8", "zh", "补转出来的句子"))
    monkeypatch.setattr(llm_module, "create_provider", lambda profile: FakeProvider([chunk], profile))

    logged: list[tuple[ModelProfile, Usage]] = []
    filled, note = asr_fallback.fill_skipped(
        cfg, CourseSettings(), wav, [(0, 1000)], tmp_path / "fb_ok",
        on_usage=lambda profile, usage: logged.append((profile, usage)),
    )
    assert len(filled) == 1 and filled[0].text == "补转出来的句子"
    assert len(logged) == 1
    assert logged[0][0].model == "fb-audio-model"
    assert logged[0][1].input_tokens > 0


def test_extract_json_text_ignores_braces_in_surrounding_prose():
    assert extract_json_text('说明 {1} 如下：{"a": 1}') == '{"a": 1}'
    assert extract_json_text('{"a": 1} 备注：{完成}') == '{"a": 1}'
    fenced = '好的：\n```json\n{"a": {"b": [1, 2]}}\n```\n注：{ok}'
    assert extract_json_text(fenced) == '{"a": {"b": [1, 2]}}'
    embedded_fence = json.dumps(
        {"found": True, "answer_md": "写成\n```json\n{}\n```", "supplement_md": ""},
        ensure_ascii=False,
    )
    assert extract_json_text(embedded_fence) == embedded_fence


def test_extract_json_text_truncated_output_keeps_old_slice():
    truncated = (
        '{"lines": ['
        '{"start": "00:00", "end": "00:01", "lang": "zh", "text": "第一句"}, '
        '{"start": "00:01", "end": "00:02", "lang": "zh", "text": "第二句"}, '
        '{"start": "00:02", "end": "00:03", "lang": "zh", "text": "第三句"}, '
        '{"start": "00:03", "end": "00:04", "lang": "zh", "text": "被截断'
    )
    sliced = extract_json_text(truncated)
    assert sliced == truncated[: truncated.rfind("}") + 1]
    with pytest.raises(ValueError, match="EOF while parsing"):
        TranscriptChunk.model_validate_json(sliced)

    deep = '{"a": ' + "[" * 100000 + "]" * 100000 + "}"
    assert extract_json_text(deep) == deep
