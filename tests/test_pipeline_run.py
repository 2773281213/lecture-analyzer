import json

import pytest

from lecture_analyzer.asr.base import Segment
from lecture_analyzer.library import Library
from lecture_analyzer.llm import LLMError
from lecture_analyzer.pipeline.run import STEPS, LectureRunner, RunOptions, _gap_warning

from .conftest import make_wav


def _run(env, **options):
    messages: list[str] = []
    with Library(env.cfg.data_dir / "lecture.db") as library:
        lecture = library.lecture_by_id(env.lecture_id)
        LectureRunner(env.cfg, library, lecture, RunOptions(**options), report=messages.append).run()
        return library.lecture_by_id(env.lecture_id), messages


def test_full_run_produces_book(m1_env):
    lecture, messages = _run(m1_env)
    states = {step: lecture.step_state(step) for step in STEPS}
    assert states == {"media": "done", "slides": "skipped", "transcribe": "done", "extract": "done",
                      "validate": "done", "integrate": "done", "book": "done"}
    assert lecture.title == "量效关系" and abs(lecture.duration_ms - 3000) < 100
    assert "原话校验通过 1/2（50%）" in lecture.status["validate"]["detail"]
    assert "新增 1 个知识点" in lecture.status["integrate"]["detail"]

    course_folder = m1_env.cfg.data_dir / "books" / "药理学"
    data = json.loads((course_folder / "第03讲" / "lecture.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 3 and data["course"]["name"] == "药理学"
    assert data["lecture"]["media_type"] == "audio" and len(data["transcript"]) == 3
    assert [s["valid"] for s in data["signals"]] == [True, False] and data["signals"][0]["kp_key"] == 1
    assert data["memory"]["knowledge_points"] == [{"note_index": 1, "kp_key": 1}]
    markdown = (course_folder / "第03讲" / "笔记.md").read_text(encoding="utf-8")
    assert "药理学 第3讲：量效关系" in markdown and "未找到原话" in markdown

    book = json.loads((course_folder / "book.json").read_text(encoding="utf-8"))
    assert book["knowledge_points"][0]["name"] == "阿司匹林的剂量分段"
    assert book["knowledge_points"][0]["chapter"] == "药理学总论"
    assert "药理学总论" in (course_folder / "课程总览.md").read_text(encoding="utf-8")

    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        assert [row["step"] for row in library.cost_summary()] == ["extract", "integrate"]


def test_second_run_skips_finished_steps(m1_env):
    _run(m1_env)
    _, messages = _run(m1_env)
    assert all(message.startswith("跳过") for message in messages)
    assert m1_env.asr.calls == 1 and len(m1_env.provider_calls) == 2  # 整理笔记、整合各一次


def test_force_reruns_step_and_dependents_only(m1_env):
    _run(m1_env)
    _run(m1_env, force=frozenset({"extract"}))
    assert m1_env.asr.calls == 1  # 转写没有重跑
    assert len(m1_env.provider_calls) == 4  # 笔记和整合都重新做了


def test_asr_cache_kept_on_retry_and_force_but_not_when_source_changes(m1_env):
    marker = m1_env.cfg.data_dir / "work" / f"lecture-{m1_env.lecture_id}" / "asr" / "part-1_chunks_600s" / "x.json"
    _run(m1_env)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        library.set_step(m1_env.lecture_id, "transcribe", "failed", "模拟失败")
    _run(m1_env)  # 失败后续跑：缓存保留
    assert marker.exists()
    _run(m1_env, force=frozenset({"transcribe"}))  # 强制重跑：仍然有效的缓存保留，只重试没有结果的片段
    assert marker.exists() and m1_env.asr.calls == 3
    manifest = marker.parent.parent / "part-1.json"
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps({**saved, "hotwords": ["乙", "甲", *saved["hotwords"]]}), encoding="utf-8")
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        library.set_step(m1_env.lecture_id, "transcribe", "failed", "模拟失败")
    _run(m1_env)  # 热词变了：缓存作废
    assert not marker.exists()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps({**saved, "hotwords": saved["hotwords"][::-1]}), encoding="utf-8")
    _run(m1_env, force=frozenset({"transcribe"}))  # 只是顺序不同（旧版本写下的顺序是随机的）：缓存保留
    assert marker.exists()
    make_wav(m1_env.media, 2)  # 换了源文件：这一节的缓存作废
    _run(m1_env, force=frozenset({"media"}))
    assert not marker.exists()


def test_added_part_keeps_earlier_cache_and_offsets_timeline(m1_env):
    _run(m1_env)
    work = m1_env.cfg.data_dir / "work" / f"lecture-{m1_env.lecture_id}"
    marker = work / "asr" / "part-1_chunks_600s" / "x.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")
    wav_mtime = (work / "parts" / "part-1.wav").stat().st_mtime_ns
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        library.add_part(m1_env.lecture_id, make_wav(m1_env.tmp_path / "药理学3-2.wav", 2))
    lecture, _ = _run(m1_env)
    assert marker.exists()  # 第 1 节的分片缓存保留
    assert (work / "parts" / "part-1.wav").stat().st_mtime_ns == wav_mtime  # 第 1 节没有重新抽取
    assert lecture.step_state("slides") == "skipped" and "2 节" in lecture.status["media"]["detail"]
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        segments = library.segments(m1_env.lecture_id)
    assert len(segments) == 6 and segments[3].start_ms == lecture.part_offsets()[1]
    extractions = [p for p in m1_env.provider_calls if p.calls and p.calls[0]["schema"].__name__ == "LectureNotes"]
    transcript = extractions[-1].calls[0]["parts"][1].text  # 补上第 2 节后重新整理笔记的那次
    assert "—— 第 1 节 ——" in transcript and "—— 第 2 节 ——\n[S4]" in transcript


def test_skipped_ranges_go_to_local_fallback(m1_env, monkeypatch):
    from lecture_analyzer.asr import fallback

    m1_env.asr.skipped = [(3000, 3500)]
    monkeypatch.setattr(fallback, "create_backend", lambda *a, **kw: m1_env.asr)
    lecture, _ = _run(m1_env)
    assert "被上游拦截，已用 fake-asr 补转 3 句" in lecture.status["transcribe"]["detail"]
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        assert len(library.segments(lecture.id)) == 6


def test_model_override_switches_extract_and_integrate_profile(m1_env):
    lecture, _ = _run(m1_env, model="gemini")
    assert {p.profile.name for p in m1_env.provider_calls} == {"gemini"}
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        assert library.notes_model(lecture.id)[0] == "gemini"
        assert [row["step"] for row in library.cost_summary()] == ["extract", "integrate"]


def test_failure_is_recorded(m1_env, monkeypatch):
    from lecture_analyzer.pipeline import run as run_module

    class Broken:
        def __init__(self, profile):
            self.profile = profile

        def generate_structured(self, *args, **kwargs):
            raise LLMError("中转站不可用")

    monkeypatch.setattr(run_module, "create_provider", Broken)
    with pytest.raises(LLMError):
        _run(m1_env)
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        lecture = library.lecture_by_id(m1_env.lecture_id)
    assert lecture.step_state("transcribe") == "done"
    assert lecture.step_state("extract") == "failed" and "中转站不可用" in lecture.status["extract"]["detail"]


def test_gap_warning():
    segments = [Segment(0, 1000, "a"), Segment(1000, 2000, "b"), Segment(200_000, 201_000, "c")]
    assert "03:20" not in _gap_warning(segments, 202_000)
    assert "00:02 起有 3.3 分钟" in _gap_warning(segments, 202_000)
    assert _gap_warning(segments[:2], 2_500) == ""
    assert "02:01" not in _gap_warning(segments[:2], 125_000) and "2.0 分钟" in _gap_warning(segments[:2], 125_000)


def test_auto_title_follows_new_notes_but_edited_title_stays(m1_env):
    """改了笔记语言重新整理笔记后，自动生成的标题跟着更新；手动改过的标题保留。"""
    from .conftest import LECTURE_NOTES, integration_json

    lecture, _ = _run(m1_env)
    assert lecture.title == "量效关系"
    renamed = {**LECTURE_NOTES, "title": "Dose-response relationship"}
    m1_env.routes["LectureNotes"] = [json.dumps(renamed, ensure_ascii=False)]
    m1_env.routes["IntegrationResult"] = [integration_json(renamed, chapter="药理学总论")]
    lecture, _ = _run(m1_env, force=frozenset({"extract"}))
    assert lecture.title == "Dose-response relationship"  # 原来的标题就是上次笔记生成的
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        library.update_lecture(m1_env.lecture_id, title="我改过的标题")
    m1_env.routes["LectureNotes"] = [json.dumps(LECTURE_NOTES, ensure_ascii=False)]
    lecture, _ = _run(m1_env, force=frozenset({"extract"}))
    assert lecture.title == "我改过的标题"
