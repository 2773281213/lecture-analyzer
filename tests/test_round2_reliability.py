"""第二轮改进（可靠性）：ffmpeg 原子输出、转写与资料计费、锁与任务执行器边界测试。"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from lecture_analyzer import llm as llm_module
from lecture_analyzer import locks, media
from lecture_analyzer.asr import fallback as asr_fallback
from lecture_analyzer.asr import funasr_backend, llm_backend, qwen3_backend
from lecture_analyzer.asr.base import ASRError
from lecture_analyzer.asr.llm_backend import LLMTranscriber, TranscriptChunk
from lecture_analyzer.config import load_config
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.job_runner import JobRunner, RunnerLock, log_path
from lecture_analyzer.jobs import JobStore
from lecture_analyzer.library import Library, LibraryError
from lecture_analyzer.llm import ContentBlockedError, FallbackProvider, LLMError, ModelProfile, RefusalError, Usage
from lecture_analyzer.llm.base import extract_json_text
from lecture_analyzer.materials import MaterialStore
from lecture_analyzer.media import MediaError, encode_mp3, extract_audio
from lecture_analyzer.pipeline import ingest, organize
from lecture_analyzer.pipeline import materials as materials_module
from lecture_analyzer.pipeline import run as run_module
from lecture_analyzer.pipeline.run import LectureRunner

from .conftest import FakeProvider, make_profile, make_wav
from .test_jobs import STUB, wait_for
from .test_material_items import _item, _items_json
from .test_materials import _digest_json, _pdf
from .test_pipeline_run import _run


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


# ---------- T7：锁文件打开重试上限、JobRunner 锁与日志清理、资料三步按实际作答档案记账 ----------


def test_lock_open_failure_is_not_retried_forever(tmp_path: Path, monkeypatch):
    occupied = tmp_path / "dir.lock"
    occupied.mkdir()
    monkeypatch.setattr(locks, "OPEN_RETRY_SECONDS", 0.5)

    started = time.monotonic()
    with pytest.raises(LibraryError, match="打不开锁文件"):
        with locks.hold(occupied, wait=None):
            pass
    assert time.monotonic() - started < 5.0

    # 进程内登记未被卡住：另一线程可正常拿到别的锁，且目录移除后也能拿到原锁路径
    other = tmp_path / "other.lock"
    occupied.rmdir()
    acquired = threading.Event()

    def worker() -> None:
        with locks.hold(other, wait=1.0), locks.hold(occupied, wait=1.0):
            acquired.set()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5.0)
    assert acquired.is_set()


def test_job_runner_start_failure_releases_lock(example_config: Path, monkeypatch):
    cfg = load_config(example_config)
    db = cfg.data_dir / "lecture.db"
    runner = JobRunner(cfg, db)

    def fail_pause(self: JobStore) -> int:
        raise RuntimeError("书库暂时不可写")

    monkeypatch.setattr(JobStore, "pause_interrupted", fail_pause)
    with pytest.raises(RuntimeError, match="书库暂时不可写"):
        runner.start()
    assert runner.owner is False

    # start 失败后执行器锁必须已释放，新的 RunnerLock 可以立刻拿到
    lock_file = log_path(cfg.data_dir, 0).with_name("runner.lock")
    probe = RunnerLock(lock_file)
    assert probe.acquire() is True
    probe.release()

    # 任务启动前已被取消：_run 不会留下 0 字节的空日志文件
    with Library(db) as library:
        course = library.add_course("测试课", CourseSettings())
        store = JobStore(library)
        job = store.add(course, "run", {})
        assert store.cancel(job.id)
    runner._run(job.id)
    assert not log_path(cfg.data_dir, job.id).exists()


def test_material_steps_bill_fallback_profile(m1_env, monkeypatch):
    _run(m1_env)  # 先生成课程记忆（含 K1「阿司匹林的剂量分段」）

    primary = make_profile(name="claude", model="claude-x", vision=True)
    backup = make_profile(name="gpt", model="gpt-x", vision=True)
    m1_env.cfg.models["claude"] = primary
    m1_env.cfg.steps["page_digest"] = "claude"
    m1_env.cfg.steps["material"] = "claude"

    def fallback_provider(response_json: str) -> FallbackProvider:
        return FallbackProvider(
            FakeProvider([RefusalError("blocked", Usage(7, 0))], primary),
            FakeProvider([response_json], backup),
        )

    pdf = _pdf(m1_env.tmp_path / "模拟题.pdf", pages=1, text="1. 阿司匹林小剂量的作用？")
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        store = MaterialStore(library)
        material = ingest.import_materials(
            library, m1_env.cfg.data_dir, course, [pdf], "mock_exam", report=lambda _: None
        ).materials[0]

        # 1. 逐页识别（page_digest）
        monkeypatch.setattr(ingest, "create_provider", lambda prof: fallback_provider(_digest_json([1])))
        ingest.digest_material(m1_env.cfg, library, material, report=lambda _: None)
        digest_rows = [
            dict(r) for r in library.conn.execute("SELECT model, input_tokens FROM llm_call WHERE step = 'page_digest'")
        ]
        assert digest_rows == [{"model": "gpt-x", "input_tokens": 17}]
        digest_detail = store.get(material.id).status["digest"]["detail"]
        assert "gpt-x" in digest_detail and "claude-x" not in digest_detail

        # 2. 拆题（ensure_items → step "material"）
        items_json = _items_json([_item("1", "阿司匹林小剂量的作用？", "short", score=10)])
        monkeypatch.setattr(materials_module, "create_provider", lambda prof: fallback_provider(items_json))
        materials_module.ensure_items(m1_env.cfg, library, store.get(material.id), report=lambda _: None)
        item_rows = [
            dict(r) for r in library.conn.execute("SELECT model, input_tokens FROM llm_call WHERE step = 'material'")
        ]
        assert item_rows == [{"model": "gpt-x", "input_tokens": 17}]
        items_detail = store.get(material.id).status["items"]["detail"]
        assert "gpt-x" in items_detail and "claude-x" not in items_detail

        # 3. 对应知识点（ensure_mapped → step "material"）
        mapping_json = json.dumps({"mappings": [{"item": 1, "kps": ["K1"]}]})
        monkeypatch.setattr(materials_module, "create_provider", lambda prof: fallback_provider(mapping_json))
        done = materials_module.ensure_mapped(m1_env.cfg, library, course, report=lambda _: None)
        assert done == 1
        all_material_rows = [
            dict(r) for r in library.conn.execute("SELECT model, input_tokens FROM llm_call WHERE step = 'material'")
        ]
        assert all_material_rows == [{"model": "gpt-x", "input_tokens": 17}, {"model": "gpt-x", "input_tokens": 17}]
        map_detail = store.get(material.id).status["kp_map"]["detail"]
        assert "gpt-x" in map_detail and "claude-x" not in map_detail


# ---------- 补齐上一轮测试缺口 ----------


def test_funasr_read_wav_without_numpy_hints_asr_extra(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(ASRError, match="uv sync --extra asr"):
        funasr_backend.read_wav_float32(tmp_path / "any.wav")


def test_qwen3_language_table_fallback_without_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    table = qwen3_backend._language_table()
    assert "fr" in table and "ja" in table


def test_delete_materials_deduplicates_and_discards_lock(m1_env, monkeypatch):
    data_dir = m1_env.cfg.data_dir
    removed_ids: list[int] = []
    real_remove = organize.remove_material

    def tracked_remove(library: Library, root: Path, material_id: int):
        removed_ids.append(material_id)
        return real_remove(library, root, material_id)

    monkeypatch.setattr(organize, "remove_material", tracked_remove)
    with Library(data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        material = MaterialStore(library).add(course, "slides", "重复资料", "dup.pdf")
        lock = locks.material_lock(data_dir, material.id)
        with locks.hold(lock):
            pass
        assert lock.exists()

        organize.delete_materials(library, data_dir, course, [material, material])
        assert removed_ids == [material.id]
        assert MaterialStore(library).list(course) == []
        assert not lock.exists()


def test_move_lecture_locks_course_memory_in_ascending_id_order(m1_env, monkeypatch):
    data_dir = m1_env.cfg.data_dir
    locked_order: list[int] = []
    real_memory_free = organize.memory_free

    def tracked_memory_free(root: Path, course, action: str):
        locked_order.append(course.id)
        return real_memory_free(root, course, action)

    monkeypatch.setattr(organize, "memory_free", tracked_memory_free)
    with Library(data_dir / "lecture.db") as library:
        first_course = library.course_by_id(m1_env.course_id)
        second_course = library.add_course("病毒学", CourseSettings(subject="病毒学"))
        assert first_course.id < second_course.id

        # 正向移动：小 id → 大 id
        lecture = library.lecture_by_id(m1_env.lecture_id)
        moved = organize.move_lecture(library, data_dir, lecture, second_course)
        assert locked_order == [first_course.id, second_course.id]

        # 反向移回：大 id → 小 id，加锁顺序仍按 course.id 升序
        locked_order.clear()
        organize.move_lecture(library, data_dir, moved.lecture, first_course)
        assert locked_order == [first_course.id, second_course.id]


def test_job_runner_queue_survives_course_deleted_mid_run(example_config: Path):
    cfg = load_config(example_config)
    db = cfg.data_dir / "lecture.db"
    with Library(db) as library:
        course_a = library.add_course("要删的课", CourseSettings())
        course_b = library.add_course("留下的课", CourseSettings())

    runner = JobRunner(cfg, db, command=lambda job_id: [sys.executable, str(STUB), str(db), str(job_id)])
    runner.start()
    try:
        with Library(db) as library:
            job_a = JobStore(library).add(course_a, "run", {"sleep": 60})
        runner.submit(job_a.id)

        def job_a_started() -> bool:
            with Library(db) as lib:
                return JobStore(lib).get(job_a.id).progress == "子进程已启动"

        wait_for(job_a_started, timeout=15)
        assert job_a.id in runner._processes

        # 任务运行期间删除它所属的课程（级联删除任务记录）
        with Library(db) as library:
            library.delete_course(course_a.id)
        wait_for(lambda: job_a.id not in runner._processes, timeout=15)

        # 再往同一条重任务队列提交另一门课的任务，验证队列线程仍存活并正常完成
        with Library(db) as library:
            job_b = JobStore(library).add(course_b, "run", {})
        runner.submit(job_b.id)

        def job_b_done() -> bool:
            with Library(db) as lib:
                return JobStore(lib).get(job_b.id).state == "done"

        wait_for(job_b_done, timeout=15)
    finally:
        runner.shutdown()
