"""评审改进项的回归与边界测试。"""

from __future__ import annotations

import io
import json
import struct
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pymupdf
import pytest
from google.genai import errors as genai_errors

from lecture_analyzer import archive, books, locks, qa_attachments
from lecture_analyzer.asr import models as asr_models
from lecture_analyzer.asr import resolve_device
from lecture_analyzer.asr.base import Segment
from lecture_analyzer.config import ConfigError, load_config
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.documents import pixmap_to_png
from lecture_analyzer.job_runner import JobRunner
from lecture_analyzer.jobs import JobStore
from lecture_analyzer.knowledge import Reference
from lecture_analyzer.library import Library, LibraryError
from lecture_analyzer.llm import FallbackProvider, ModelProfile, RefusalError, TextPart, create_provider
from lecture_analyzer.llm.anthropic_adapter import AnthropicProvider
from lecture_analyzer.llm.gemini_adapter import GeminiProvider
from lecture_analyzer.llm.openai_compat import OpenAICompatProvider
from lecture_analyzer.llm.probe import ping
from lecture_analyzer.materials import Material, MaterialStore, NewItem
from lecture_analyzer.pipeline import exam_plan as exam_plan_module
from lecture_analyzer.pipeline import extract as extract_module
from lecture_analyzer.pipeline import slides as slides_module
from lecture_analyzer.pipeline.exam_plan import to_plan
from lecture_analyzer.pipeline.memory import Item, Section, assemble
from lecture_analyzer.pipeline.organize import delete_course
from lecture_analyzer.pipeline.report_render import render_report_markdown
from lecture_analyzer.qa_attachments import (
    AttachmentError, get_attachment, read_attachments, root_of, save_attachment,
)
from lecture_analyzer.render import render_references
from lecture_analyzer.schemas import ExamPlanOutput, LectureNotes, PageDigest
from lecture_analyzer.timecode import parse_timecode
from lecture_analyzer.web.rendering import render_markdown

from .conftest import LECTURE_NOTES, FakeProvider, make_profile
from .test_exam_plan import PLAN_OUTPUT, _plan_form
from .test_web import _post
from .test_web_analysis import _client


# ---------- ASR 延迟加载与下载进度 ----------


def test_asr_resolve_device_without_torch_and_progress_ignores_missing(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_device("auto") == "cpu"

    # 模拟下载过程中临时文件被并发重命名而触发 OSError
    ghost = tmp_path / "gone.bin"
    ghost.write_bytes(b"123")
    real_stat = Path.stat

    def flaky_stat(self: Path, *args, **kwargs):
        if self == ghost:
            raise FileNotFoundError("vanished")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    assert asr_models._file_size(ghost) == 0


# ---------- LLM 延迟备用、Gemini 重试、Anthropic 域名与用量累计 ----------


def test_fallback_is_lazy_and_probe_ignores_fallback(monkeypatch):
    fallback = make_profile(name="fb", adapter="openai_compat", model="fb-model", api_key_env="MISSING_FB_KEY")
    primary = make_profile(name="pri", adapter="openai_compat", model="pri-model", api_key_env="PRI_KEY",
                           fallback_profile=fallback)
    monkeypatch.setenv("PRI_KEY", "sk-primary")
    monkeypatch.delenv("MISSING_FB_KEY", raising=False)

    # 仅创建主模型时不应提前初始化备用档案（否则缺少备用密钥会直接失败）
    provider = create_provider(primary)
    assert isinstance(provider, FallbackProvider)
    assert provider._fallback is None

    # 探活只测目标档案本身，不触发备用回退
    seen: list[ModelProfile] = []

    def fake_create(prof: ModelProfile):
        seen.append(prof)
        return FakeProvider([RefusalError("内容被拒答")], prof)

    monkeypatch.setattr("lecture_analyzer.llm.probe.create_provider", fake_create)
    ok, message = ping(primary)
    assert not ok and "拒答" in message
    assert len(seen) == 1 and seen[0].fallback_profile is None


def test_gemini_retries_transient_api_error_and_transport_error(monkeypatch):
    profile = make_profile(name="gem", adapter="gemini", model="gemini-2.5-pro", api_key_env="GEM_KEY")
    monkeypatch.setenv("GEM_KEY", "fake-key")
    captured: dict[str, object] = {}
    attempts = 0

    class FakeModels:
        def generate_content(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise genai_errors.ServerError(503, {"error": {"message": "overloaded"}}, None)
            if attempts == 2:
                raise httpx.ConnectError("connection reset")
            return SimpleNamespace(
                text="ok",
                candidates=[SimpleNamespace(finish_reason="STOP")],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12, candidates_token_count=4, thoughts_token_count=0,
                    cached_content_token_count=0,
                ),
            )

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.models = FakeModels()

    monkeypatch.setattr("lecture_analyzer.llm.gemini_adapter.genai.Client", FakeClient)
    provider = GeminiProvider(profile)
    result = provider.generate_text([TextPart("hi")])
    assert result.output == "ok" and attempts == 3
    assert result.usage.input_tokens == 12 and result.usage.output_tokens == 4
    assert captured["http_options"].retry_options.attempts == 1


def test_anthropic_refusal_fallback_checks_exact_official_host():
    def enabled_for(url: str | None) -> bool:
        profile = make_profile(name="cl", adapter="anthropic", model="claude-test", base_url=url)
        return AnthropicProvider(profile, client=object())._use_refusal_fallback()

    assert enabled_for(None) is True
    assert enabled_for("https://api.anthropic.com/") is True
    assert enabled_for("https://API.ANTHROPIC.COM/v1") is True
    assert enabled_for("https://api.anthropic.com.relay.example/v1") is False
    assert enabled_for("https://relay.example/api.anthropic.com") is False


def test_openai_compat_accumulates_usage_across_empty_choices_retry():
    profile = make_profile(name="oa", adapter="openai_compat", model="gpt-test")
    responses = [
        SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10)),
        SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="done"))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=25),
        ),
    ]

    class FakeCompletions:
        def create(self, **kwargs):
            return responses.pop(0)

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    provider = OpenAICompatProvider(profile, client=fake_client)
    result = provider.generate_text([TextPart("hello")])
    assert result.output == "done"
    assert result.usage.input_tokens == 200 and result.usage.output_tokens == 35


def test_extract_slides_and_exam_plan_report_fallback_profile(m1_env, monkeypatch):
    primary_profile = make_profile(name="claude", adapter="anthropic", model="claude-primary")
    fallback_profile = make_profile(name="gemini", adapter="openai_compat", model="gemini-fallback")

    # 1. extract_lecture 触发拒答回退时应记录备用档案
    extract_fb = FallbackProvider(
        FakeProvider([RefusalError("blocked")], primary_profile),
        FakeProvider([json.dumps(LECTURE_NOTES, ensure_ascii=False)], fallback_profile),
    )
    extracted = extract_module.extract_lecture(
        extract_fb, [Segment(0, 1000, "阿司匹林三段剂量")], [], CourseSettings(subject="药理学"), starts=[0]
    )
    assert extracted.fallback_used and extracted.used_profile.model == "gemini-fallback"

    # 2. match_lecture 触发拒答回退时应记录备用档案
    slides_fb = FallbackProvider(
        FakeProvider([RefusalError("blocked")], primary_profile),
        FakeProvider([json.dumps({"ranges": [{"file": 1, "start": 1, "end": 1}], "note": ""})], fallback_profile),
    )
    mat = Material(id=1, course_id=1, kind="slides", title="课件1", path="", source="s1.pdf", position=1, page_count=1)
    files = [slides_module.MatchFile(1, mat, [PageDigest(page_no=1, title="总论", content_md="内容", terms=[])])]
    matched = slides_module.match_lecture(slides_fb, files, [Segment(0, 1000, "总论")], CourseSettings(subject="药理学"))
    assert matched.fallback_used and matched.used_profile.model == "gemini-fallback"

    # 3. analyze_exam_plan 返回的 model 应为实际计费的备用型号
    plan_fb = FallbackProvider(
        FakeProvider([RefusalError("blocked")], primary_profile),
        FakeProvider([json.dumps(PLAN_OUTPUT, ensure_ascii=False)], fallback_profile),
    )
    monkeypatch.setattr(exam_plan_module, "create_provider", lambda prof: plan_fb)
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        analysis = exam_plan_module.analyze_exam_plan(
            m1_env.cfg, library, course, user_text="闭卷考试，简答题 4 题", report=lambda _: None
        )
    assert analysis.model == "gemini-fallback"


# ---------- 锁、子进程清理、压缩包、配置与边界 ----------


def test_locks_release_in_process_guard_on_open_error_and_retry_permission(tmp_path, monkeypatch):
    lock_path = tmp_path / "test.lock"
    calls = 0
    sleeps = 0
    real_open = Path.open

    def fail_twice(self: Path, *args, **kwargs):
        nonlocal calls
        if self == lock_path and calls < 2:
            calls += 1
            raise PermissionError("busy")
        return real_open(self, *args, **kwargs)

    def fake_sleep(seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1

    monkeypatch.setattr(Path, "open", fail_twice)
    monkeypatch.setattr(locks.time, "sleep", fake_sleep)
    with locks.hold(lock_path, wait=1.0):
        assert calls == 2 and sleeps >= 1

    # 如果 _lock_file 抛出错误，进程内 RLock 必须被释放，不能导致其他线程永久死锁
    def always_fail(path, deadline):
        raise LibraryError("模拟锁失败")

    monkeypatch.setattr(locks, "_lock_file", always_fail)
    with pytest.raises(LibraryError, match="模拟锁失败"):
        with locks.hold(lock_path, wait=0):
            pass

    monkeypatch.undo()
    acquired = threading.Event()

    def worker():
        with locks.hold(lock_path, wait=1.0):
            acquired.set()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    assert acquired.is_set()


def test_job_runner_stops_process_if_job_deleted_mid_run(example_config):
    cfg = load_config(example_config)
    db_path = cfg.data_dir / "lecture.db"
    # 先启动执行器再入队，避免 pause_interrupted 把刚入队的测试任务标成已暂停
    runner = JobRunner(cfg, db_path, command=lambda _: [sys.executable, "-c", "import time; time.sleep(30)"])
    runner.start()
    try:
        with Library(db_path) as library:
            course = library.add_course("测试课", CourseSettings())
            job = JobStore(library).add(course, "run", {"lectures": [], "force": []})
        runner.submit(job.id)
        deadline = time.monotonic() + 10
        while not runner._processes and time.monotonic() < deadline:
            time.sleep(0.05)
        assert job.id in runner._processes
        with Library(db_path) as library:
            library.delete_course(course.id)
        deadline = time.monotonic() + 10
        while job.id in runner._processes and time.monotonic() < deadline:
            time.sleep(0.05)
        assert job.id not in runner._processes
    finally:
        runner.shutdown()


def test_archive_rejects_spoofed_zip_size_and_corrupt_deflate(tmp_path, monkeypatch):
    # 1. 流式拷贝按实际字节限流，超过 MAX_TOTAL_BYTES 立即抛出 ArchiveError
    monkeypatch.setattr(archive, "MAX_TOTAL_BYTES", 1024)
    raw_zip = tmp_path / "bomb.zip"
    payload = b"A" * 5000
    with zipfile.ZipFile(raw_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.txt", payload)
    with pytest.raises(archive.ArchiveError, match="解压后太大"):
        archive._copy_limited(io.BytesIO(payload), io.BytesIO(), 0, raw_zip)

    # 2. 伪造 ZIP 头部声明大小为 1 字节的压缩包会被拒绝为 ArchiveError，且不会写出超大文件
    blob = raw_zip.read_bytes()
    packed_real = struct.pack("<I", len(payload))
    packed_fake = struct.pack("<I", 1)
    assert blob.count(packed_real) >= 2
    spoofed = tmp_path / "spoofed.zip"
    spoofed.write_bytes(blob.replace(packed_real, packed_fake))
    out1 = tmp_path / "out1"
    with pytest.raises(archive.ArchiveError):
        archive.extract_zip(spoofed, out1)
    assert sum(p.stat().st_size for p in out1.rglob("*") if p.is_file()) <= 1024

    # 3. 损坏的 deflate 流应统一转为 ArchiveError，而不是未捕获的 zlib.error
    monkeypatch.undo()
    corrupt = tmp_path / "corrupt.zip"
    data = bytearray(raw_zip.read_bytes())
    data[36:44] = b"\xff\x00\xfe\x01\xfd\x02\xfc\x03"
    corrupt.write_bytes(bytes(data))
    with pytest.raises(archive.ArchiveError):
        archive.extract_zip(corrupt, tmp_path / "out2")


def test_config_broken_toml_raises_config_error(tmp_path):
    bad = tmp_path / "config.toml"
    bad.write_text("[models\nbroken = ", encoding="utf-8")
    with pytest.raises(ConfigError, match="配置文件 .* 有误"):
        load_config(bad)


def test_pixmap_to_png_handles_extreme_aspect_ratio():
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 4000, 1), False)
    png = pixmap_to_png(pix, max_side=1600)
    assert png.startswith(b"\x89PNG")
    out = pymupdf.Pixmap(png)
    assert out.width == 1600 and out.height == 1


def test_qa_attachments_budget_and_corrupt_meta(tmp_path, monkeypatch):
    # 1. 损坏的 meta.json（非法后缀或字段类型错误）应抛出 AttachmentError
    saved = save_attachment(tmp_path, 1, "note.txt", "hello".encode())
    meta_path = saved.path.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["suffix"] = ".exe"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(AttachmentError):
        get_attachment(tmp_path, 1, saved.id)

    # 2. 图片预算耗尽时仍标记 truncated 并保留占位说明，而不是抛出无内容错误
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
    img_att = save_attachment(tmp_path, 1, "shot.png", pix.tobytes("png"))
    monkeypatch.setattr(qa_attachments, "MAX_IMAGE_PAGES", 0)
    contents = read_attachments([img_att], tmp_path / "work")
    assert len(contents) == 1 and contents[0].images == [] and contents[0].truncated is True and contents[0].text

    # 3. PPTX 在文本预算为 0 时仍应正确标记 has_text 和 truncated
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "ppt/slides/slide1.xml",
            '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
            ' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            "<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>阿司匹林药理作用</a:t></a:r></a:p>"
            "</p:txBody></p:sp></p:spTree></p:cSld></p:sld>",
        )
    pptx_att = save_attachment(tmp_path, 1, "slides.pptx", buf.getvalue())
    monkeypatch.setattr(qa_attachments, "MAX_TEXT_CHARS", 0)
    pptx_contents = read_attachments([pptx_att], tmp_path / "work2")
    assert pptx_contents[0].has_text is True and pptx_contents[0].truncated is True and pptx_contents[0].text


def test_books_safe_name_escapes_windows_reserved_names():
    for reserved in ("CON", "nul", "PRN.txt", "COM1", "lpt9.md"):
        cleaned = books.safe_name(reserved)
        assert cleaned.split(".", 1)[0].upper() not in {"CON", "PRN", "AUX", "NUL", "COM1", "LPT9"}


def test_timecode_and_exam_plan_reject_nan_and_inf(m1_env):
    for bad in ("nan", "NaN", "inf", "-inf"):
        with pytest.raises(ValueError):
            parse_timecode(bad)

    client = _client(m1_env)
    resp_nan = _post(client, f"/courses/{m1_env.course_id}/exam-plan/save", action="save",
                     **_plan_form(duration_minutes="nan"))
    resp_inf = _post(client, f"/courses/{m1_env.course_id}/exam-plan/save", action="save",
                     **_plan_form(total_points="inf"))
    assert resp_nan.status_code == 400 and resp_inf.status_code == 400


# ---------- Pipeline、原话窗口定位、预算裁剪与渲染边界 ----------


def test_delete_course_cleans_qa_attachments_and_locks(m1_env):
    att = save_attachment(m1_env.cfg.data_dir, m1_env.course_id, "q.txt", b"hello")
    course_att_dir = root_of(m1_env.cfg.data_dir) / str(m1_env.course_id)
    assert course_att_dir.is_dir() and att.path.is_file()
    mem_lock = locks.memory_lock(m1_env.cfg.data_dir, m1_env.course_id)
    with locks.hold(mem_lock):
        pass
    assert mem_lock.exists()

    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        delete_course(library, m1_env.cfg.data_dir, library.course_by_id(m1_env.course_id))
    assert not course_att_dir.exists()
    assert not mem_lock.exists()


def test_exam_plan_to_plan_localizes_within_window_not_earlier_duplicate(m1_env):
    segments = [
        Segment(0, 1000, "简答题四道每题十分"),  # 早期重复句（第 1 句）
        Segment(1000, 2000, "今天先复习一下药动学"),
        Segment(2000, 3000, "再讲药物代谢"),
        Segment(3000, 4000, "下面说期末安排"),
        Segment(4000, 5000, "简答题四道每题十分"),  # 真正引用的那一句（第 5 句）
        Segment(5000, 6000, "大家认真准备"),
    ]
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        lecture = library.lecture_by_id(m1_env.lecture_id)
    output = ExamPlanOutput.model_validate(PLAN_OUTPUT | {
        "question_types": [{
            "type": "简答题", "count": 4, "points_each": 10, "total_points": 0, "note": "",
            "source": "lecture", "quote": "简答题四道每题十分", "segment_ids": [6],  # 偏了 1 句，窗口覆盖 5..6
        }]
    })
    plan = to_plan(output, lecture, segments)
    ev = plan.question_types[0].evidence
    assert ev.verified and ev.segment_ids == [5] and ev.t == 4000


def test_memory_assemble_exact_budget_with_empty_text():
    profile = make_profile(tokens_per_char=1.0)
    # 第一段唯一条目裁掉后换成较长的 empty_text，第二段还有可裁条目：旧估算会提前跳出导致误报 BudgetError
    s1 = Section("A", [Item("xx", "", keep=1)], trim_rank=1, empty_text="（这里有很长的空占位说明文字）")
    s2 = Section("B", [Item("y" * 40, "", keep=1)], trim_rank=2, empty_text="无")
    assembled = assemble([s1, s2], budget=30, profile=profile)
    assert assembled.tokens <= 30
    assert assembled.trimmed == {"A": 1, "B": 1}


def test_report_render_escapes_table_cells_and_render_edge_cases():
    result = {
        "version": 3, "course": "药理学", "generated_at": "2026-11-24T10:00:00",
        "models": [{"profile": "gemini", "model": "g"}],
        "scope": {"lectures": [], "materials": []}, "warnings": [],
        "plan_form": "闭卷",
        "plan_table": [{"name": "A|B型题", "count": 2, "points_each": 5, "total_points": 10, "share": 1.0,
                        "note": "第一行\n含|竖线"}],
        "groups": [{"type": "A|B型题", "ranks": []}],
        "points": [], "excluded": [], "answer_rules": [],
    }
    md = render_report_markdown("考前报告", result)
    assert r"| A\|B型题 | 2 题 × 5 分 = 10 分，占 100% | 第一行 含\|竖线 |" in md

    # 正文自带私用区占位符样式的文本时不能 IndexError
    rendered = str(render_markdown("正常文字 \ue00099\ue001 和 $x^2$"))
    assert "\ue00099\ue001" in rendered and "$x^2$" in rendered

    # note_index=0 不能负索引命中最后一个知识点
    notes = LectureNotes.model_validate(LECTURE_NOTES)
    refs = [Reference(lecture_id=1, note_index=0, key=1, relation="recap", target_lecture_id=1, target_seq=1,
                      target_note_index=1, kp_name="受体", segment_ids=[1], explanation="回顾")]
    rendered_refs = render_references(refs, notes, [Segment(0, 1000, "x")])
    assert "阿司匹林的剂量分段" not in rendered_refs[0]


# ---------- 资料聚合计数、设置页原子校验与问答附件 HTTP 边界 ----------


def test_material_store_counts_settings_atomic_and_qa_http_edges(m1_env, monkeypatch):
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        store = MaterialStore(library)
        m1 = store.add(course, "slides", "课件1", "s1.pdf")
        m2 = store.add(course, "past_exam", "真题1", "e1.pdf")
        store.save_pages(m1.id, [PageDigest(page_no=1, title="p1", content_md="a", terms=[]),
                                 PageDigest(page_no=2, title="p2", content_md="b", terms=[])])
        store.replace_items(m2.id, [NewItem("1", "short", "题干", 10, "答案", [1])])
        assert store.page_counts(course) == {m1.id: 2}
        assert store.item_counts(course) == {m2.id: 1}

    # 设置页同时改 model 和非法 base_url 时，不能先半写入 model
    for env_key in ("GEMINI_MODEL", "GEMINI_BASE_URL"):
        monkeypatch.delenv(env_key, raising=False)
    client = _client(m1_env)
    _post(client, "/settings/models/gemini", model="gemini-good", base_url="https://relay.example")
    bad = _post(client, "/settings/models/gemini", model="gemini-corrupted", base_url="not-a-url")
    assert "接口地址要以 http:// 或 https:// 开头" in client.get(bad.headers["location"]).text
    assert load_config(m1_env.cfg.root / "config.toml").profile("gemini").model == "gemini-good"

    # 问答附件接口：课程不存在返回 404 JSON；下载带 nosniff；分块超大表单返回 413
    token = client.app.state.csrf_token
    missing_up = client.post("/courses/999999/qa/attachments?name=a.txt", content=b"hi",
                             headers={"X-CSRF": token})
    assert missing_up.status_code == 404 and "没有这门课" in missing_up.json()["error"]
    missing_del = client.delete("/courses/999999/qa/attachments/0123456789abcdef01234567",
                                headers={"X-CSRF": token})
    assert missing_del.status_code == 404 and "没有这门课" in missing_del.json()["error"]

    ok_up = client.post(f"/courses/{m1_env.course_id}/qa/attachments?name=a.txt", content=b"hi",
                        headers={"X-CSRF": token})
    assert ok_up.status_code == 200
    att_id = ok_up.json()["attachment"]["id"]
    dl = client.get(f"/courses/{m1_env.course_id}/qa/attachments/{att_id}")
    assert dl.status_code == 200 and dl.headers.get("x-content-type-options") == "nosniff"

    # 无 Content-Length 的分块流式请求体超过表单上限时仍返回 413
    def huge_chunks():
        for _ in range(10):
            yield b"a=" + b"x" * 1024

    too_big = client.post(f"/courses/{m1_env.course_id}/exam-plan/skip", content=huge_chunks())
    assert too_big.status_code == 413
