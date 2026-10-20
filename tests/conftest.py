"""测试公共设施：不发真实请求的假模型、示例配置。"""

from __future__ import annotations

import json
import shutil
import sqlite3
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import keyring
import pytest
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError
from pydantic import BaseModel

from lecture_analyzer.asr import models as asr_models
from lecture_analyzer.asr.base import ASRBackend, Segment
from lecture_analyzer.library import Library
from lecture_analyzer.llm import base as llm_base
from lecture_analyzer.llm.base import Effort, LLMProvider, ModelProfile, Part, Usage

REPO_ROOT = Path(__file__).resolve().parents[1]


class MemoryKeyring(KeyringBackend):
    """测试用的凭据管理器：只存在内存里，绝不读写本机真实的 Windows 凭据。"""

    priority = 1  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self.store:
            raise PasswordDeleteError("没有这个密钥")
        del self.store[(service, username)]


@pytest.fixture(autouse=True)
def isolated_appdata(tmp_path_factory, monkeypatch):
    """桌面程序记下的书库位置在 %APPDATA%：测试换成临时目录，不读写本机真实的设置。

    测试里启动的任务子进程用空的凭据管理器（PYTHON_KEYRING_BACKEND），也不碰本机真实的 Windows 凭据。
    """
    monkeypatch.setenv("APPDATA", str(tmp_path_factory.mktemp("appdata")))
    monkeypatch.delenv("LEC_CONFIG", raising=False)
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    """临时错误照常重试，但不真的等待（次数与正式运行相同）。"""
    monkeypatch.setattr(llm_base, "RETRY_DELAYS", (0.0,) * len(llm_base.RETRY_DELAYS))


@pytest.fixture(autouse=True)
def no_model_download(monkeypatch):
    """测试不下载本地语音识别模型（几个 GB）。"""

    def refuse(repo: str) -> None:
        raise AssertionError(f"测试不应下载模型 {repo}")

    monkeypatch.setattr(asr_models, "_snapshot", refuse)


_OPEN_LIBRARIES: weakref.WeakSet[Library] = weakref.WeakSet()


@pytest.fixture(autouse=True)
def track_libraries(monkeypatch):
    """记下打开的书库连接，供假模型检查：调用模型时不能有没提交的写操作（会挡住其他进程写书库）。"""
    original = Library.__init__

    def tracked(self: Library, path: Path) -> None:
        original(self, path)
        _OPEN_LIBRARIES.add(self)

    monkeypatch.setattr(Library, "__init__", tracked)


def assert_no_open_transaction() -> None:
    for library in list(_OPEN_LIBRARIES):
        try:
            busy = library.conn.in_transaction
        except sqlite3.ProgrammingError:  # 连接已关闭，或属于别的线程
            continue
        assert not busy, "调用模型时书库还有没提交的写操作"


@pytest.fixture(autouse=True)
def memory_keyring():
    previous = keyring.get_keyring()
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    yield backend
    keyring.set_keyring(previous)


class FakeProvider(LLMProvider):
    """按顺序返回预设结果的假模型；结果可以是文本，也可以是要抛出的异常。"""

    def __init__(self, responses: Sequence[str | Exception], profile: ModelProfile | None = None) -> None:
        super().__init__(profile or make_profile())
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def _complete(
        self,
        parts: Sequence[Part],
        *,
        system: str | None,
        schema: type[BaseModel] | None,
        effort: Effort | None,
        max_tokens: int,
    ) -> tuple[str, Usage]:
        assert_no_open_transaction()
        self.calls.append(
            {"parts": list(parts), "system": system, "schema": schema, "effort": effort, "max_tokens": max_tokens}
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, Usage(input_tokens=10, output_tokens=5)


class RoutingProvider(FakeProvider):
    """按请求的输出结构（schema 类名）返回预设结果的假模型；某种结构只剩一个结果时重复使用它。"""

    def __init__(self, routes: dict[str, list[str | Exception]], profile: ModelProfile | None = None) -> None:
        super().__init__([], profile)
        self.routes = routes

    def _complete(self, parts, *, system, schema, effort, max_tokens):
        assert_no_open_transaction()
        self.calls.append(
            {"parts": list(parts), "system": system, "schema": schema, "effort": effort, "max_tokens": max_tokens}
        )
        queue = self.routes[schema.__name__ if schema else "text"]
        response = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(response, Exception):
            raise response
        return response, Usage(input_tokens=10, output_tokens=5)


def integration_json(notes: dict[str, Any], chapter: str = "总论", targets: Sequence[str] = (),
                     references: Sequence[dict[str, Any]] = ()) -> str:
    """假的整合结果：本讲知识点按 targets 对应（缺省为新知识点），信号都归第 1 个知识点。"""
    points = [
        {"note_kp": i, "target": targets[i - 1] if i <= len(targets) else "new", "name": kp["name"],
         "chapter": chapter, "role": "first", "aliases": []}
        for i, kp in enumerate(notes["knowledge_points"], start=1)
    ]
    signals = [{"signal": i, "note_kp": 1} for i in range(1, len(notes["signals"]) + 1)]
    return json.dumps({"knowledge_points": points, "signals": signals, "references": list(references)},
                      ensure_ascii=False)


def make_profile(**overrides: Any) -> ModelProfile:
    values: dict[str, Any] = {"name": "fake", "adapter": "openai_compat", "model": "fake-model"}
    values.update(overrides)
    return ModelProfile(**values)


@pytest.fixture
def example_config(tmp_path: Path) -> Path:
    """把仓库里的 config.example.toml 复制成临时目录下的 config.toml。"""
    target = tmp_path / "config.toml"
    shutil.copy(REPO_ROOT / "config.example.toml", target)
    return target


# ---------- M1：完整流程用的假转写、假模型和示例数据 ----------

LECTURE_SEGMENTS = [
    Segment(0, 1000, "上节课讲了受体。"),
    Segment(1000, 2000, "这个期末一定会考，"),
    Segment(2000, 3000, "大家记住阿司匹林的三段剂量。"),
]

LECTURE_NOTES = {
    "title": "量效关系",
    "outline": ["回顾受体", "阿司匹林的三段剂量"],
    "knowledge_points": [
        {"name": "阿司匹林的剂量分段", "summary_md": "小于 $75$ mg：**抗血凝**", "slide_pages": [], "segment_ids": [3],
         "minutes": 1.5}
    ],
    "examples": [],
    "homework": [],
    "signals": [
        {"type": "exam_hint", "knowledge_point": "阿司匹林的剂量分段", "strength": 3, "quote": "这个期末一定会考",
         "segment_ids": [2], "slide_pages": []},
        {"type": "emphasis", "knowledge_point": "阿司匹林的剂量分段", "strength": 1, "quote": "编造的一句话",
         "segment_ids": [2], "slide_pages": []},
    ],
}


class FakeASR(ASRBackend):
    name = "fake-asr"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, wav: Path, hotwords: Sequence[str] = ()) -> list[Segment]:
        self.calls += 1
        return list(LECTURE_SEGMENTS)


def make_wav(path: Path, seconds: float = 3) -> Path:
    from lecture_analyzer.media import run_ffmpeg

    run_ffmpeg(["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-ar", "16000", "-ac", "1", str(path)])
    return path


@pytest.fixture
def m1_env(example_config: Path, tmp_path: Path, monkeypatch):
    """准备书库、一讲 3 秒的录音，并把流程里的转写和模型换成假的。

    env.routes 是假模型按输出结构返回的结果，测试可以改写；env.provider_calls 记录每次创建的假模型。
    """
    from lecture_analyzer.config import load_config
    from lecture_analyzer.course import CourseSettings
    from lecture_analyzer.library import Library
    from lecture_analyzer.pipeline import run as run_module

    media = make_wav(tmp_path / "药理学3.wav")
    cfg = load_config(example_config)
    asr = FakeASR()
    provider_calls: list[RoutingProvider] = []
    routes: dict[str, list[str | Exception]] = {
        "LectureNotes": [json.dumps(LECTURE_NOTES, ensure_ascii=False)],
        "IntegrationResult": [integration_json(LECTURE_NOTES, chapter="药理学总论")],
    }

    def fake_provider(profile: ModelProfile) -> RoutingProvider:
        provider = RoutingProvider(routes, profile.model_copy(update={"model": "fake-model"}))
        provider_calls.append(provider)
        return provider

    monkeypatch.setattr(run_module, "create_backend", lambda *args, **kwargs: asr)
    monkeypatch.setattr(run_module, "download_models", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_module, "create_provider", fake_provider)
    with Library(cfg.data_dir / "lecture.db") as library:
        course = library.add_course("药理学", CourseSettings(subject="药理学"))
        library.update_course(course.id, slides="none")  # 默认不提供课件；测试课件匹配时再改
        lecture = library.add_lecture(course, 3, [media])
    return SimpleNamespace(cfg=cfg, media=media, asr=asr, provider_calls=provider_calls, lecture_id=lecture.id,
                           course_id=course.id, routes=routes, tmp_path=tmp_path)
