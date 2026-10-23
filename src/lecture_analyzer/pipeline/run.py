"""单讲处理流程：抽音频 → 转写 → 匹配课件 → 整理笔记 → 校验原话 → 整合课程记忆 → 上架。

每一步的进度记在书库里，可以断点续跑。某一步重跑后，依赖它的步骤（STEP_DEPS）也跟着重跑；
一讲由多节组成时，各节分别抽音频、分别转写（分片缓存按节保留），再拼成整讲的统一时间轴。
开始分析前，课程必须先选择是否提供课件（课件关卡）。课件在转写之后匹配：之后再导入课件，
只需重新匹配和整理笔记，不用重新转写；重新匹配的结果与笔记所用的页相同时，也不重新整理笔记。
整理笔记前，考试方案必须已确认或已跳过（方案关卡，见 pipeline/exam_plan.py）：还没有方案时先从来源讲分析，
然后停在整理笔记之前（不算失败），等用户确认后接着处理。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from ..asr import create_backend, resolve_spec
from ..asr.base import ASRBackend, Segment
from ..asr.fallback import fill_skipped
from ..asr.models import download_models
from ..books import export_course, work_dir
from ..config import AppConfig
from ..exam_plans import ExamPlanStore, plan_block
from ..knowledge import CourseMemory
from ..library import Course, Lecture, LecturePart, Library, LibraryError, PageRef
from ..llm import ModelProfile, Usage, create_provider
from ..materials import PAGE_KINDS, MaterialStore
from ..media import extract_audio, wav_duration_ms
from ..schemas import PageDigest
from ..timecode import format_ms
from .exam_plan import ensure_exam_plan
from .extract import extract_lecture, part_starts
from .ingest import digest_materials, digest_pending
from .integrate import IntegrateContext, IntegrateOutcome, integrate_through, plan_rebuild, rebuild
from .slides import course_terms, describe_pages, match_files, match_lecture
from .validate import pass_rate, validate_signals

STEPS = ("media", "transcribe", "slides", "extract", "validate", "integrate", "book")
STEP_LABELS = {
    "media": "抽取音频",
    "transcribe": "转写",
    "slides": "匹配课件",
    "extract": "整理笔记",
    "validate": "校验原话",
    "integrate": "整合课程记忆",
    "book": "上架",
}
# 每一步依赖的上游步骤：上游在本次运行中重跑了（且结果有变化），这一步也要重跑
STEP_DEPS: dict[str, tuple[str, ...]] = {
    "media": (),
    "transcribe": ("media",),
    "slides": ("transcribe",),
    "extract": ("transcribe", "slides"),
    "validate": ("extract",),
    "integrate": ("validate",),
    "book": ("integrate",),
}
FINISHED = ("done", "skipped")
GAP_WARNING_MS = 120_000  # 转写中超过 2 分钟没有内容时提醒（可能漏转）


@dataclass(frozen=True)
class RunOptions:
    force: frozenset[str] = field(default_factory=frozenset)  # 强制重跑的步骤
    model: str | None = None  # 临时指定"整理笔记"和"整合课程记忆"使用的模型档案
    asr: str | None = None  # 临时指定转写方案，如 llm:gemini、qwen3:1.7b


class LectureRunner:
    def __init__(
        self,
        cfg: AppConfig,
        library: Library,
        lecture: Lecture,
        options: RunOptions | None = None,
        report: Callable[[str], None] = print,
    ) -> None:
        self.cfg = cfg
        self.library = library
        self.lecture_id = lecture.id
        self.course = library.course_by_id(lecture.course_id)
        self.options = options or RunOptions()
        self.report = report
        self.work_dir = work_dir(cfg.data_dir, lecture.id)
        self._executed: set[str] = set()  # 本次运行中实际执行过、结果有变化的步骤
        self._unchanged: set[str] = set()  # 执行了但结果和之前相同的步骤（不带动下游重跑）

    @property
    def lecture(self) -> Lecture:
        return self.library.lecture_by_id(self.lecture_id)

    def run(self) -> Lecture:
        check_slides_choice(self.course)
        self._executed.clear()
        self._unchanged.clear()
        for step in STEPS:
            if not self._needs(step):
                self.report(f"跳过「{STEP_LABELS[step]}」（已完成）")
                continue
            if step == "extract":  # 方案关卡：没确认时抛出 AwaitingExamPlan，这一步不记为失败
                self.course = ensure_exam_plan(self.cfg, self.library, self.course, self.report)
            self.report(f"开始「{STEP_LABELS[step]}」…")
            try:
                state, detail = getattr(self, f"_step_{step}")()
            except Exception as exc:  # 记录失败原因后照常抛出，交给调用方处理
                self.library.set_step(self.lecture_id, step, "failed", f"{type(exc).__name__}: {exc}")
                raise
            self.library.set_step(self.lecture_id, step, state, detail)
            if step not in self._unchanged:
                self._executed.add(step)
            self.report(f"完成「{STEP_LABELS[step]}」：{detail}")
        return self.lecture

    def _needs(self, step: str) -> bool:
        return (
            self.lecture.step_state(step) not in FINISHED
            or step in self.options.force
            or bool(self._executed.intersection(STEP_DEPS[step]))
        )

    # ---------- 各步骤：返回（状态，说明） ----------

    def _step_media(self) -> tuple[str, str]:
        lecture = self.lecture
        total = 0
        for part in lecture.parts:
            self._extract_part(part, only_if_changed="media" not in self.options.force)
            duration = wav_duration_ms(self._part_wav(part))
            self.library.set_part_duration(self.lecture_id, part.part, duration)
            total += duration
        self.library.update_lecture(self.lecture_id, duration_ms=total)
        parts = f"{len(lecture.parts)} 节，" if len(lecture.parts) > 1 else ""
        return "done", f"{parts}时长 {format_ms(total)}"

    def _step_slides(self) -> tuple[str, str]:
        if self.course.slides == "none":
            return self._pages_result([], "skipped", "这门课不提供课件")
        store = MaterialStore(self.library)
        current = store.lecture_pages(self.lecture_id)
        titles = {m.id: m.title for m in store.list(self.course)}
        if current and all(page.source == "manual" for page in current):
            refs = [page.ref for page in current]
            # 手动指定的页也要先识别，整理笔记时才用得上
            pinned = [store.get(material_id) for material_id in sorted({ref.material_id for ref in refs})]
            digest_materials(self.cfg, self.library, pinned, report=self.report)
            return self._pages_result(refs, "done", "手动指定：" + describe_pages(refs, titles))
        digest_pending(self.cfg, self.library, self.course, PAGE_KINDS, report=self.report)
        files = match_files(store, self.course)
        if not files:
            return self._pages_result([], "skipped", "课程资料里还没有可匹配的课件页")
        profile = self.cfg.profile_for_step("match")
        result = match_lecture(create_provider(profile), files, self.library.segments(self.lecture_id),
                               self.course.settings, effort=self.cfg.effort_for_step("match"))
        self._log("match", profile, result.usage)
        store.set_lecture_pages(self.lecture_id, result.refs, "auto")
        dropped = f"；目录过长，按术语筛掉了 {result.dropped_files} 份文件" if result.dropped_files else ""
        detail = f"{describe_pages(result.refs, titles)}（{profile.model}）{dropped}；{result.note}"
        return self._pages_result(result.refs, "done", detail)

    def _pages_result(self, refs: list[PageRef], state: str, detail: str) -> tuple[str, str]:
        """匹配到的页与现有笔记所用的页相同时，不带动"整理笔记"重跑。"""
        notes_pages = self.library.note_pages(self.lecture_id)
        if self.library.notes(self.lecture_id) is not None and sorted(refs) == sorted(notes_pages):
            self._unchanged.add("slides")
            detail += "（与现有笔记所用的页相同，不需要重新整理笔记）"
        return state, detail

    def _step_transcribe(self) -> tuple[str, str]:
        lecture = self.lecture
        hotwords = course_terms(self.library, self.course)
        asr_dir = self.work_dir / "asr"
        spec = resolve_spec(self.cfg, self.options.asr or self.course.asr, self.course.settings)
        download_models(self.cfg, spec, self.report)  # 本地方案第一次使用时先下载模型（显示进度）
        backend = create_backend(self.cfg, spec, work_dir=asr_dir, settings=self.course.settings)
        segments: list[Segment] = []
        notes: list[str] = []
        try:
            for part, offset in zip(lecture.parts, lecture.part_offsets()):
                part_segments, note = self._transcribe_part(backend, asr_dir, part, hotwords)
                segments += [replace(s, start_ms=s.start_ms + offset, end_ms=s.end_ms + offset) for s in part_segments]
                notes += [note] if note else []
        finally:
            backend.close()
        for profile, usage in getattr(backend, "usage_by_model", {}).values():  # 大模型转写按型号记录用量
            self._log("transcribe", profile, usage)
        self.library.replace_segments(self.lecture_id, segments)
        detail = "".join(f"；{note}" for note in notes) + _gap_warning(segments, lecture.duration_ms)
        return "done", f"{len(segments)} 句（{backend.name}）{detail}"

    def _step_extract(self) -> tuple[str, str]:
        segments = self.library.segments(self.lecture_id)
        if not segments:
            raise RuntimeError("转写结果为空，无法整理笔记")
        profile = self.cfg.profile_for_step("extract", self.options.model)
        refs, digests, sources = self._lecture_slides()
        result = extract_lecture(
            create_provider(profile), segments, digests, self.course.settings,
            effort=self.cfg.effort_for_step("extract"), starts=part_starts(segments, self.lecture.part_offsets()),
            instructions=self.course.instructions, sources=sources,
            exam_plan=plan_block(ExamPlanStore(self.library).active(self.course), "extract"),
        )
        notes = result.output
        previous = self.library.notes(self.lecture_id)
        self.library.save_notes(self.lecture_id, notes, profile.name, profile.model, refs)
        self._log("extract", profile, result.usage)
        title = self.lecture.title
        # 标题是空的，或者就是上次笔记生成的（没手动改过）：跟着新笔记更新，例如改了笔记语言之后
        if not title or (previous is not None and title == previous.title):
            self.library.update_lecture(self.lecture_id, title=notes.title)
        windows = f"，分 {result.windows} 段整理" if result.windows > 1 else ""
        return "done", (f"{len(notes.knowledge_points)} 个知识点、{len(notes.signals)} 个考点信号"
                        f"（{profile.model}{windows}）")

    def _step_validate(self) -> tuple[str, str]:
        notes = self.library.notes(self.lecture_id)
        if notes is None:
            raise RuntimeError("还没有笔记，无法校验原话")
        checked = validate_signals(notes.signals, self.library.segments(self.lecture_id))
        self.library.replace_signals(self.lecture_id, checked)
        passed = sum(c.valid for c in checked)
        return "done", f"原话校验通过 {passed}/{len(checked)}（{pass_rate(checked):.0%}）"

    def _step_integrate(self) -> tuple[str, str]:
        context = integrate_context(self.cfg, self.library, self.options.model, self.report)
        outcome = integrate_through(context, self.course, self.lecture)
        return "done", outcome.details.get(self.lecture_id, "") + _outcome_note(outcome, self.lecture_id,
                                                                                   self.course.name)

    def _step_book(self) -> tuple[str, str]:
        folders = export_course(self.library, self.cfg.data_dir, self.course)
        return "done", str(folders[self.lecture_id])

    # ---------- 辅助 ----------

    def _lecture_slides(self) -> tuple[list[PageRef], list[PageDigest], list[str]]:
        """本讲对应的课件页：按本讲内的顺序编号为 P1、P2…，并注明出处（"资料名 第 N 页"）。"""
        store = MaterialStore(self.library)
        refs = [page.ref for page in store.lecture_pages(self.lecture_id)]
        found = store.page_digests(refs)
        refs = [ref for ref in refs if ref in found]
        digests = [found[ref].model_copy(update={"page_no": i}) for i, ref in enumerate(refs, start=1)]
        return refs, digests, store.page_labels(refs)

    def _part_wav(self, part: LecturePart) -> Path:
        return self.work_dir / "parts" / f"part-{part.part}.wav"

    def _extract_part(self, part: LecturePart, only_if_changed: bool) -> None:
        """抽出这一节的音频；源文件没变且已有音频时跳过。"""
        wav = self._part_wav(part)
        stamp = wav.with_suffix(".source.json")
        source = _fingerprint(Path(part.media_path))
        if only_if_changed and wav.is_file() and source is not None and _read_json(stamp) == source:
            return
        extract_audio(Path(part.media_path), wav)
        stamp.write_text(json.dumps(source), encoding="utf-8")

    def _transcribe_part(
        self, backend: ASRBackend, asr_dir: Path, part: LecturePart, hotwords: Sequence[str]
    ) -> tuple[list[Segment], str]:
        wav = self._part_wav(part)
        if not wav.is_file():  # 中间文件被清理过时重新抽取
            self._extract_part(part, only_if_changed=False)
        self._refresh_part_cache(asr_dir, part, hotwords)
        prefix = f"第 {part.part} 节 " if len(self.lecture.parts) > 1 else ""
        backend.progress = lambda text: self.report(f"转写{prefix.strip()}：{text}")
        segments = backend.transcribe(wav, hotwords)
        skipped = list(getattr(backend, "skipped", []))
        if not skipped:
            return segments, ""
        # 被上游拦截的片段，用本地模型补转
        fallback_dir = asr_dir / f"part-{part.part}-fallback"
        filled, message = fill_skipped(self.cfg, self.course.settings, wav, skipped, fallback_dir)
        return sorted([*segments, *filled], key=lambda s: s.start_ms), prefix + message

    def _refresh_part_cache(self, asr_dir: Path, part: LecturePart, hotwords: Sequence[str]) -> None:
        """转写的分片缓存按节判断是否作废：这一节的源文件或热词变了才清掉（热词不看顺序：旧版本写下的顺序是随机的）。

        强制重跑转写时保留仍然有效的缓存，只重新请求没有结果的片段（例如此前被拦截、由本地模型补转的）；
        要完全重新转写，删掉 data/work/lecture-N/asr。
        """
        manifest = asr_dir / f"part-{part.part}.json"
        previous = _read_json(manifest)
        current = {"source": _fingerprint(Path(part.media_path)), "hotwords": list(hotwords)}
        if current["source"] is None and previous:  # 源文件不在了：已抽出的音频仍可用，保留缓存
            current["source"] = previous.get("source")
        if not _same_cache_key(previous, current):
            for path in asr_dir.glob(f"part-{part.part}[_-]*"):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")

    def _log(self, step: str, profile: ModelProfile, usage: Usage) -> None:
        self.library.log_call(self.lecture_id, step, profile, usage)


def process_lectures(
    cfg: AppConfig, library: Library, course: Course, lectures: Sequence[Lecture] | None, options: RunOptions,
    report: Callable[[str], None], announce: Callable[[Lecture], None] = lambda lecture: None,
) -> int:
    """依次处理讲，返回处理了几讲（命令行和后台任务共用）。

    lectures 为 None 时处理这门课所有没完成的讲；强制重跑某一步时为全部讲。announce 在开始处理每一讲前调用。
    考试方案还没确认时，处理到整理笔记前停下并抛出 AwaitingExamPlan（调用方当作"等待确认"而不是失败）。
    """
    check_slides_choice(course)  # 课件关卡：先选择是否提供课件
    processed = 0
    for lecture in lectures if lectures is not None else library.lectures(course):
        # 处理前一讲可能把后面的讲标记为课程记忆待更新，所以到这一讲时再判断要不要处理
        current = library.lecture_by_id(lecture.id)
        if lectures is None and not options.force and all(current.step_state(step) in FINISHED for step in STEPS):
            continue
        announce(current)
        LectureRunner(cfg, library, current, options, report=report).run()
        processed += 1
    return processed


def integrate_context(
    cfg: AppConfig, library: Library, model: str | None = None, report: Callable[[str], None] = lambda message: None
) -> IntegrateContext:
    profile = cfg.profile_for_step("integrate", model)
    return IntegrateContext(
        library, CourseMemory(library), create_provider(profile), effort=cfg.effort_for_step("integrate"),
        budget=cfg.memory.integrate_budget, recent_lectures=cfg.memory.recent_lectures, report=report,
    )


def rebuild_course(
    cfg: AppConfig, library: Library, course: Course, from_seq: int | None = None, model: str | None = None,
    report: Callable[[str], None] = print,
) -> IntegrateOutcome | None:
    """重建课程记忆并重新导出（命令行和后台任务共用）；已是最新时返回 None，不调用模型。"""
    planned = plan_rebuild(library, CourseMemory(library), course, from_seq)
    if not planned:
        return None
    report(f"将调用模型整合 {len(planned)} 讲：" + "、".join(f"第 {x.seq} 讲" for x in planned))
    outcome = rebuild(integrate_context(cfg, library, model, report), course, from_seq)
    if outcome is None:
        return None
    for lecture in outcome.integrated:
        report(f"第 {lecture.seq} 讲：{outcome.details[lecture.id]}")
    export_course(library, cfg.data_dir, course)
    return outcome


def check_slides_choice(course: Course) -> None:
    """课件关卡：分析开始前，课程必须先选择是否提供课件。"""
    if course.slides == "unset":
        raise LibraryError(
            f"开始分析前，请先选择「{course.name}」是否提供课件：有课件就在课程页导入（导入后会自动匹配到各讲），"
            f"没有课件就在课程页声明这门课没有课件（命令行：`lec material add {course.name} 文件或压缩包 --kind slides`"
            f" 或 `lec course set {course.name} --no-slides`）"
        )


def _same_cache_key(previous: Any, current: dict[str, Any]) -> bool:
    if not isinstance(previous, dict) or previous.get("source") != current["source"]:
        return False
    return sorted(previous.get("hotwords") or []) == sorted(current["hotwords"])


def _fingerprint(path: Path) -> dict[str, Any] | None:
    """源文件的指纹（路径、大小、修改时间）；文件不存在时为 None。"""
    try:
        stat = path.stat()
    except OSError:
        return None
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _outcome_note(outcome: IntegrateOutcome, lecture_id: int, course_name: str) -> str:
    notes = []
    others = [lecture.seq for lecture in outcome.integrated if lecture.id != lecture_id]
    if others:
        notes.append("同时补做了第 " + "、".join(map(str, others)) + " 讲的整合")
    if outcome.stale:
        seqs = "、".join(str(lecture.seq) for lecture in outcome.stale)
        notes.append(f"第 {seqs} 讲的课程记忆待更新：在课程页点「重建课程记忆」（命令行：`lec rebuild {course_name}`）")
    return "".join(f"；{note}" for note in notes)


def _gap_warning(segments: list[Segment], duration_ms: int | None) -> str:
    """找出转写里最长的空白（含开头和结尾），超过阈值时提醒可能漏转。"""
    if not segments:
        return ""
    edges = [0, *(t for s in segments for t in (s.start_ms, s.end_ms)), duration_ms or segments[-1].end_ms]
    gaps = [(edges[i + 1] - edges[i], edges[i]) for i in range(0, len(edges) - 1, 2)]
    longest, at = max(gaps)
    if longest < GAP_WARNING_MS:
        return ""
    return f"；注意：{format_ms(at)} 起有 {longest / 60000:.1f} 分钟没有转写内容"
