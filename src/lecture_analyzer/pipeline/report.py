"""考前报告：直接给出判断后的知识点本身（M4 修订：分析结果只给结论，不给推断过程）。

- 判断在后台进行：各模型按程序整理的材料分别排名（只用编号引用证据，由代码校验）→ Borda 计数取共识；
  理由、分数、名次和证据都不写进报告。
- 报告里每个考点只有：名称、星级（与课程页、总复习同一规则）、可能题型、知识点要点（中文）。要点取自总复习：
  生成报告时先更新这些考点所在章节的复习讲义（没变的章节不重新生成），再把要点复制进报告保存，
  之后总复习更新不影响已生成的报告。
- 某个模型失败时用其余模型的结果，并在报告里注明；更新总复习失败时暂用各讲笔记原文并注明。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..books import export_course, export_report
from ..config import AppConfig
from ..library import Course, Lecture, Library, LibraryError
from ..llm import LLMError, TextPart, create_provider
from ..materials import MaterialStore
from ..prompts import load_prompt
from ..reports import ReportStore
from ..schemas import KPRanking
from .extract import instructions_block
from .integrate import parse_key
from .materials import ensure_mapped, stale_materials
from .memory import Section, assemble, input_budget, trimmed_note
from .report_input import Evidence, build_evidence, rank_sections, scope_statements, type_stats
from .review import EntryView, update_review_for
from .scoring import KPScore, Scope, Scoring, score_course

Report = Callable[[str], None]
REPORT_VERSION = 2  # 1：M3 的旧版报告（排名、理由、证据、叙述）；2：只给知识点本身
_EVIDENCE_ID = re.compile(r"^\s*([QMPENT])\s*(\d+)\s*$", re.I)  # 问答另有笔记要点 N、转写片段 T


@dataclass
class RankedEntry:
    key: int
    reason: str  # 模型的理由只在后台用（校验它确实给出了依据），不写进报告
    types: list[str]
    evidence: list[str]


@dataclass
class ModelRun:
    name: str  # 模型档案名
    model: str = ""
    ranking: list[RankedEntry] = field(default_factory=list)
    error: str = ""
    note: str = ""  # 输入被精简时的说明


@dataclass
class ReportPlan:
    course: Course
    scope: Scope
    profiles: list[str]
    lecture_seqs: list[int] = field(default_factory=list)  # 只用于报告里说明范围
    material_titles: list[str] = field(default_factory=list)

    @property
    def calls(self) -> int:
        """排名的调用次数（每个模型一次）；更新总复习的调用另算，没变的章节不调用。"""
        return len(self.profiles)


@dataclass
class ReportContext:
    """一次报告的全部材料：打分、证据、范围说明、题型统计，以及各模型的排名和共识。"""

    cfg: AppConfig
    course: Course
    scoring: Scoring
    evidence: Evidence
    statements: list[dict[str, Any]]
    stats: list[dict[str, Any]]
    runs: list[ModelRun] = field(default_factory=list)  # 成功的模型
    consensus: list[tuple[int, int]] = field(default_factory=list)  # （K 编号，共识分）

    def use_runs(self, runs: Sequence[ModelRun]) -> None:
        """记下成功的模型，算共识；没有一个模型成功时报错。"""
        self.runs = [run for run in runs if not run.error]
        if not self.runs:
            raise LLMError("所有模型都没能给出排名：" + "；".join(f"{run.name}：{run.error}" for run in runs))
        rankings = {run.name: [entry.key for entry in run.ranking] for run in self.runs}
        tiebreak = {item.kp.key: item.score for item in self.scoring.ranked}
        self.consensus = borda(rankings, self.cfg.report.top_n, tiebreak)

    def entries(self, key: int) -> list[RankedEntry]:
        return [e for run in self.runs for e in run.ranking if e.key == key]


# ---------- 入口 ----------


def generate_report(cfg: AppConfig, library: Library, plan: ReportPlan, report: Report = print) -> dict[str, Any]:
    """生成报告并返回内容（JSON）；存储和导出由调用方处理。"""
    ensure_mapped(cfg, library, plan.course, report)
    ctx = prepare_context(cfg, library, plan)
    sections = rank_sections(ctx.scoring, ctx.evidence, ctx.statements, ctx.stats)
    runs = [_rank_with(ctx, library, name, sections, report) for name in plan.profiles]
    ctx.use_runs(runs)
    warnings = [f"{run.name} 没有给出排名：{run.error}" for run in runs if run.error]
    try:
        entries = update_review_for(cfg, library, plan.course, [key for key, _ in ctx.consensus], report)
    except Exception as exc:  # 要点只是没能整理成复习讲义：照常出报告，暂用各讲笔记原文
        report(f"更新总复习失败：{type(exc).__name__}: {exc}")
        warnings.append(f"知识点要点没能整理成复习讲义，暂用各讲笔记原文：{type(exc).__name__}: {exc}")
        entries = update_review_for(cfg, library, plan.course, [key for key, _ in ctx.consensus], report,
                                    generate=False)
    stale = stale_materials(library, plan.course)
    if stale:
        warnings.append("这些资料的题目还没对应到知识点，没有参与分析：" + "、".join(m.title for m in stale))
    return assemble_report(ctx, plan, runs, entries, warnings)


def prepare_context(cfg: AppConfig, library: Library, plan: ReportPlan) -> ReportContext:
    scoring = score_course(library, plan.course, cfg.scoring, plan.scope)
    if not scoring.ranked:
        raise LibraryError("所选范围里还没有可以分析的知识点：先处理至少一讲，或导入资料并对应到知识点")
    evidence = build_evidence(library, plan.course, scoring, plan.scope)
    return ReportContext(cfg, plan.course, scoring, evidence, scope_statements(scoring, evidence),
                         type_stats(library, plan.course, plan.scope.material_ids))


# ---------- 各模型排名 ----------


def _rank_with(ctx: ReportContext, library: Library, name: str, sections: list[Section], report: Report) -> ModelRun:
    cfg = ctx.cfg
    try:
        profile = cfg.profile_for_step("report", name)
        provider = create_provider(profile)
        settings = ctx.course.settings
        system = load_prompt("report_rank", subject=settings.subject, teaching=settings.teaching_description,
                             instructions=instructions_block(ctx.course.instructions), top_n=str(cfg.report.top_n))
        ask = f"请输出最可能考的 {cfg.report.top_n} 个知识点。"
        assembled = assemble(sections, input_budget(profile, cfg.report.budget), profile,
                             fixed_chars=len(system) + len(ask))
        report(f"{name}（{profile.model}）正在排名…")
        result = provider.generate_structured([TextPart(assembled.text), TextPart(ask)], KPRanking, system=system,
                                              effort=cfg.effort_for_step("report"))
    except Exception as exc:  # 多模型容错：一个模型失败不影响其他模型，失败原因写进报告
        report(f"{name} 排名失败：{type(exc).__name__}: {exc}")
        return ModelRun(name, error=f"{type(exc).__name__}: {exc}")
    library.log_call(None, "report", profile, result.usage, course_id=ctx.course.id)
    valid = {item.kp.key for item in ctx.scoring.ranked}
    ranking = resolve_ranking(result.output, valid, set(ctx.evidence.views), cfg.report.top_n)
    return ModelRun(name, profile.model, ranking, note=trimmed_note(assembled.trimmed))


def resolve_ranking(output: KPRanking, valid_keys: set[int], valid_ids: set[str], top_n: int) -> list[RankedEntry]:
    """只保留有效且不重复的知识点（老师说不考的不在 valid_keys 里），证据编号必须存在，最多 top_n 个。"""
    entries: list[RankedEntry] = []
    for item in output.ranking:
        key = parse_key(item.kp)
        if key not in valid_keys or any(e.key == key for e in entries):
            continue
        ids = [eid for eid in dict.fromkeys(normalize_id(text) for text in item.evidence) if eid in valid_ids]
        types = [t.strip() for t in item.question_types if t.strip()]
        entries.append(RankedEntry(key, item.reason.strip(), types, ids))
        if len(entries) >= top_n:
            break
    return entries


def normalize_id(text: str) -> str:
    match = _EVIDENCE_ID.match(text or "")
    return f"{match.group(1).upper()}{int(match.group(2))}" if match else ""


def borda(rankings: dict[str, list[int]], top_n: int, tiebreak: dict[int, float]) -> list[tuple[int, int]]:
    """Borda 计数：每个模型的第 1 名得 top_n 分，第 2 名得 top_n − 1 分……同分按程序参考分、再按编号。"""
    points: dict[int, int] = {}
    for keys in rankings.values():
        for position, key in enumerate(keys[:top_n]):
            points[key] = points.get(key, 0) + top_n - position
    return sorted(points.items(), key=lambda kv: (-kv[1], -tiebreak.get(kv[0], 0), kv[0]))


# ---------- 拼装 ----------


def assemble_report(ctx: ReportContext, plan: ReportPlan, runs: Sequence[ModelRun], entries: dict[int, EntryView],
                    warnings: Sequence[str]) -> dict[str, Any]:
    by_key = ctx.scoring.by_key()
    return {
        "version": REPORT_VERSION,
        "course": ctx.course.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": [{"profile": run.name, "model": run.model, "error": run.error, "note": run.note} for run in runs],
        "scope": {"lectures": plan.lecture_seqs, "materials": plan.material_titles},
        "points": [_point(ctx, by_key[key], position, entries[key])
                   for position, (key, _) in enumerate(ctx.consensus, start=1) if key in entries],
        "excluded": [{"key": item.kp.key, "name": item.kp.name} for item in ctx.scoring.excluded],
        "warnings": list(warnings),
    }


def _point(ctx: ReportContext, item: KPScore, position: int, entry: EntryView) -> dict[str, Any]:
    """报告里的一个考点：只有结论（名称、星级、可能题型、要点），不带理由和证据。"""
    types = Counter(t for e in ctx.entries(item.kp.key) for t in e.types)
    return {"rank": position, "key": item.kp.key, "name": item.kp.name, "chapter": item.kp.chapter,
            "priority": entry.priority, "types": [t for t, _ in types.most_common()],
            "summary_md": entry.summary_md, "lectures": item.lectures}


# ---------- 保存 ----------


def save_report(library: Library, data_dir: Path, plan: ReportPlan, result: dict[str, Any]) -> tuple[int, Path]:
    """存入书库并导出 Markdown，返回（报告编号，导出文件）。"""
    store = ReportStore(library)
    title = f"考前报告 {datetime.now():%Y-%m-%d %H:%M}"
    params = {"lectures": plan.lecture_seqs, "materials": sorted(plan.scope.material_ids or []),
              "models": plan.profiles}
    report_id = store.add(plan.course, title, params, result, plan.profiles)
    path = export_report(data_dir, plan.course, store.get(report_id))
    export_course(library, data_dir, plan.course)  # book.json 里列出新报告
    return report_id, path


# ---------- 范围 ----------


def parse_numbers(text: str | None, what: str) -> list[int]:
    """"1,2，5" → [1, 2, 5]。"""
    numbers = []
    for part in re.split(r"[,，、\s]+", (text or "").strip()):
        if not part:
            continue
        if not part.isdigit():
            raise LibraryError(f"{what}要写成用逗号分隔的数字，如 1,2,5")
        numbers.append(int(part))
    return numbers


def build_plan(
    cfg: AppConfig, library: Library, course: Course, lectures: Sequence[int] = (), tag: str = "",
    materials: Sequence[int] = (), profiles: Sequence[str] = (),
) -> ReportPlan:
    """按命令行或网页的选择确定报告范围和模型；不选表示全部讲、全部资料、[steps] report 的模型。"""
    chosen = _chosen_lectures(library, course, lectures, tag)
    by_id = {m.id: m for m in MaterialStore(library).list(course)}
    unknown = [n for n in materials if n not in by_id]
    if unknown:
        raise LibraryError(f"「{course.name}」没有这些编号的资料：{'、'.join(map(str, unknown))}")
    names = list(dict.fromkeys(profiles)) or [cfg.steps.get("report", "claude")]
    if len(names) > cfg.report.max_models:
        raise LibraryError(f"一次报告最多用 {cfg.report.max_models} 个模型")
    for name in names:
        cfg.profile_for_step("report", name)  # 档案不存在时报错
    scope = Scope(frozenset(lecture.id for lecture in chosen) if chosen is not None else None,
                  frozenset(materials) if materials else None)
    return ReportPlan(course, scope, names, [lecture.seq for lecture in chosen] if chosen is not None else [],
                      [by_id[n].title for n in materials])


def _chosen_lectures(library: Library, course: Course, lectures: Sequence[int], tag: str) -> list[Lecture] | None:
    """按讲次和标签筛选；都没给时返回 None（全部讲）。"""
    if not lectures and not tag:
        return None
    chosen = library.lectures(course)
    if lectures:
        missing = sorted(set(lectures) - {lecture.seq for lecture in chosen})
        if missing:
            raise LibraryError(f"「{course.name}」没有第 {'、'.join(map(str, missing))} 讲")
        chosen = [lecture for lecture in chosen if lecture.seq in set(lectures)]
    if tag:
        tags = library.course_tags(course)
        chosen = [lecture for lecture in chosen if tag.strip() in tags.get(lecture.id, [])]
        if not chosen:
            raise LibraryError(f"没有带标签「{tag.strip()}」的讲")
    return chosen
