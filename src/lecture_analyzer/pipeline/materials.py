"""资料处理：逐页识别；试题、提纲、通知类资料再拆成题目或条目，并对应到课程知识点。

- 拆题按页分批调用模型（[steps] material）。跨批的题目：模型把本批开头属于上一条的部分放进 continuation_md，
  由代码接到上一条后面。题型是开放的，模型给出的类型不在列表里时记为 other。
- 对应知识点：给出课程知识点（K 编号）和题目（M 编号），每条对应 0–3 个，代码校验编号。
  对应时记下课程记忆的指纹；课程记忆变了（例如又整合了一讲）就算过期，生成报告或总复习前重做，
  过期的对应关系不参与打分和星级。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from ..config import AppConfig
from ..knowledge import CourseMemory, KnowledgePoint
from ..library import Course, Library
from ..llm import LLMError, LLMProvider, TextPart, Usage, create_provider
from ..llm.base import Effort
from ..materials import ITEM_KINDS, Material, MaterialItem, MaterialStore, NewItem
from ..prompts import load_prompt
from ..schemas import ItemBatch, ItemMappingBatch, ItemOut, PageDigest
from .ingest import digest_material
from .integrate import parse_key
from .memory import snippet

Report = Callable[[str], None]

ITEM_TYPES = {
    "choice": "选择题", "fill": "填空题", "judge": "判断题", "term": "名词解释", "short": "简答题",
    "calculation": "计算题", "essay": "论述题", "case": "病例分析", "outline": "提纲要点",
    "scope_include": "要考", "scope_exclude": "不考", "info": "说明", "other": "其他",
}
ITEMS_BATCH_CHARS = 24_000  # 每次拆题请求最多带多少字的页面内容
MAP_BATCH = 40  # 每次对应知识点的题目数
MAX_KPS = 3  # 每条最多对应几个知识点


@dataclass(frozen=True)
class MaterialHit:
    """某个知识点在资料里出现的一条题目或条目。"""

    key: int
    material: Material
    item: MaterialItem


# ---------- 处理入口 ----------


def process_materials(
    cfg: AppConfig, library: Library, course: Course, materials: Sequence[Material] | None = None,
    report: Report = print,
) -> None:
    """处理资料（不给 materials 时处理这门课里所有没处理完的）：逐页识别 → 拆题 → 对应知识点。

    一份资料失败（例如中转站断开）不影响其余资料，最后汇总报错；已识别的页都已保存，再运行会接着做。
    """
    store = MaterialStore(library)
    failures: list[str] = []
    for material in materials if materials is not None else store.list(course):
        try:
            _process_one(cfg, library, store, material, report)
        except LLMError as exc:
            report(f"「{material.title}」没处理完：{exc}")
            failures.append(material.title)
    ensure_mapped(cfg, library, course, report)
    if failures:
        raise LLMError(f"{len(failures)} 份资料没处理完（{'、'.join(failures)}）：在课程页点「继续处理没处理完的资料」"
                       f"（命令行：`lec material process {course.name}`），已识别的页不会重做")


def _process_one(cfg: AppConfig, library: Library, store: MaterialStore, material: Material, report: Report) -> None:
    if store.get(material.id).step_state("digest") != "done":
        report(f"识别「{material.title}」（{material.page_count} 页）…")
        digest_material(cfg, library, store.get(material.id), report)
    if material.kind in ITEM_KINDS:
        ensure_items(cfg, library, store.get(material.id), report)


def ensure_items(cfg: AppConfig, library: Library, material: Material, report: Report = print) -> None:
    if material.step_state("items") == "done":
        return
    store = MaterialStore(library)
    profile = cfg.profile_for_step("material")
    subject = library.course_by_id(material.course_id).settings.subject
    report(f"拆分「{material.title}」的题目…")
    try:
        items, usage = split_items(create_provider(profile), material, store.pages(material.id), subject,
                                   effort=cfg.effort_for_step("material"))
    except Exception as exc:  # 记录失败原因后照常抛出
        store.set_step(material.id, "items", "failed", f"{type(exc).__name__}: {exc}")
        raise
    store.replace_items(material.id, items)
    store.reset_steps(material.id, ["kp_map"])
    library.log_call(None, "material", profile, usage, course_id=material.course_id)
    store.set_step(material.id, "items", "done", f"{len(items)} 条（{profile.model}）")
    report(f"「{material.title}」拆出 {len(items)} 条")


def ensure_mapped(cfg: AppConfig, library: Library, course: Course, report: Report = print) -> int:
    """把题目对应到当前的课程知识点（已是最新的跳过）；还没有课程记忆时不做。返回本次重做的资料数。"""
    tree = CourseMemory(library).tree(course)
    if not tree:
        return 0
    fingerprint = memory_fingerprint(tree)
    store = MaterialStore(library)
    done = 0
    for material in store.list(course, ITEM_KINDS):
        if material.step_state("items") != "done" or mapping_fresh(material, fingerprint):
            continue
        profile = cfg.profile_for_step("material")
        items = store.items(material.id)
        report(f"把「{material.title}」的 {len(items)} 条对应到课程知识点…")
        mapping, usage = map_items(create_provider(profile), tree, items, course.settings.subject, material,
                                   effort=cfg.effort_for_step("material"))
        store.set_item_kps(material.id, mapping)
        library.log_call(None, "material", profile, usage, course_id=course.id)
        linked = sum(1 for keys in mapping.values() if keys)
        store.set_step(material.id, "kp_map", "done", f"{linked}/{len(items)} 条对应到知识点（{profile.model}）",
                       fingerprint=fingerprint)
        done += 1
    return done


def memory_fingerprint(tree: Sequence[KnowledgePoint]) -> str:
    """课程记忆的指纹：知识点编号和名称。又整合了一讲、重建或改名后都会变。"""
    text = json.dumps([[kp.key, kp.name] for kp in tree], ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def mapping_fresh(material: Material, fingerprint: str) -> bool:
    info = material.status.get("kp_map", {})
    return info.get("state") == "done" and info.get("fingerprint") == fingerprint


def stale_materials(library: Library, course: Course) -> list[Material]:
    """拆好了题、但对应关系与当前课程记忆不一致（或还没对应）的资料。"""
    fingerprint = memory_fingerprint(CourseMemory(library).tree(course))
    return [m for m in MaterialStore(library).list(course, ITEM_KINDS)
            if m.step_state("items") == "done" and not mapping_fresh(m, fingerprint)]


def material_hits(library: Library, course: Course) -> dict[int, list[MaterialHit]]:
    """各知识点在资料里出现的题目或条目：{K 编号: [...]}。只用与当前课程记忆一致的对应关系。"""
    store = MaterialStore(library)
    fingerprint = memory_fingerprint(CourseMemory(library).tree(course))
    hits: dict[int, list[MaterialHit]] = {}
    for material in store.list(course, ITEM_KINDS):
        if not mapping_fresh(material, fingerprint):
            continue
        for item in store.items(material.id):
            for key in item.kp_keys:
                hits.setdefault(key, []).append(MaterialHit(key, material, item))
    return hits


# ---------- 拆题 ----------


def page_batches(pages: Sequence[PageDigest], limit: int = ITEMS_BATCH_CHARS) -> list[list[PageDigest]]:
    batches: list[list[PageDigest]] = []
    size = 0
    for page in pages:
        if batches and size + len(page.content_md) > limit:
            batches.append([])
            size = 0
        if not batches:
            batches.append([])
        batches[-1].append(page)
        size += len(page.content_md)
    return batches


def split_items(
    provider: LLMProvider, material: Material, pages: Sequence[PageDigest], subject: str, effort: Effort | None = None
) -> tuple[list[NewItem], Usage]:
    system = load_prompt("material_items", subject=subject, kind=material.kind_label)
    items: list[NewItem] = []
    usage = Usage()
    for batch in page_batches(pages, ITEMS_BATCH_CHARS):
        parts = [TextPart("\n\n".join(f"=== 第 {page.page_no} 页 ===\n{page.content_md}" for page in batch))]
        if items:
            parts.append(TextPart(f"上一批最后一条：{items[-1].label} {snippet(items[-1].content_md, 200)}"))
        parts.append(TextPart(f"请输出第 {batch[0].page_no}–{batch[-1].page_no} 页的题目或条目。"))
        result = provider.generate_structured(parts, ItemBatch, system=system, effort=effort)
        usage = usage + result.usage
        continuation = result.output.continuation_md.strip()
        if continuation and items:
            last = items[-1]
            items[-1] = replace(last, content_md=f"{last.content_md}\n{continuation}",
                                pages=sorted({*last.pages, batch[0].page_no}))
        items += [_new_item(out, batch) for out in result.output.items if out.content_md.strip()]
    return items, usage


def _new_item(out: ItemOut, batch: Sequence[PageDigest]) -> NewItem:
    valid = {page.page_no for page in batch}
    item_type = out.item_type.strip().lower()
    return NewItem(
        label=out.label.strip(),
        item_type=item_type if item_type in ITEM_TYPES else "other",
        content_md=out.content_md.strip(),
        score=out.score if out.score > 0 else None,
        answer_md=out.answer_md.strip(),
        pages=sorted({p for p in out.pages if p in valid}) or [batch[0].page_no],
    )


# ---------- 对应知识点 ----------


def map_items(
    provider: LLMProvider, tree: Sequence[KnowledgePoint], items: Sequence[MaterialItem], subject: str,
    material: Material, effort: Effort | None = None,
) -> tuple[dict[int, list[int]], Usage]:
    """返回 {题目 item.id: [K 编号]}；模型漏掉的题目记为没有对应的知识点。"""
    system = load_prompt("material_map", subject=subject, kind=material.kind_label)
    kp_text = "\n".join(_kp_line(kp) for kp in tree)
    valid = {kp.key for kp in tree}
    mapping: dict[int, list[int]] = {item.id: [] for item in items}
    usage = Usage()
    for start in range(0, len(items), MAP_BATCH):
        batch = items[start : start + MAP_BATCH]
        lines = [_item_line(item) for item in batch]
        parts = [TextPart("## 课程知识点\n" + kp_text), TextPart("## 题目或条目\n" + "\n".join(lines)),
                 TextPart("请输出每一条对应的课程知识点。")]
        result = provider.generate_structured(parts, ItemMappingBatch, system=system, effort=effort)
        usage = usage + result.usage
        by_idx = {item.idx: item for item in batch}
        for entry in result.output.mappings:
            item = by_idx.get(entry.item)
            if item is None or mapping[item.id]:
                continue
            keys = [key for key in dict.fromkeys(parse_key(text) for text in entry.kps) if key in valid]
            mapping[item.id] = keys[:MAX_KPS]
    return mapping, usage


def _kp_line(kp: KnowledgePoint) -> str:
    extra = f"｜章节：{kp.chapter}" if kp.chapter else ""
    aliases = f"｜又称：{'、'.join(kp.aliases)}" if kp.aliases else ""
    return f"[K{kp.key}] {kp.name}{extra}{aliases}"


def _item_line(item: MaterialItem) -> str:
    label = f" {item.label}" if item.label else ""
    return f"[M{item.idx}]（{ITEM_TYPES.get(item.item_type, '其他')}{label}）{snippet(item.content_md, 300)}"
