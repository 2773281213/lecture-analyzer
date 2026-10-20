"""资料入库：展开文件 / 文件夹 / 压缩包 → 复制到 data/materials/<编号>/ → 拆页 → 逐页识别。

- 拆页在本机完成：页面图片存为 pages/<页码>.png，抽取的文字存为 pages.json；Office 文档先转成 PDF。
- 逐页识别调用模型（[steps] page_digest），按批并发，每批完成就写入书库：中断后再运行只识别还没识别的页。
- 课件类资料导入或删除后，这门课自动匹配的讲标记为"待重新匹配"，下次运行时重新匹配课件
  （匹配结果没变就不重新整理笔记）。
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..archive import FoundDocument, expand, safe_relative
from ..config import AppConfig
from ..documents import DocPage, images_to_pdf, iter_pages
from ..library import Course, Library, LibraryError, PageRef
from ..llm import Usage, create_provider
from ..materials import PAGE_KINDS, Material, MaterialStore
from ..schemas import PageDigest
from .page_digest import digest_pages

Report = Callable[[str], None]


@dataclass
class ImportResult:
    materials: list[Material] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # （位置，原因）


def material_dir(data_dir: Path, material_id: int) -> Path:
    return data_dir / "materials" / str(material_id)


def page_image(data_dir: Path, material: Material, page_no: int) -> Path | None:
    """页面图片的位置；纯文本资料或页码不存在时返回 None。"""
    if not 1 <= page_no <= material.page_count:
        return None
    path = material_dir(data_dir, material.id) / "pages" / f"{page_no}.png"
    return path if path.is_file() else None


# ---------- 导入 ----------


def import_materials(
    library: Library, data_dir: Path, course: Course, inputs: Sequence[Path], kind: str,
    title: str = "", report: Report = print,
) -> ImportResult:
    """导入文件、文件夹或压缩包：每个文档成为一份资料，复制进书库目录并拆页（不调用模型）。"""
    store = MaterialStore(library)
    staging_root = data_dir / "work" / "import"
    staging_root.mkdir(parents=True, exist_ok=True)
    result = ImportResult()
    with tempfile.TemporaryDirectory(dir=staging_root) as staging:
        expanded = expand([Path(item) for item in inputs], Path(staging))
        result.skipped = expanded.skipped
        if not expanded.documents:
            reasons = "；".join(f"{source}：{reason}" for source, reason in expanded.skipped[:5])
            raise LibraryError("没有找到可以导入的文档" + (f"（{reasons}）" if reasons else ""))
        for document in expanded.documents:
            name = title if title and len(expanded.documents) == 1 else document.title
            material = _import_one(store, data_dir, course, kind, name, document)
            report(f"已导入「{material.title}」：{material.page_count} 页")
            result.materials.append(material)
    if kind in PAGE_KINDS:
        mark_rematch(library, store, course)
    return result


def _import_one(
    store: MaterialStore, data_dir: Path, course: Course, kind: str, title: str, document: FoundDocument
) -> Material:
    material = store.add(course, kind, title, document.source)
    folder = material_dir(data_dir, material.id)
    try:
        stored = _store_file(document, folder)
        count = split_pages(stored, folder)
    except Exception:  # 导入失败时撤销这份资料，再照常抛出
        store.remove(material.id)
        shutil.rmtree(folder, ignore_errors=True)
        raise
    store.set_file(material.id, stored.relative_to(data_dir).as_posix(), count)
    store.set_step(material.id, "pages", "done", f"{count} 页")
    return store.get(material.id)


def _store_file(document: FoundDocument, folder: Path) -> Path:
    """复制进资料目录；几张图片先合成一份 PDF。"""
    folder.mkdir(parents=True, exist_ok=True)
    if document.is_images and len(document.paths) > 1:
        name = safe_relative(f"{document.title}.pdf") or Path("images.pdf")
        return images_to_pdf(document.paths, folder / name.name)
    source = document.paths[0]
    name = safe_relative(source.name) or Path(f"document{source.suffix.lower()}")
    destination = folder / name.name
    shutil.copyfile(source, destination)
    return destination


def split_pages(path: Path, folder: Path) -> int:
    """拆页：页面图片存为 pages/<页码>.png，文字存为 pages.json；返回页数。"""
    pages_dir = folder / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    texts = []
    for page in iter_pages(path, folder / "converted"):
        if page.image_png is not None:
            (pages_dir / f"{page.page_no}.png").write_bytes(page.image_png)
        texts.append({"page_no": page.page_no, "text": page.text, "has_images": page.has_images})
    (folder / "pages.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    return len(texts)


def cached_pages(folder: Path, page_nos: Sequence[int]) -> list[DocPage]:
    """读回拆页结果（只读需要的页）。"""
    wanted = set(page_nos)
    texts = json.loads((folder / "pages.json").read_text(encoding="utf-8"))
    pages = []
    for item in texts:
        if item["page_no"] in wanted:
            image = folder / "pages" / f"{item['page_no']}.png"
            png = image.read_bytes() if image.is_file() else None
            pages.append(DocPage(item["page_no"], item["text"], png, item["has_images"]))
    return pages


def mark_rematch(library: Library, store: MaterialStore, course: Course) -> list[int]:
    """课件类资料变了：这门课自动匹配（不是手动指定）的讲，下次运行时重新匹配课件。"""
    marked = []
    for lecture in library.lectures(course):
        if any(page.source == "manual" for page in store.lecture_pages(lecture.id)):
            continue
        if lecture.step_state("slides"):
            library.reset_steps(lecture.id, ["slides"])
            marked.append(lecture.id)
    return marked


def remove_material(library: Library, data_dir: Path, material_id: int) -> Material:
    store = MaterialStore(library)
    material = store.get(material_id)
    store.remove(material_id)
    shutil.rmtree(material_dir(data_dir, material_id), ignore_errors=True)
    return material


def pin_lecture_pages(library: Library, lecture_id: int, material: Material) -> None:
    """把这一讲固定对应到这份资料的全部页（`lec lecture add --slides` 用）。"""
    refs = [PageRef(material.id, page_no) for page_no in range(1, material.page_count + 1)]
    MaterialStore(library).set_lecture_pages(lecture_id, refs, "manual")
    library.reset_steps(lecture_id, ["slides"])


# ---------- 逐页识别 ----------


@dataclass
class DigestOutcome:
    pages: int = 0  # 本次识别的页数
    usage: Usage = field(default_factory=Usage)


def digest_material(cfg: AppConfig, library: Library, material: Material, report: Report = print) -> DigestOutcome:
    """识别还没识别的页（按批并发）；每批完成就写入书库。有批次失败时，其余批次照常保存，最后抛出第一个错误。"""
    store = MaterialStore(library)
    done = {digest.page_no for digest in store.pages(material.id)}
    pending = [page_no for page_no in range(1, material.page_count + 1) if page_no not in done]
    outcome = DigestOutcome()
    if not pending:
        store.set_step(material.id, "digest", "done", f"{material.page_count} 页")
        return outcome
    profile = cfg.profile_for_step("page_digest")
    folder = material_dir(cfg.data_dir, material.id)
    size = max(cfg.materials.digest_batch, 1)
    batches = [pending[i : i + size] for i in range(0, len(pending), size)]
    subject = library.course_by_id(material.course_id).settings.subject
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=max(cfg.materials.digest_workers, 1)) as pool:
        futures = [pool.submit(_digest_batch, cfg, folder, batch, subject) for batch in batches]
        for future in futures:
            digests, usage = _result_or_error(future, errors)
            if digests is None:
                continue
            store.save_pages(material.id, digests)
            library.log_call(None, "page_digest", profile, usage, course_id=material.course_id)
            outcome.pages += len(digests)
            outcome.usage = outcome.usage + usage
            report(f"「{material.title}」已识别 {len(done) + outcome.pages}/{material.page_count} 页")
    if errors:
        store.set_step(material.id, "digest", "failed", f"{type(errors[0]).__name__}: {errors[0]}")
        raise errors[0]
    store.set_step(material.id, "digest", "done", f"{material.page_count} 页（{profile.model}）")
    return outcome


def _digest_batch(cfg: AppConfig, folder: Path, batch: Sequence[int], subject: str) -> tuple[list[PageDigest], Usage]:
    """在线程里识别一批页；每批单独创建适配器，不在线程之间共用连接。"""
    provider = create_provider(cfg.profile_for_step("page_digest"))
    pages = cached_pages(folder, batch)
    return digest_pages(provider, pages, subject=subject, batch_size=len(pages),
                        effort=cfg.effort_for_step("page_digest"))


def _result_or_error(
    future: Future[tuple[list[PageDigest], Usage]], errors: list[BaseException]
) -> tuple[list[PageDigest] | None, Usage]:
    try:
        digests, usage = future.result()
    except Exception as exc:  # 记下错误，先保存其他批次的结果
        errors.append(exc)
        return None, Usage()
    return digests, usage


def digest_pending(
    cfg: AppConfig, library: Library, course: Course, kinds: Sequence[str] | None = None, report: Report = print
) -> list[Material]:
    """识别这门课里还没识别完的资料（kinds 限定类型），返回本次处理过的资料。"""
    return digest_materials(cfg, library, MaterialStore(library).list(course, kinds), report)


def digest_materials(
    cfg: AppConfig, library: Library, materials: Sequence[Material], report: Report = print
) -> list[Material]:
    """识别其中还没识别完的资料，返回本次处理过的资料。"""
    processed = []
    for material in materials:
        if material.step_state("digest") == "done":
            continue
        report(f"识别资料「{material.title}」（{material.page_count} 页）…")
        digest_material(cfg, library, material, report)
        processed.append(material)
    return processed
