"""问答附件：暂存学生上传的文件，取出时还原成模型能读的文字或页面图片。

附件只属于一次提问，不进入课程资料库：

- 上传后先放在 data/qa_attachments/<课程编号>/<随机编号>/，提问提交前可以删掉；
- 提问提交后标记为 used，删除这段对话时一并删除；没提交的上传两天后自动清理；
- 文字类直接读取；PDF 按页取文字，扫描页渲染成图片；DOCX/PPTX 直接读 XML，
  不需要本机 Office；旧版 DOC/PPT 仍借助 Office 转成 PDF。

网页只按随机编号取文件，不接受任意路径。
"""

from __future__ import annotations

import json
import mimetypes
import re
import secrets
import shutil
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

import pymupdf

from .documents import iter_pages, pixmap_to_png, read_text_any, render_page_png
from .library import LibraryError

Report = Callable[[str], None]

# 一次提问最多几个附件、每个多大、合计多大。都在本机处理，限制主要防止误拖整个文件夹或超大视频。
MAX_FILES = 6
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
UNUSED_KEEP_HOURS = 48

SUPPORTED_SUFFIXES = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".txt", ".md",
    ".png", ".jpg", ".jpeg", ".bmp",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
TEXT_SUFFIXES = {".txt", ".md"}

MAX_PAGES = 12
MAX_TEXT_CHARS = 20_000  # 所有附件合计展开的文字上限（留出课程证据的预算）
MAX_IMAGE_PAGES = 8  # 所有附件合计送给模型的页面图片上限
MAX_IMAGE_SIDE = 1600
MAX_XML_BYTES = 20 * 1024 * 1024  # Office 压缩包里的单个 XML 解压后的上限
TOKEN = re.compile(r"^[0-9a-f]{24}$")
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


class AttachmentError(LibraryError):
    """附件不能上传、读取或已不存在。"""


@dataclass(frozen=True)
class StoredAttachment:
    id: str
    course_id: int
    name: str
    size: int
    mime: str
    suffix: str
    path: Path
    created_at: str
    used: bool = False

    @property
    def size_text(self) -> str:
        return human_size(self.size)

    @property
    def is_image(self) -> bool:
        return self.suffix in IMAGE_SUFFIXES

    def summary(self) -> dict[str, object]:
        """存进任务参数和问答结果的信息；文件本身仍按 id 从附件目录读取。"""
        return {"id": self.id, "name": self.name, "size": self.size, "size_text": self.size_text,
                "mime": self.mime, "is_image": self.is_image}


@dataclass
class AttachmentContent:
    attachment: StoredAttachment
    text: str
    page_count: int
    images: list[tuple[int, bytes]] = field(default_factory=list)
    has_text: bool = False
    truncated: bool = False


def human_size(size: int) -> str:
    value = float(max(size, 0))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def root_of(data_dir: Path) -> Path:
    return data_dir / "qa_attachments"


def save_attachment(data_dir: Path, course_id: int, raw_name: str, data: bytes) -> StoredAttachment:
    """保存一个上传的文件；调用方已经从请求里读出了全部字节。"""
    name = _clean_name(raw_name)
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise AttachmentError(_unsupported_message(suffix))
    if not data:
        raise AttachmentError(f"附件「{name}」是空文件")
    if len(data) > MAX_FILE_BYTES:
        raise AttachmentError(f"附件「{name}」超过 {human_size(MAX_FILE_BYTES)}，请压缩或拆分后再上传")

    cleanup_unused(data_dir)
    token = secrets.token_hex(12)
    folder = root_of(data_dir) / str(course_id) / token
    folder.mkdir(parents=True, exist_ok=False)
    content = folder / f"content{suffix}"
    content.write_bytes(data)
    meta = {
        "id": token, "course_id": course_id, "name": name, "size": len(data),
        "mime": mimetypes.guess_type(name)[0] or "application/octet-stream",
        "suffix": suffix, "created_at": datetime.now().isoformat(timespec="seconds"), "used": False,
    }
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return _stored(meta, content)


def get_attachment(data_dir: Path, course_id: int, token: str) -> StoredAttachment:
    """按随机编号取附件；编号不属于这门课时当作不存在。"""
    token = str(token).strip()
    if not TOKEN.fullmatch(token):
        raise AttachmentError("附件编号不正确")
    folder = root_of(data_dir) / str(course_id) / token
    meta_path = folder / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict) or int(meta.get("course_id", 0)) != course_id:
            raise ValueError("课程不匹配")
        suffix = str(meta.get("suffix", "")).lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError("后缀不受支持")
        content = folder / f"content{suffix}"
        if not content.is_file():
            raise FileNotFoundError(content)
        return _stored(meta, content)
    except (OSError, ValueError, TypeError):
        raise AttachmentError("附件不存在或已经被清理，请重新添加") from None


def resolve_attachments(data_dir: Path, course_id: int, tokens: Sequence[str]) -> list[StoredAttachment]:
    """校验一次提问里提交的附件编号，去重并检查数量和总大小。"""
    unique = list(dict.fromkeys(str(token).strip() for token in tokens if str(token).strip()))
    if len(unique) > MAX_FILES:
        raise AttachmentError(f"一次最多添加 {MAX_FILES} 个附件")
    attachments = [get_attachment(data_dir, course_id, token) for token in unique]
    total = sum(item.size for item in attachments)
    if total > MAX_TOTAL_BYTES:
        raise AttachmentError(f"附件合计超过 {human_size(MAX_TOTAL_BYTES)}，请减少附件或拆分后再提问")
    return attachments


def load_job_attachments(data_dir: Path, course_id: int, saved: object) -> list[StoredAttachment]:
    """从任务参数里的 [{"id": ...}] 重新取出附件；排队期间被清理时给出明确说明。"""
    values = saved if isinstance(saved, list) else []
    tokens = [str(item.get("id", "")) for item in values if isinstance(item, Mapping)]
    try:
        attachments = resolve_attachments(data_dir, course_id, tokens)
    except AttachmentError as exc:
        raise LibraryError(f"这次提问的附件已经不可用：{exc}") from exc
    if len(attachments) != len(tokens):
        raise LibraryError("这次提问的附件已经不可用，请重新提问并添加附件")
    return attachments


def mark_used(data_dir: Path, attachment: StoredAttachment) -> None:
    _write_meta(data_dir, attachment, used=True)


def discard_attachment(data_dir: Path, course_id: int, token: str, *, only_unused: bool = False) -> bool:
    """删除附件目录；only_unused 为真时已随提问提交的附件不删。"""
    try:
        attachment = get_attachment(data_dir, course_id, token)
    except AttachmentError:
        return False
    if only_unused and attachment.used:
        return False
    shutil.rmtree(attachment.path.parent, ignore_errors=True)
    return True


def discard_saved(data_dir: Path, course_id: int, saved: object) -> None:
    """删除问答结果里记录的附件（删除一段对话时用）。"""
    values = saved if isinstance(saved, list) else []
    for item in values:
        if isinstance(item, Mapping):
            discard_attachment(data_dir, course_id, str(item.get("id", "")))


def cleanup_unused(data_dir: Path, now: datetime | None = None) -> int:
    """清理超过两天没有提交提问的上传；每次上传时顺手清理，不阻塞页面。"""
    cutoff = (now or datetime.now()) - timedelta(hours=UNUSED_KEEP_HOURS)
    count = 0
    root = root_of(data_dir)
    if not root.is_dir():
        return 0
    for meta_path in root.glob("*/*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(str(meta.get("created_at", "")))
            used = bool(meta.get("used"))
            folder = meta_path.parent
        except (OSError, ValueError, TypeError):
            continue
        if not used and created < cutoff:
            shutil.rmtree(folder, ignore_errors=True)
            count += 1
    return count


def read_attachments(attachments: Sequence[StoredAttachment], work_root: Path,
                     report: Report = lambda _message: None) -> list[AttachmentContent]:
    """把附件展开成文字和页面图片；文字总量和图片数量都有上限。"""
    work_root.mkdir(parents=True, exist_ok=True)
    remaining_text = MAX_TEXT_CHARS
    remaining_images = MAX_IMAGE_PAGES
    contents: list[AttachmentContent] = []
    for item in attachments:
        report(f"正在读取附件「{item.name}」…")
        work = work_root / f"read-{item.id}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            content = _read_one(item, work, remaining_text, remaining_images)
        except AttachmentError:
            raise
        except Exception as exc:  # noqa: BLE001 —— 损坏的 Office/PDF 文件会抛出各家库自己的异常
            raise AttachmentError(f"附件「{item.name}」读取失败：{exc}") from exc
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if not content.text and not content.images:
            if content.has_text:  # 文字额度被前面的附件用完，至少告诉模型这份附件存在但没展开
                content.text = "（附件文字较长，前面附件已用完展开额度，这里没有展开）"
                content.truncated = True
            elif content.truncated:  # 图片额度被前面的附件用完，同样告知模型附件存在
                content.text = "（前面的附件已用完图片展开额度，这里没有展开）"
            else:
                raise AttachmentError(f"附件「{item.name}」里没有可读取的内容")
        contents.append(content)
        remaining_text = max(remaining_text - len(content.text), 0)
        remaining_images = max(remaining_images - len(content.images), 0)
    return contents


def _read_one(item: StoredAttachment, work: Path, text_budget: int, image_budget: int) -> AttachmentContent:
    suffix = item.suffix
    if suffix in TEXT_SUFFIXES:
        text = read_text_any(item.path).strip()
        fitted, cut = _fit_text(text, text_budget)
        return AttachmentContent(item, fitted, 1, has_text=bool(text), truncated=cut)
    if suffix in IMAGE_SUFFIXES:
        png = _image_file_png(item.path)
        images = [(1, png)] if image_budget > 0 else []
        return AttachmentContent(item, "", 1, images=images, has_text=False, truncated=image_budget <= 0)
    if suffix == ".pdf":
        return _read_pdf(item, text_budget, image_budget)
    if suffix == ".docx":
        return _read_docx(item, text_budget)
    if suffix == ".pptx":
        return _read_pptx(item, text_budget)
    if suffix in {".doc", ".ppt"}:
        return _read_legacy_office(item, work, text_budget, image_budget)
    raise AttachmentError(_unsupported_message(suffix))


def _read_pdf(item: StoredAttachment, text_budget: int, image_budget: int) -> AttachmentContent:
    lines: list[str] = []
    images: list[tuple[int, bytes]] = []
    remaining = max(text_budget, 0)
    has_text = False
    truncated = False
    # 从内存打开：文件损坏时 PyMuPDF 不会一直占着文件句柄（Windows 上占着就删不掉）
    with pymupdf.open(stream=item.path.read_bytes(), filetype="pdf") as doc:
        total = len(doc)
        for page_no, page in enumerate(doc, start=1):
            if page_no > MAX_PAGES:
                truncated = True
                break
            text = page.get_text("text").strip()
            if text:
                has_text = True
                chunk = f"【第 {page_no} 页】\n{text}" if total > 1 else text
                fitted, cut = _fit_text(chunk, remaining)
                if fitted:
                    lines.append(fitted)
                    remaining = max(remaining - len(fitted), 0)
                truncated = truncated or cut
            # 扫描页没有文字；文字很少或页面里有图片时也把页面渲染图一起给模型，公式、表格不会丢
            want_image = not text or len(text) < 160 or (page.get_images(full=False) and len(text) < 600)
            if want_image:
                if len(images) < image_budget:
                    images.append((page_no, _page_png(page)))
                else:
                    truncated = True
        page_count = max(total, 1)
    if not images and not has_text and total == 0:
        raise AttachmentError(f"PDF「{item.name}」没有页面")
    return AttachmentContent(item, "\n\n".join(lines), page_count, images, has_text, truncated)


def _read_docx(item: StoredAttachment, text_budget: int) -> AttachmentContent:
    with zipfile.ZipFile(item.path) as archive:
        xml = _read_xml_member(archive, "word/document.xml", item.name)
    root = ElementTree.fromstring(xml)
    paragraphs = []
    for paragraph in root.iter(f"{_W_NS}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{_W_NS}t")).strip()
        if text:
            paragraphs.append(text)
    text = "\n\n".join(paragraphs)
    fitted, cut = _fit_text(text, text_budget)
    return AttachmentContent(item, fitted, 1, has_text=bool(text), truncated=cut)


def _read_pptx(item: StoredAttachment, text_budget: int) -> AttachmentContent:
    with zipfile.ZipFile(item.path) as archive:
        names = [name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)]
        names.sort(key=lambda name: int(re.search(r"(\d+)", Path(name).stem).group(1)))  # type: ignore[union-attr]
        lines: list[str] = []
        remaining = max(text_budget, 0)
        has_text = False
        truncated = len(names) > MAX_PAGES
        for page_no, name in enumerate(names[:MAX_PAGES], start=1):
            root = ElementTree.fromstring(_read_xml_member(archive, name, item.name))
            text = "\n".join(node.text or "" for node in root.iter(f"{_A_NS}t")).strip()
            if not text:
                continue
            has_text = True
            fitted, cut = _fit_text(f"【第 {page_no} 页】\n{text}", remaining)
            if fitted:
                lines.append(fitted)
                remaining = max(remaining - len(fitted), 0)
            truncated = truncated or cut
    text = "\n\n".join(lines)
    return AttachmentContent(item, text, max(len(names), 1), has_text=has_text, truncated=truncated)


def _read_legacy_office(item: StoredAttachment, work: Path, text_budget: int,
                        image_budget: int) -> AttachmentContent:
    lines: list[str] = []
    images: list[tuple[int, bytes]] = []
    remaining = max(text_budget, 0)
    has_text = False
    truncated = False
    page_count = 0
    for page in iter_pages(item.path, work, max_image_side=MAX_IMAGE_SIDE):
        page_count = page.page_no
        if page.page_no > MAX_PAGES:
            truncated = True
            break
        text = page.text.strip()
        if text:
            has_text = True
            fitted, cut = _fit_text(f"【第 {page.page_no} 页】\n{text}", remaining)
            if fitted:
                lines.append(fitted)
                remaining = max(remaining - len(fitted), 0)
            truncated = truncated or cut
        if page.image_png and (not text or len(text) < 160):
            if len(images) < image_budget:
                images.append((page.page_no, page.image_png))
            else:
                truncated = True
    return AttachmentContent(item, "\n\n".join(lines), max(page_count, 1), images, has_text, truncated)


def _read_xml_member(archive: zipfile.ZipFile, name: str, attachment_name: str) -> bytes:
    """读 Office 压缩包里的一个 XML；先按中央目录声明的解压大小拦截，避免高压缩比的 XML 撑爆内存。"""
    info = archive.getinfo(name)
    if info.file_size > MAX_XML_BYTES:  # 在函数体内读模块常量，测试可以 monkeypatch
        raise AttachmentError(
            f"附件「{attachment_name}」内容过大（解压后超过 {human_size(MAX_XML_BYTES)}），请另存为 PDF 后再上传"
        )
    return archive.read(info)  # 读出的字节不会超过声明的 file_size；伪造小值会因 CRC 不符抛 BadZipFile


def _clean_name(raw_name: str) -> str:
    name = Path(str(raw_name).replace("\\", "/")).name.strip()
    name = "".join(ch for ch in name if ch >= " " and ch not in '<>:"/\\|?*')
    if not name:
        raise AttachmentError("附件没有文件名")
    if len(name) > 160:
        stem, suffix = Path(name).stem[:130], Path(name).suffix
        name = f"{stem}{suffix}"
    return name


def _fit_text(text: str, budget: int) -> tuple[str, bool]:
    if budget <= 0:
        return "", bool(text)
    if len(text) <= budget:
        return text, False
    return text[: max(budget - 1, 0)].rstrip() + "…", True


def _page_png(page: pymupdf.Page) -> bytes:
    return render_page_png(page, 110, MAX_IMAGE_SIDE)


def _image_file_png(path: Path) -> bytes:
    return pixmap_to_png(pymupdf.Pixmap(str(path)), MAX_IMAGE_SIDE)


def _stored(meta: Mapping[str, object], content: Path) -> StoredAttachment:
    return StoredAttachment(
        id=str(meta.get("id", "")), course_id=int(meta.get("course_id", 0)), name=str(meta.get("name", "")),
        size=int(meta.get("size", 0)), mime=str(meta.get("mime", "application/octet-stream")),
        suffix=str(meta.get("suffix", content.suffix)), path=content,
        created_at=str(meta.get("created_at", "")), used=bool(meta.get("used")),
    )


def _write_meta(data_dir: Path, attachment: StoredAttachment, *, used: bool) -> None:
    meta_path = attachment.path.parent / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    meta["used"] = used
    if used:
        meta["used_at"] = datetime.now().isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def _unsupported_message(suffix: str) -> str:
    shown = suffix or "（没有扩展名）"
    return (f"不支持这种附件：{shown}。可以添加 PDF、Word、PPT、TXT、Markdown 或图片"
            "（PNG、JPG、BMP）；其他格式请先另存为 PDF。")
