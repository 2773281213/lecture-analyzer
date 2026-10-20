"""文档处理：把课件和复习资料统一拆成"页"（抽取的文字 + 页面图片），两者共用。

- PDF：直接用 PyMuPDF 处理
- PPTX/PPT、DOCX/DOC：先调用本机 Office（COM 自动化）转成 PDF
- 图片：作为单页，过大的照片按长边缩放；几张照片可以先合成一份 PDF（images_to_pdf）
- TXT/MD：纯文本，较长时按段落拆成几页；编码不限于 UTF-8（自动识别 GBK、CP949 等）
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from charset_normalizer import from_bytes

SLIDE_SUFFIXES = {".pptx", ".ppt"}
WORD_SUFFIXES = {".docx", ".doc"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
TEXT_SUFFIXES = {".txt", ".md"}
SUPPORTED_SUFFIXES = {".pdf"} | SLIDE_SUFFIXES | WORD_SUFFIXES | IMAGE_SUFFIXES | TEXT_SUFFIXES
# 常见但不能直接读取的格式：提示怎样转换
UNSUPPORTED_HINTS = {
    ".hwp": "韩文 HWP 文档请用韩文 Office（Hancom）另存为 PDF 或 DOCX",
    ".hwpx": "韩文 HWP 文档请用韩文 Office（Hancom）另存为 PDF 或 DOCX",
    ".key": "Keynote 文稿请导出为 PDF 或 PPTX",
    ".pages": "Pages 文稿请导出为 PDF 或 DOCX",
    ".xls": "表格请另存为 PDF",
    ".xlsx": "表格请另存为 PDF",
    ".wps": "WPS 文档请另存为 PDF 或 DOCX",
    ".dps": "WPS 演示请另存为 PDF 或 PPTX",
    ".caj": "CAJ 文档请用 CAJViewer 另存为 PDF",
}
TEXT_PAGE_CHARS = 3000  # 纯文本按段落拆页，每页大约这么多字

# Office COM 常量
_MSO_TRUE, _MSO_FALSE = -1, 0
_PP_SAVE_AS_PDF = 32  # ppSaveAsPDF
_WD_EXPORT_FORMAT_PDF = 17  # wdExportFormatPDF
_WD_DO_NOT_SAVE = 0  # wdDoNotSaveChanges
_WD_ALERTS_NONE = 0  # wdAlertsNone


class DocumentError(RuntimeError):
    """文档读取或转换失败。"""


@dataclass(frozen=True)
class DocPage:
    page_no: int  # 从 1 开始
    text: str  # 抽取到的文字；扫描件和图片为空
    image_png: bytes | None  # 页面渲染图；纯文本文件为 None
    has_images: bool  # 页面里是否嵌有图片（公式、图表常以图片形式出现）


def load_pages(path: Path, work_dir: Path, dpi: int = 110, max_image_side: int = 2000) -> list[DocPage]:
    """读取任意支持格式的文档，返回按页拆分的内容。work_dir 用于存放 Office 转出的 PDF。"""
    return list(iter_pages(path, work_dir, dpi, max_image_side))


def iter_pages(path: Path, work_dir: Path, dpi: int = 110, max_image_side: int = 2000) -> Iterator[DocPage]:
    """逐页读取（几百页的文档不必一次把所有页面图片放进内存）。"""
    check_supported(path)
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        for page_no, text in enumerate(split_text(read_text_any(path)), start=1):
            yield DocPage(page_no, text, None, False)
        return
    if suffix in IMAGE_SUFFIXES:
        yield DocPage(1, "", _load_image_png(path, max_image_side), True)
        return
    # 从内存打开：文件损坏时 PyMuPDF 不会一直占着文件句柄（Windows 上占着就删不掉）
    with pymupdf.open(stream=convert_to_pdf(path, work_dir).read_bytes(), filetype="pdf") as doc:
        for page_no, page in enumerate(doc, start=1):
            yield DocPage(
                page_no=page_no,
                text=page.get_text("text").strip(),
                image_png=page.get_pixmap(dpi=dpi).tobytes("png"),
                has_images=bool(page.get_images(full=False)),
            )


def check_supported(path: Path) -> None:
    if not path.is_file():
        raise DocumentError(f"找不到文件：{path}")
    suffix = path.suffix.lower()
    if suffix in UNSUPPORTED_HINTS:
        raise DocumentError(f"不支持的文件格式：{path.name}（{UNSUPPORTED_HINTS[suffix]}）")
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentError(f"不支持的文件格式：{path.suffix}（支持：{', '.join(sorted(SUPPORTED_SUFFIXES))}）")


def read_text_any(path: Path) -> str:
    """读取纯文本：先按 UTF-8，不是 UTF-8 时自动识别编码（GBK、CP949、Shift-JIS……）。"""
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        best = from_bytes(data).best()
        if best is None:
            raise DocumentError(f"无法识别文本文件的编码：{path.name}（请另存为 UTF-8）") from None
        return str(best)


def split_text(text: str, limit: int = TEXT_PAGE_CHARS) -> list[str]:
    """按段落（空行）把长文本拆成几页，每页不超过 limit 字；单个段落超长时整段成页。"""
    pages: list[str] = []
    current = ""
    for paragraph in (p.strip() for p in text.replace("\r\n", "\n").split("\n\n")):
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 2 > limit:
            pages.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        pages.append(current)
    return pages or [""]


def images_to_pdf(images: Sequence[Path], out: Path, max_side: int = 2000) -> Path:
    """几张图片（例如拍下的试卷）按顺序合成一份 PDF，每张一页。"""
    doc = pymupdf.open()
    try:
        for image in images:
            png = _load_image_png(image, max_side)
            pix = pymupdf.Pixmap(png)
            page = doc.new_page(width=pix.width, height=pix.height)
            page.insert_image(page.rect, stream=png)
        out.parent.mkdir(parents=True, exist_ok=True)
        doc.save(out)
    finally:
        doc.close()
    return out


def convert_to_pdf(src: Path, out_dir: Path) -> Path:
    """PPTX/DOCX 等借助本机 Office 转成 PDF；PDF 原样返回。"""
    suffix = src.suffix.lower()
    if suffix == ".pdf":
        return src
    if suffix not in SLIDE_SUFFIXES | WORD_SUFFIXES:
        raise DocumentError(f"不支持转换为 PDF 的格式：{src.suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = (out_dir / f"{src.stem}.pdf").resolve()
    _ensure_com_initialized()
    converter = _powerpoint_to_pdf if suffix in SLIDE_SUFFIXES else _word_to_pdf
    converter(src.resolve(), out)
    if not out.is_file():
        raise DocumentError(f"转换后没有生成 PDF：{out}")
    return out


def _ensure_com_initialized() -> None:
    """确保当前线程已初始化 COM。

    不调用 CoUninitialize：pythoncom 导入时对主线程的初始化不计入引用计数，
    配对调用一次 CoUninitialize 就会让本线程后续所有 COM 调用失败（已实测）。
    """
    import pythoncom

    pythoncom.CoInitialize()


def _start_office_app(prog_id: str, label: str):  # noqa: ANN202 —— 返回 COM 动态对象
    import pywintypes
    import win32com.client

    try:
        return win32com.client.DispatchEx(prog_id)
    except pywintypes.com_error as exc:
        raise DocumentError(f"无法启动 {label}（需要本机安装 Office）：{exc}。可以手动导出 PDF 后再导入。") from exc


def _powerpoint_to_pdf(src: Path, out: Path) -> None:
    import pywintypes

    app = _start_office_app("PowerPoint.Application", "PowerPoint")
    try:
        # Open(文件名, 只读, 无标题, 不显示窗口)
        presentation = app.Presentations.Open(str(src), _MSO_TRUE, _MSO_FALSE, _MSO_FALSE)
        try:
            presentation.SaveAs(str(out), _PP_SAVE_AS_PDF)
        finally:
            presentation.Close()
            del presentation  # 先释放文档对象再退出程序，否则释放时会对已退出的进程发起远程调用
    except pywintypes.com_error as exc:
        raise DocumentError(f"PowerPoint 转换 {src.name} 失败：{exc}。可以手动导出 PDF 后再导入。") from exc
    finally:
        # PowerPoint 是单实例程序：用户自己还开着别的演示文稿时不能退出
        if app.Presentations.Count == 0:
            app.Quit()


def _word_to_pdf(src: Path, out: Path) -> None:
    import pywintypes

    app = _start_office_app("Word.Application", "Word")
    app.Visible = False
    app.DisplayAlerts = _WD_ALERTS_NONE
    try:
        # Open(文件名, 确认转换, 只读)
        document = app.Documents.Open(str(src), False, True)
        try:
            document.ExportAsFixedFormat(str(out), _WD_EXPORT_FORMAT_PDF)
        finally:
            document.Close(_WD_DO_NOT_SAVE)
            del document  # 同上：退出 Word 前先释放文档对象
    except pywintypes.com_error as exc:
        raise DocumentError(f"Word 转换 {src.name} 失败：{exc}。可以手动导出 PDF 后再导入。") from exc
    finally:
        if app.Documents.Count == 0:
            app.Quit()


def _load_image_png(path: Path, max_side: int) -> bytes:
    pix = pymupdf.Pixmap(str(path))
    if pix.colorspace is not None and pix.colorspace.n >= 4:  # CMYK 转 RGB，PNG 不支持 CMYK
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    longest = max(pix.width, pix.height)
    if longest > max_side:
        scale = max_side / longest
        pix = pymupdf.Pixmap(pix, int(pix.width * scale), int(pix.height * scale), None)
    return pix.tobytes("png")
