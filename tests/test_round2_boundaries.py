"""第二轮边界回归：压缩包联接点与问答附件资源上限。"""

from __future__ import annotations

import io
import os
import struct
import subprocess
import tracemalloc
import zipfile
from pathlib import Path

import pymupdf
import pytest

from lecture_analyzer import archive, qa_attachments
from lecture_analyzer.archive import ArchiveError, expand
from lecture_analyzer.documents import DocPage, load_pages
from lecture_analyzer.qa_attachments import AttachmentError, read_attachments, save_attachment


def _junction(link: Path, target: Path) -> None:
    if os.name != "nt":
        pytest.skip("目录联接点仅 Windows 支持")
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


def _png_bytes(width: int = 40, height: int = 30) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pix.clear_with(200)
    return pix.tobytes("png")


def _scanned_pdf(path: Path, pages: int) -> Path:
    image = _png_bytes()
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=612, height=792)
        page.insert_image(page.rect, stream=image)
    doc.save(path)
    doc.close()
    return path


def _docx_bytes(text: str) -> bytes:
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xml = (
        f'<w:document xmlns:w="{namespace}"><w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body>'
        "</w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", xml)
    return buf.getvalue()


def _pptx_bytes(text: str) -> bytes:
    presentation = "http://schemas.openxmlformats.org/presentationml/2006/main"
    drawing = "http://schemas.openxmlformats.org/drawingml/2006/main"
    xml = (
        f'<p:sld xmlns:p="{presentation}" xmlns:a="{drawing}"><p:cSld><p:spTree><p:sp><p:txBody>'
        f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ppt/slides/slide1.xml", xml)
    return buf.getvalue()


def _forged_docx_bytes(xml_bytes: bytes, fake_size: int = 10) -> bytes:
    """构造把中央目录里 word/document.xml 的未压缩大小改成 fake_size 的 .docx。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", xml_bytes)
    raw = bytearray(buf.getvalue())
    cd_pos = raw.find(b"PK\x01\x02")
    assert cd_pos != -1
    struct.pack_into("<I", raw, cd_pos + 24, fake_size)  # 中央目录记录里 uncompressed size 字段的偏移
    return bytes(raw)


def test_check_extracted_removes_junction_without_touching_external(tmp_path: Path, monkeypatch):
    staging = tmp_path / "staging"
    (staging / "d").mkdir(parents=True)
    (staging / "ok.txt").write_text("ok", encoding="utf-8")
    (staging / "d" / "f.txt").write_text("f", encoding="utf-8")
    external = tmp_path / "external"
    (external / "sub").mkdir(parents=True)
    (external / "secret.txt").write_bytes(b"secret")
    (external / "sub" / "nested.txt").write_bytes(b"nested")
    _junction(staging / "junc", external)
    _junction(staging / "d" / "junc2", external)

    gone = tmp_path / "gone"
    gone.mkdir()
    _junction(staging / "dangling", gone)
    gone.rmdir()

    monkeypatch.setattr(archive, "MAX_FILES", 2)
    archive._check_extracted(staging, tmp_path / "x.7z")

    for link in (staging / "junc", staging / "d" / "junc2", staging / "dangling"):
        assert not os.path.lexists(link)
    assert (external / "secret.txt").read_bytes() == b"secret"
    assert (external / "sub" / "nested.txt").read_bytes() == b"nested"
    assert (staging / "ok.txt").read_text(encoding="utf-8") == "ok"
    assert (staging / "d" / "f.txt").read_text(encoding="utf-8") == "f"


def test_expand_archive_junction_end_to_end(tmp_path: Path, monkeypatch):
    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.txt").write_bytes(b"secret")
    rar = tmp_path / "x.rar"
    rar.write_bytes(b"Rar!")

    def fake_extract(_archive: Path, target: Path) -> None:
        (target / "doc.txt").write_text("课程内容", encoding="utf-8")
        _junction(target / "junc", external)

    monkeypatch.setattr(archive, "extract_with_tool", fake_extract)
    result = archive.expand([rar], tmp_path / "staging")

    assert [doc.title for doc in result.documents] == ["doc"]
    assert [doc.source for doc in result.documents] == [f"{rar}!doc.txt"]
    assert (external / "secret.txt").read_bytes() == b"secret"


def test_expand_skips_junctions_in_folder(tmp_path: Path):
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "lecture.txt").write_text("讲义", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    (external / "outside.txt").write_text("外部文件", encoding="utf-8")
    _junction(folder / "junc", external)
    _junction(folder / "back", folder)

    result = archive.expand([folder], tmp_path / "staging")

    assert [doc.title for doc in result.documents] == ["lecture"]
    skipped = {Path(source).name: reason for source, reason in result.skipped}
    assert set(skipped) == {"junc", "back"}
    assert all("联接点" in reason for reason in skipped.values())


def test_extract_zip_wraps_file_dir_collision_as_archive_error(tmp_path: Path):
    for index, names in enumerate((("a", "a/b.txt"), ("a/b.txt", "a"))):
        path = tmp_path / f"collision-{index}.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for name in names:
                zf.writestr(name, b"x")
        with pytest.raises(ArchiveError, match="无法解出"):
            archive.extract_zip(path, tmp_path / f"out-{index}")


def test_scanned_pdf_attachment_marks_truncated_when_image_budget_exhausted(tmp_path: Path, monkeypatch):
    pdf = _scanned_pdf(tmp_path / "扫描件.pdf", pages=3)
    attachment = save_attachment(tmp_path, 1, "扫描件.pdf", pdf.read_bytes())

    monkeypatch.setattr(qa_attachments, "MAX_IMAGE_PAGES", 0)
    contents = read_attachments([attachment], tmp_path / "work-zero")
    assert contents[0].images == []
    assert contents[0].truncated is True
    assert contents[0].text and "图片" in contents[0].text

    monkeypatch.setattr(qa_attachments, "MAX_IMAGE_PAGES", 2)
    limited = read_attachments([attachment], tmp_path / "work-two")
    assert len(limited[0].images) == 2
    assert limited[0].truncated is True


def test_legacy_office_marks_truncated_without_office(tmp_path: Path, monkeypatch):
    attachment = save_attachment(tmp_path, 1, "旧版讲义.doc", b"legacy office bytes")
    png = _png_bytes()

    def fake_iter_pages(_path: Path, _work_dir: Path, dpi: int = 110, max_image_side: int = 2000):
        yield DocPage(1, "", png, True)

    monkeypatch.setattr(qa_attachments, "iter_pages", fake_iter_pages)
    monkeypatch.setattr(qa_attachments, "MAX_IMAGE_PAGES", 0)
    contents = read_attachments([attachment], tmp_path / "work")

    assert contents[0].truncated is True
    assert contents[0].images == []
    assert contents[0].text and "图片" in contents[0].text


def test_docx_and_pptx_reject_oversized_xml_member(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(qa_attachments, "MAX_XML_BYTES", 1024)
    oversized = "大" * 2000

    docx = save_attachment(tmp_path, 1, "大文档.docx", _docx_bytes(oversized))
    with pytest.raises(AttachmentError, match="过大"):
        read_attachments([docx], tmp_path / "work-docx")

    pptx = save_attachment(tmp_path, 1, "大幻灯片.pptx", _pptx_bytes(oversized))
    with pytest.raises(AttachmentError, match="过大"):
        read_attachments([pptx], tmp_path / "work-pptx")

    small = save_attachment(tmp_path, 1, "小文档.docx", _docx_bytes("小文档内容"))
    contents = read_attachments([small], tmp_path / "work-small")
    assert "小文档内容" in contents[0].text


def test_pdf_page_render_respects_max_image_side(tmp_path: Path):
    large_path = tmp_path / "大页面.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=3000, height=2000)
    page.insert_text((72, 72), "大页面", fontname="china-s")
    doc.save(large_path)
    doc.close()

    pages = load_pages(large_path, tmp_path / "work-large", max_image_side=800)
    pix = pymupdf.Pixmap(pages[0].image_png)
    assert max(pix.width, pix.height) <= 800

    normal_path = tmp_path / "普通页面.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "普通页面", fontname="china-s")
    doc.save(normal_path)
    doc.close()

    normal = load_pages(normal_path, tmp_path / "work-normal")
    normal_pix = pymupdf.Pixmap(normal[0].image_png)
    assert (normal_pix.width, normal_pix.height) == (935, 1210)


def test_docx_forged_file_size_uses_bounded_memory(tmp_path: Path):
    # 中央目录把 file_size 伪造成 10，实际解压出约 4MB XML：分块读取时峰值内存应远小于整段解压
    xml = b"<w:p><w:r><w:t>" + b"x" * (4 * 1024 * 1024) + b"</w:t></w:r></w:p>"
    attachment = save_attachment(tmp_path, 1, "伪造.docx", _forged_docx_bytes(xml, fake_size=10))
    tracemalloc.start()
    try:
        with pytest.raises(AttachmentError):
            read_attachments([attachment], tmp_path / "work")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024 * 1024
