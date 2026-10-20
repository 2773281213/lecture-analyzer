import sys
from pathlib import Path

import pymupdf
import pytest

from lecture_analyzer.documents import DocumentError, load_pages

PNG_SIGNATURE = b"\x89PNG"


def _png_bytes(width: int, height: int) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pix.clear_with(200)
    return pix.tobytes("png")


def _make_pdf(path: Path, with_image: bool = False) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "牛顿第二定律 F=ma", fontname="china-s")
    if with_image:
        page.insert_image(pymupdf.Rect(72, 100, 272, 200), stream=_png_bytes(200, 100))
    doc.save(path)
    doc.close()
    return path


def test_pdf_pages_have_text_and_render(tmp_path: Path):
    pages = load_pages(_make_pdf(tmp_path / "课件.pdf"), tmp_path / "work")
    assert len(pages) == 1
    page = pages[0]
    assert page.page_no == 1 and "牛顿第二定律" in page.text
    assert page.image_png is not None and page.image_png.startswith(PNG_SIGNATURE)
    assert page.has_images is False


def test_pdf_embedded_image_detected(tmp_path: Path):
    pages = load_pages(_make_pdf(tmp_path / "图.pdf", with_image=True), tmp_path / "work")
    assert pages[0].has_images is True


def test_large_photo_is_downscaled(tmp_path: Path):
    photo = tmp_path / "板书.png"
    photo.write_bytes(_png_bytes(3000, 1000))
    pages = load_pages(photo, tmp_path / "work", max_image_side=1500)
    pix = pymupdf.Pixmap(pages[0].image_png)
    assert (pix.width, pix.height) == (1500, 500)
    assert pages[0].text == "" and pages[0].has_images


def test_text_file_is_single_page(tmp_path: Path):
    note = tmp_path / "通知.md"
    note.write_text("# 期末考试\n考试范围：第一至六章", encoding="utf-8")
    pages = load_pages(note, tmp_path / "work")
    assert len(pages) == 1 and pages[0].image_png is None and "第一至六章" in pages[0].text


def test_unsupported_and_missing_files(tmp_path: Path):
    bad = tmp_path / "a.xyz"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(DocumentError, match="不支持"):
        load_pages(bad, tmp_path)
    with pytest.raises(DocumentError, match="找不到"):
        load_pages(tmp_path / "missing.pdf", tmp_path)


# ---- 以下需要本机 Office，通过 COM 自动化生成并转换真实的 PPTX / DOCX ----

office = pytest.mark.skipif(sys.platform != "win32", reason="仅 Windows 支持 Office COM")


def _make_pptx(path: Path, text: str) -> None:
    import win32com.client

    app = win32com.client.DispatchEx("PowerPoint.Application")
    presentation = app.Presentations.Add(0)  # 不显示窗口
    slide = presentation.Slides.Add(1, 2)  # ppLayoutText
    slide.Shapes(1).TextFrame.TextRange.Text = text
    presentation.SaveAs(str(path), 24)  # ppSaveAsOpenXMLPresentation
    presentation.Close()
    del slide, presentation  # 退出前先释放对象
    if app.Presentations.Count == 0:
        app.Quit()


def _make_docx(path: Path, text: str) -> None:
    import win32com.client

    app = win32com.client.DispatchEx("Word.Application")
    app.Visible = False
    document = app.Documents.Add()
    document.Content.Text = text
    document.SaveAs2(str(path), 16)  # wdFormatXMLDocument
    document.Close(0)
    del document  # 退出前先释放对象
    if app.Documents.Count == 0:
        app.Quit()


@office
@pytest.mark.office
def test_pptx_converted_via_powerpoint(tmp_path: Path):
    source = tmp_path / "第一讲.pptx"
    _make_pptx(source, "课件测试标题")
    pages = load_pages(source, tmp_path / "work")
    assert len(pages) == 1 and "课件测试标题" in pages[0].text
    assert (tmp_path / "work" / "第一讲.pdf").is_file()


@office
@pytest.mark.office
def test_docx_converted_via_word(tmp_path: Path):
    source = tmp_path / "模拟题.docx"
    _make_docx(source, "资料测试内容：求解微分方程")
    pages = load_pages(source, tmp_path / "work")
    assert "资料测试内容" in pages[0].text
