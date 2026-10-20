import json
import zipfile
from pathlib import Path

import pymupdf
import pytest

from lecture_analyzer import archive
from lecture_analyzer.archive import ArchiveError, expand, natural_key, safe_relative
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.documents import images_to_pdf, load_pages, read_text_any, split_text
from lecture_analyzer.library import Library, LibraryError, PageRef
from lecture_analyzer.materials import MaterialStore
from lecture_analyzer.pipeline import ingest
from lecture_analyzer.schemas import PageDigest

from .conftest import FakeProvider


def _pdf(path: Path, pages: int = 2, text: str = "受体") -> Path:
    doc = pymupdf.open()
    for number in range(1, pages + 1):
        doc.new_page().insert_text((72, 72), f"{text} {number}", fontname="china-s")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


def _png(path: Path) -> Path:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
    pix.clear_with(180)
    path.write_bytes(pix.tobytes("png"))
    return path


class _RawNameInfo(zipfile.ZipInfo):
    """按原始字节写文件名、不加 UTF-8 标记（模拟中文、韩文、日文 Windows 打的包）。"""

    def _encodeFilenameFlags(self):  # noqa: N802 —— 覆盖标准库的方法
        return self.filename.encode("cp437"), self.flag_bits


def _zip_with_raw_names(path: Path, entries: dict[bytes, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for raw, data in entries.items():
            zf.writestr(_RawNameInfo(raw.decode("cp437")), data)
    return path


@pytest.fixture
def library(tmp_path: Path):
    with Library(tmp_path / "data" / "lecture.db") as lib:
        yield lib


# ---------- 压缩包与文件名 ----------


def test_natural_key_handles_digits_and_chinese_numerals():
    names = ["第十章 抗菌药.pdf", "第二章 药动学.pdf", "第一章 绪论.pdf", "第十四章 五官.pdf", "第三章.pdf"]
    assert sorted(names, key=natural_key) == [
        "第一章 绪论.pdf", "第二章 药动学.pdf", "第三章.pdf", "第十章 抗菌药.pdf", "第十四章 五官.pdf"
    ]
    assert sorted(["ch10.pdf", "ch2.pdf", "ch1.pdf"], key=natural_key) == ["ch1.pdf", "ch2.pdf", "ch10.pdf"]


@pytest.mark.parametrize(
    ("names", "encoding"),
    [
        (["病毒学/ch1-介绍.pdf", "病毒学/ch2-分类与复制策略.pdf"], "gb18030"),
        (["1장 바이러스 개론.pdf", "중간고사 기출문제.pdf"], "cp949"),
        (["第1章 ウイルス学概論.pdf", "期末試験 過去問.pdf"], "shift_jis"),
    ],
)
def test_zip_names_in_any_east_asian_encoding(tmp_path: Path, names, encoding):
    entries = {name.encode(encoding): b"%PDF" for name in names}
    path = _zip_with_raw_names(tmp_path / "a.zip", entries)
    with zipfile.ZipFile(path) as zf:
        assert archive.zip_names(zf.infolist()) == names


def test_utf8_flagged_names_are_kept(tmp_path: Path):
    path = tmp_path / "u.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("강의/1주차.pdf", b"x")  # 非 ASCII 名称：标准库自动加 UTF-8 标记
    with zipfile.ZipFile(path) as zf:
        assert archive.zip_names(zf.infolist()) == ["강의/1주차.pdf"]


def test_safe_relative_blocks_traversal_and_reserved_names():
    assert safe_relative("../../etc/passwd") == Path("etc/passwd")
    assert safe_relative("C:\\Windows\\x.pdf") == Path("Windows/x.pdf")
    assert safe_relative("/abs/a:b?.pdf") == Path("abs/a_b_.pdf")
    assert safe_relative("dir/CON.pdf") == Path("dir/_CON.pdf")
    assert safe_relative("../..") is None


def test_zip_slip_stays_inside_target(tmp_path: Path):
    path = _zip_with_raw_names(tmp_path / "evil.zip", {b"../../outside.pdf": b"%PDF", b"ok/inside.pdf": b"%PDF"})
    target = tmp_path / "out"
    written = archive.extract_zip(path, target)
    assert sorted(written) == ["ok/inside.pdf", "outside.pdf"]
    assert not (tmp_path / "outside.pdf").exists() and (target / "outside.pdf").is_file()


def test_zip_limits_and_password(tmp_path: Path, monkeypatch):
    path = _zip_with_raw_names(tmp_path / "many.zip", {f"{i}.pdf".encode(): b"x" for i in range(3)})
    monkeypatch.setattr(archive, "MAX_FILES", 2)
    with pytest.raises(ArchiveError, match="文件太多"):
        archive.extract_zip(path, tmp_path / "out")
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ArchiveError, match="损坏"):
        archive.extract_zip(bad, tmp_path / "out2")


def test_rar_without_tool_asks_to_extract_first(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(archive, "find_tool", lambda suffix: None)
    rar = tmp_path / "PDF.rar"
    rar.write_bytes(b"Rar!")
    with pytest.raises(ArchiveError, match="请先手动解压 PDF.rar"):
        expand([rar], tmp_path / "staging")


def test_expand_folder_and_archive(tmp_path: Path):
    folder = tmp_path / "资料"
    _pdf(folder / "ch2.pdf")
    _pdf(folder / "ch10.pdf")
    (folder / "说明.hwp").write_bytes(b"x")
    (folder / "~$临时.pptx").write_bytes(b"x")
    photos = folder / "试卷照片"
    photos.mkdir()
    for name in ("2.png", "1.png"):
        _png(photos / name)
    bundle = tmp_path / "病毒学.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.write(folder / "ch2.pdf", "病毒学/ch1-介绍.pdf")
        zf.writestr("病毒学/video.mp4", b"x")

    result = expand([folder, bundle], tmp_path / "staging")
    titles = [(doc.title, len(doc.paths)) for doc in result.documents]
    assert titles == [("ch2", 1), ("ch10", 1), ("试卷照片", 2), ("ch1-介绍", 1)]
    assert [p.name for p in result.documents[2].paths] == ["1.png", "2.png"]
    assert result.documents[3].source == f"{bundle}!病毒学/ch1-介绍.pdf"
    reasons = dict(result.skipped)
    assert "HWP" in reasons[str(folder / "说明.hwp")] and "不是支持的文档格式" in reasons[f"{bundle}!病毒学/video.mp4"]


# ---------- 文档 ----------


def test_text_in_legacy_encoding_and_split(tmp_path: Path):
    note = tmp_path / "通知.txt"
    note.write_bytes("期末考试范围：第一至六章。".encode("gbk") * 30)
    assert read_text_any(note).startswith("期末考试范围")
    pages = split_text("\n\n".join(["段落" * 100] * 10), limit=500)
    assert len(pages) == 5 and all(len(page) <= 500 for page in pages)


def test_images_to_pdf_keeps_order(tmp_path: Path):
    pdf = images_to_pdf([_png(tmp_path / "a.png"), _png(tmp_path / "b.png")], tmp_path / "out.pdf")
    assert len(load_pages(pdf, tmp_path / "work")) == 2


# ---------- 资料库 ----------


def test_store_tracks_slides_choice_and_lecture_pages(library: Library, tmp_path: Path):
    course = library.add_course("病毒学", CourseSettings())
    store = MaterialStore(library)
    mock = store.add(course, "mock_exam", "模拟题（样例）", "x.pdf")
    assert library.course_by_id(course.id).slides == "unset"  # 试题不算课件
    slides = store.add(library.course_by_id(course.id), "slides", "ch1", "ch1.pdf")
    assert library.course_by_id(course.id).slides == "provided"
    store.set_file(slides.id, "materials/2/ch1.pdf", 3)
    lecture = library.add_lecture(course, 1, [tmp_path / "a.mp4"])
    library.set_step(lecture.id, "slides", "done")
    store.set_lecture_pages(lecture.id, [PageRef(slides.id, 2), PageRef(slides.id, 1)], "auto")
    assert [p.ref.page_no for p in store.lecture_pages(lecture.id)] == [1, 2]
    assert store.page_labels([PageRef(slides.id, 2), PageRef(99, 1)]) == ["ch1 第 2 页", "（已删除的资料） 第 1 页"]

    assert store.remove(slides.id) == [lecture.id]
    assert store.lecture_pages(lecture.id) == [] and library.lecture_by_id(lecture.id).step_state("slides") == ""
    assert library.course_by_id(course.id).slides == "unset"  # 课件删光了：回到还没选择
    assert [m.id for m in store.list(course)] == [mock.id]
    with pytest.raises(LibraryError, match="资料类型"):
        store.add(course, "homework", "x", "x")


# ---------- 导入与逐页识别 ----------


def _digest_json(page_nos: list[int]) -> str:
    pages = [{"page_no": n, "title": f"第{n}页标题", "content_md": f"内容{n}", "terms": [f"术语{n}", "受体"]}
             for n in page_nos]
    return json.dumps({"pages": pages}, ensure_ascii=False)


def test_import_split_and_resumable_digest(library: Library, example_config: Path, tmp_path: Path, monkeypatch):
    from lecture_analyzer.config import load_config

    cfg = load_config(example_config)
    cfg.materials.digest_batch = 2
    course = library.add_course("药理学", CourseSettings())
    lecture = library.add_lecture(course, 1, [tmp_path / "a.mp4"])
    library.set_step(lecture.id, "slides", "skipped")
    result = ingest.import_materials(library, cfg.data_dir, course, [_pdf(tmp_path / "第二章.pdf", pages=5)],
                                     "slides", report=lambda _: None)
    material = result.materials[0]
    folder = ingest.material_dir(cfg.data_dir, material.id)
    assert material.page_count == 5 and material.path == f"materials/{material.id}/第二章.pdf"
    assert (folder / "pages" / "5.png").is_file() and ingest.page_image(cfg.data_dir, material, 6) is None
    assert library.lecture_by_id(lecture.id).step_state("slides") == ""  # 导入课件后待重新匹配

    calls: list[FakeProvider] = []
    responses = {(1, 2): _digest_json([1, 2]), (3, 4): RuntimeError("中转站超时"), (5,): _digest_json([5])}

    def fake_provider(profile):
        provider = FakeProvider([], profile.model_copy(update={"model": "fake-model"}))
        provider._complete = lambda parts, **kw: _answer(parts, responses)  # type: ignore[method-assign]
        calls.append(provider)
        return provider

    monkeypatch.setattr(ingest, "create_provider", fake_provider)
    with pytest.raises(RuntimeError, match="中转站超时"):
        ingest.digest_material(cfg, library, material, report=lambda _: None)
    store = MaterialStore(library)
    assert [d.page_no for d in store.pages(material.id)] == [1, 2, 5]  # 其他批次照常保存
    assert store.get(material.id).step_state("digest") == "failed"

    responses[(3, 4)] = _digest_json([3, 4])
    calls.clear()
    outcome = ingest.digest_material(cfg, library, store.get(material.id), report=lambda _: None)
    assert outcome.pages == 2 and len(calls) == 1  # 只识别上次失败的两页
    assert store.get(material.id).step_state("digest") == "done"
    assert [row["step"] for row in library.cost_summary(course)] == ["page_digest"]


def _answer(parts, responses):
    from lecture_analyzer.llm import Usage

    texts = [getattr(p, "text", "") for p in parts]
    requested = tuple(int(t.split("第 ")[1].split(" 页")[0]) for t in texts if t.startswith("=== 第"))
    response = responses[requested]
    if isinstance(response, Exception):
        raise response
    return response, Usage(10, 5)


def test_failed_import_is_rolled_back(library: Library, tmp_path: Path):
    course = library.add_course("药理学", CourseSettings())
    broken = tmp_path / "坏文件.pdf"
    broken.write_bytes(b"not a pdf")
    with pytest.raises(Exception):
        ingest.import_materials(library, tmp_path / "data", course, [broken], "slides", report=lambda _: None)
    assert MaterialStore(library).list(course) == []
    assert not any((tmp_path / "data" / "materials").glob("*"))


def test_pin_lecture_pages(library: Library, tmp_path: Path):
    course = library.add_course("药理学", CourseSettings())
    lecture = library.add_lecture(course, 3, [tmp_path / "a.mp4"])
    result = ingest.import_materials(library, tmp_path / "data", course, [_pdf(tmp_path / "第三讲.pdf", pages=3)],
                                     "slides", report=lambda _: None)
    ingest.pin_lecture_pages(library, lecture.id, result.materials[0])
    pages = MaterialStore(library).lecture_pages(lecture.id)
    assert [(p.ref.page_no, p.source) for p in pages] == [(1, "manual"), (2, "manual"), (3, "manual")]


def test_page_digest_roundtrip_through_store(library: Library):
    course = library.add_course("药理学", CourseSettings())
    store = MaterialStore(library)
    material = store.add(course, "textbook", "教材", "book.pdf")
    store.save_pages(material.id, [PageDigest(page_no=2, title="t", content_md="c", terms=["受体"])])
    store.save_pages(material.id, [PageDigest(page_no=2, title="t2", content_md="c2", terms=[])])  # 覆盖同一页
    found = store.page_digests([PageRef(material.id, 2), PageRef(material.id, 3)])
    assert list(found) == [PageRef(material.id, 2)] and found[PageRef(material.id, 2)].title == "t2"
