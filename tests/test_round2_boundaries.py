"""第二轮边界回归：压缩包联接点与问答附件资源上限。"""

from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

import pytest

from lecture_analyzer import archive
from lecture_analyzer.archive import ArchiveError, expand


def _junction(link: Path, target: Path) -> None:
    if os.name != "nt":
        pytest.skip("目录联接点仅 Windows 支持")
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


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
