"""把导入的文件、文件夹和压缩包展开成一份份文档（课程资料导入用）。

- zip 用标准库解压。文件名有 UTF-8 标记（或 Unicode 扩展字段）时直接用；否则自动识别编码
  （GBK、CP949、Shift-JIS、Big5……），中文、韩文、日文系统打的包都能显示正确的文件名。
- rar、7z 调用 7-Zip（安装版随附在 tools\7zip，也找本机装的）或 WinRAR（UnRAR.exe）；都没有时提示先手动解压。
- 同一文件夹里的图片（例如拍下的几页试卷）合成一份资料；单独给出的几张图片也合成一份。
- 安全：压缩包里的路径一律去掉盘符和 ".."，只保留在暂存目录之内；限制文件数和解压后的总大小；
  不跟随符号链接；有密码的压缩包直接报错，不会等待输入密码。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from charset_normalizer import from_bytes

from .documents import IMAGE_SUFFIXES, SUPPORTED_SUFFIXES, UNSUPPORTED_HINTS

ARCHIVE_SUFFIXES = {".zip", ".rar", ".7z"}
MAX_FILES = 2000  # 一个压缩包最多解出的文件数
MAX_TOTAL_BYTES = 4 * 1024**3  # 解压后的总大小上限
MAX_DEPTH = 2  # 压缩包里再套压缩包，最多展开两层
NAME_ENCODINGS = ["gb18030", "cp949", "shift_jis", "big5"]  # zip 文件名没标 UTF-8 时比较的编码
_IGNORED_NAMES = {"thumbs.db", "desktop.ini", ".ds_store", "__macosx"}
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_TOOL_TIMEOUT = 600
_TOOL_OK = (0, 1)  # 7-Zip 和 UnRAR：0 成功，1 有警告但已解出
BUNDLED_7ZIP = Path(sys.prefix).parent / "tools" / "7zip" / "7z.exe"  # 安装版：runtime 旁边的 tools\7zip
_ZIP_UTF8_FLAG = 0x800
_ZIP_ENCRYPTED_FLAG = 0x1


class ArchiveError(RuntimeError):
    """压缩包无法展开（格式损坏、有密码、缺少解压工具、超出大小限制等）。"""


@dataclass(frozen=True)
class FoundDocument:
    paths: tuple[Path, ...]  # 本地可读的文件；几张图片合成一份资料时有多个
    source: str  # 原始位置，压缩包里的文件写成「压缩包!内部路径」
    title: str  # 默认标题：文件名去掉扩展名

    @property
    def is_images(self) -> bool:
        return self.paths[0].suffix.lower() in IMAGE_SUFFIXES


@dataclass
class Expanded:
    documents: list[FoundDocument] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # （位置，原因）


# ---------- 自然排序 ----------

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_NUMBER = re.compile(r"\d+|[零〇一二两三四五六七八九十百千]+")


def _cn_number(text: str) -> int:
    total, digit = 0, 0
    for char in text:
        if char in _CN_DIGITS:
            digit = _CN_DIGITS[char]
        else:
            total += (digit or 1) * _CN_UNITS[char]
            digit = 0
    return total + digit


def natural_key(text: str) -> list[tuple[int, Any]]:
    """自然排序：数字（含中文数字）按数值比较，ch2 排在 ch10 前，第二章排在第十章前。"""
    key: list[tuple[int, Any]] = []
    position = 0
    for match in _NUMBER.finditer(text):
        if match.start() > position:
            key.append((1, text[position : match.start()].casefold()))
        token = match.group()
        key.append((0, int(token) if token.isdigit() else _cn_number(token)))
        position = match.end()
    if position < len(text):
        key.append((1, text[position:].casefold()))
    return key


# ---------- 展开 ----------


def expand(inputs: Sequence[Path], staging: Path) -> Expanded:
    """把输入展开成文档列表（按输入顺序，文件夹和压缩包内按自然顺序）。压缩包解压到 staging 下。"""
    result = Expanded()
    loose_images = [item for item in inputs if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES]
    for index, item in enumerate(inputs, start=1):
        if item not in loose_images:
            _expand_item(item, None, staging / f"input-{index}", result, depth=0)
    if loose_images:
        result.documents.append(FoundDocument(tuple(loose_images), str(loose_images[0]), loose_images[0].stem))
    return result


def _expand_item(item: Path, label: str | None, staging: Path, result: Expanded, depth: int) -> None:
    """label：压缩包里的条目显示为「压缩包!内部路径」；None 表示本机上的文件，显示完整路径。"""
    source = label or str(item)
    if item.is_symlink():
        result.skipped.append((source, "符号链接，已跳过"))
    elif item.is_dir():
        _expand_folder(item, label, item.name, staging, result, depth)
    elif not item.is_file():
        result.skipped.append((source, "找不到文件"))
    elif item.suffix.lower() in ARCHIVE_SUFFIXES:
        _expand_archive(item, source, staging, result, depth)
    else:
        _add_document(item, source, result)


def _expand_folder(folder: Path, label: str | None, name: str, staging: Path, result: Expanded, depth: int) -> None:
    """文件夹里的文档逐个成为资料；同一文件夹里的图片合成一份，标题用文件夹名。"""
    images: list[tuple[Path, str]] = []
    for index, child in enumerate(_sorted_children(folder), start=1):
        if child.name.lower() in _IGNORED_NAMES or child.name.startswith(("~$", "._")):
            continue
        child_label = _child_label(label, child)
        if child.is_file() and not child.is_symlink() and child.suffix.lower() in IMAGE_SUFFIXES:
            images.append((child, child_label))
            continue
        _expand_item(child, child_label if label is not None else None, staging / str(index), result, depth)
    if images:
        title = name if len(images) > 1 else images[0][0].stem
        result.documents.append(FoundDocument(tuple(path for path, _ in images), images[0][1], title))


def _child_label(label: str | None, child: Path) -> str:
    if label is None:
        return str(child)
    return f"{label}{child.name}" if label.endswith("!") else f"{label}/{child.name}"


def _sorted_children(folder: Path) -> list[Path]:
    return sorted(folder.iterdir(), key=lambda p: (p.is_dir(), natural_key(p.name)))


def _add_document(path: Path, source: str, result: Expanded) -> None:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_SUFFIXES:
        result.documents.append(FoundDocument((path,), source, path.stem))
    elif suffix in UNSUPPORTED_HINTS:
        result.skipped.append((source, UNSUPPORTED_HINTS[suffix]))
    else:
        result.skipped.append((source, f"不是支持的文档格式（{suffix or '无扩展名'}）"))


def _expand_archive(archive: Path, source: str, staging: Path, result: Expanded, depth: int) -> None:
    if depth >= MAX_DEPTH:
        result.skipped.append((source, "压缩包套了太多层，请先手动解压"))
        return
    target = staging / "files"
    target.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        extract_zip(archive, target)
    else:
        extract_with_tool(archive, target)
        _check_extracted(target, archive)
    _expand_folder(target, f"{source}!", archive.stem, staging / "nested", result, depth + 1)


# ---------- zip ----------


def extract_zip(archive: Path, target: Path) -> list[str]:
    """安全地解压 zip，返回解出的相对路径（文件名编码见 zip_names）。"""
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            _check_limits(len(infos), sum(info.file_size for info in infos), archive)
            if any(info.flag_bits & _ZIP_ENCRYPTED_FLAG for info in infos):
                raise ArchiveError(f"压缩包有密码，请先手动解压：{archive.name}")
            written = []
            for info, name in zip(infos, zip_names(infos)):
                relative = safe_relative(name)
                if relative is None:
                    continue
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                written.append(relative.as_posix())
            return written
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"压缩包已损坏或不是 zip 格式：{archive.name}") from exc


def zip_names(infos: Sequence[zipfile.ZipInfo]) -> list[str]:
    """各文件的正确文件名。没标 UTF-8 的文件名，标准库按 CP437 解码（中文会乱码），这里按原始字节重新识别。"""
    raw = [_raw_name(info) for info in infos]
    undecided = [data for data in raw if data is not None and not data.isascii()]
    encoding = detect_name_encoding(undecided) if undecided else "ascii"
    return [
        info.filename if data is None else data.decode(encoding, errors="replace").replace("\\", "/")
        for info, data in zip(infos, raw)
    ]


def _raw_name(info: zipfile.ZipInfo) -> bytes | None:
    """标准库按 CP437 兜底解码的文件名还原成原始字节；已正确解码的（UTF-8 标记或 Unicode 扩展字段）返回 None。"""
    if info.flag_bits & _ZIP_UTF8_FLAG:
        return None
    original = info.orig_filename.split("\x00")[0].replace(os.sep, "/")
    if info.filename != original:
        return None  # 有 Unicode 路径扩展字段，标准库已用它给出正确的名字
    return info.orig_filename.encode("cp437")


def detect_name_encoding(names: Sequence[bytes]) -> str:
    """识别一组文件名的编码：先看是不是 UTF-8，再在常见的东亚编码里选最可信的一个。"""
    data = b"\n".join(names)
    try:
        data.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    best = from_bytes(data, cp_isolation=NAME_ENCODINGS).best()
    return best.encoding if best is not None else "gb18030"


def safe_relative(name: str) -> Path | None:
    """压缩包内路径转成安全的相对路径：去掉盘符、开头的斜杠和 ".."，替换 Windows 不允许的字符；什么都不剩时返回 None。"""
    parts = []
    for part in PurePosixPath(name.replace("\\", "/")).parts:
        if part in ("/", ".", "..") or part.endswith(":"):
            continue
        clean = re.sub(r'[<>:"|?*\x00-\x1f]', "_", part).strip(" .")
        if clean.split(".")[0].upper() in _RESERVED:
            clean = f"_{clean}"
        if clean:
            parts.append(clean)
    return Path(*parts) if parts else None


def _check_limits(count: int, total: int, archive: Path) -> None:
    if count > MAX_FILES:
        raise ArchiveError(f"压缩包里的文件太多（{count} 个，上限 {MAX_FILES}）：{archive.name}")
    if total > MAX_TOTAL_BYTES:
        raise ArchiveError(f"压缩包解压后太大（{total / 1024**3:.1f} GB）：{archive.name}")


# ---------- rar / 7z：调用本机工具 ----------


def find_tool(suffix: str) -> list[str] | None:
    """返回解压这种格式的命令前缀；本机没有可用工具时返回 None。"""
    seven_zip = _which("7z", [str(BUNDLED_7ZIP), r"C:\Program Files\7-Zip\7z.exe",
                              r"C:\Program Files (x86)\7-Zip\7z.exe"])
    if seven_zip:
        return [seven_zip, "x", "-y", "-bd"]
    if suffix == ".rar":
        unrar = _which("unrar", [r"C:\Program Files\WinRAR\UnRAR.exe", r"C:\Program Files (x86)\WinRAR\UnRAR.exe"])
        if unrar:
            return [unrar, "x", "-idq", "-o-", "-p-", "-y"]  # -p- 不询问密码
    return None


def _which(name: str, candidates: Sequence[str]) -> str | None:
    return shutil.which(name) or next((path for path in candidates if Path(path).is_file()), None)


def extract_with_tool(archive: Path, target: Path) -> None:
    suffix = archive.suffix.lower()
    command = find_tool(suffix)
    if command is None:
        raise ArchiveError(
            f"本机没有能解压 {suffix} 的工具（7-Zip 或 WinRAR）：请先手动解压 {archive.name}，再导入解压出的文件夹"
        )
    if Path(command[0]).stem.lower() == "7z":
        args = [*command, f"-o{target}", str(archive)]
    else:
        args = [*command, str(archive), f"{target}{os.sep}"]
    try:
        # stdin 接空设备：有密码的压缩包会直接失败，而不是卡在等待输入
        completed = subprocess.run(args, capture_output=True, stdin=subprocess.DEVNULL, timeout=_TOOL_TIMEOUT,
                                   check=False)
    except subprocess.TimeoutExpired as exc:
        raise ArchiveError(f"解压超时：{archive.name}") from exc
    if completed.returncode not in _TOOL_OK:
        output = completed.stderr or completed.stdout
        message = output.decode("oem" if os.name == "nt" else "utf-8", "replace").strip()
        raise ArchiveError(f"解压 {archive.name} 失败（可能已损坏或有密码）：{message[:200]}")


def _check_extracted(target: Path, archive: Path) -> None:
    """外部工具解出的内容：去掉符号链接和跑出暂存目录的条目，并检查总量。"""
    root = target.resolve()
    count, total = 0, 0
    for path in list(target.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            path.unlink(missing_ok=True)
        elif path.is_file():
            count += 1
            total += path.stat().st_size
    _check_limits(count, total, archive)
