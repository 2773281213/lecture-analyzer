"""从书架的书本标志生成程序图标 app.ico（窗口、任务栏、启动程序和网页的 favicon 共用）。

用 PyMuPDF 把 SVG 渲染成各尺寸的 PNG，再按 ICO 格式拼起来（ICO 里直接存 PNG，Windows Vista 起支持），
不需要额外的图像库。用法：uv run python packaging/make_icon.py [输出文件]
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pymupdf

# 圆角方块（书架的主色）上放三本书（与网页顶栏的标志同一个路径）
SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
<rect x="2" y="2" width="60" height="60" rx="14" fill="#2f6f62"/>
<g transform="translate(8 8) scale(2)" fill="#f5efe4">
<path d="M4 4h4v16H4zM10 4h4v16h-4zM16.2 5.3l3.8-1 3.1 15.5-3.9 1z"/>
</g>
</svg>"""
SIZES = (16, 24, 32, 48, 64, 128, 256)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "src" / "lecture_analyzer" / "web" / "static" / "app.ico"


def render_png(size: int) -> bytes:
    page = pymupdf.open(stream=SVG.encode("utf-8"), filetype="svg")[0]
    zoom = size / page.rect.width
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=True).tobytes("png")


def build_ico(images: list[tuple[int, bytes]]) -> bytes:
    """ICO 文件：6 字节文件头、每张图 16 字节目录项，再接各张 PNG。宽高 256 在目录项里写 0。"""
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries, data = b"", b""
    for size, png in images:
        side = size % 256
        entries += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(png), offset + len(data))
        data += png
    return header + entries + data


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    output.write_bytes(build_ico([(size, render_png(size)) for size in SIZES]))
    print(f"已生成 {output}")


if __name__ == "__main__":
    main()
