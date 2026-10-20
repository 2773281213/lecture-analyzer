"""列出安装包随附的第三方组件和许可证，写成 THIRD-PARTY-NOTICES.txt（build.ps1 用打包好的运行时执行）。

Python 包的名称、版本、许可证取自各包的元数据；完整的许可证文本在 runtime\\Lib\\site-packages\\<包>.dist-info 里。
用法：runtime\\python.exe -I notices.py <输出文件>
"""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

HEADER = """课程书架（lecture-analyzer）随附的第三方组件
==============================================

- Python {python}（python-build-standalone；Python Software Foundation License），在 runtime\\
- 7-Zip（GNU LGPL，rar 解压部分另有 unRAR 许可限制），在 tools\\7zip\\，许可证见 tools\\7zip\\License.txt
- FFmpeg（imageio-ffmpeg 随附的 ffmpeg 程序；GPL / LGPL），见 runtime\\Lib\\site-packages\\imageio_ffmpeg
- PyTorch 的 CUDA 版随附 NVIDIA CUDA 运行库（NVIDIA 软件许可协议），见 runtime\\Lib\\site-packages\\torch
- 本地语音识别模型（Paraformer、SenseVoice、Qwen3-ASR 等）不在安装包里，第一次使用时从 ModelScope 下载，许可证见各模型页面

随附的 Python 包（名称 版本：许可证）：
"""


def license_of(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    expression = meta.get("License-Expression")
    if expression:
        return expression
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        return "、".join(dict.fromkeys(classifiers))
    lines = (meta.get("License") or "").strip().splitlines()
    return lines[0][:100] if lines else "（元数据里没有写，见包内的许可证文件）"


def main() -> None:
    rows = sorted({(dist.metadata["Name"], dist.version, license_of(dist)) for dist in metadata.distributions()},
                  key=lambda row: row[0].lower())
    text = HEADER.format(python=sys.version.split()[0])
    text += "\n".join(f"- {name} {version}：{license_text}" for name, version, license_text in rows) + "\n"
    Path(sys.argv[1]).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
