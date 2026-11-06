"""外部命令（ffmpeg、7-Zip、taskkill 等）的公共开关。

桌面程序本身没有控制台（pythonw / 窗口程序）：子进程不带 CREATE_NO_WINDOW 时，Windows 会给它单独开一个
控制台窗口，在屏幕上闪一下。转写时每一片音频都要调一次 ffmpeg 转 mp3、被拦截时还要切小再转，一个任务能闪
几十次，所以所有外部命令一律带上这个开关。其他平台取 0，等于不加。
"""

from __future__ import annotations

import subprocess

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
