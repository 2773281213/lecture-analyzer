"""界面主题：六种配色，样式表里每种是一组 CSS 变量（html 的 data-theme），布局完全相同。

选择记在 Cookie 里，由页面脚本写入、这里读出，服务端渲染时直接带上主题，打开页面不会先闪一下默认配色。
Cookie 只认主机不认端口：桌面程序每次启动端口不同，选择也能记住（WebView2 用固定的存储目录）。

- lec_theme：主题编号，默认纸墨；
- lec_autodark：系统是深色模式时自动换成「夜读」，默认开（和以前的版本一样跟随系统）；
- lec_glass：极光主题的毛玻璃效果，默认开，核显电脑上可以关掉；
- lec_shelf_view：书架首页用列表还是书脊卡片；
- lec_side：是否显示左侧栏（窄窗口下默认收起）。
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request

THEMES: list[dict[str, Any]] = [
    {"id": "paper", "name": "纸墨", "full": "纸墨 · 书房", "en": "Paper & Ink", "dark": False,
     "dots": ("#fffcf6", "#8a5a35", "#2f6f62"),
     "desc": "米色纸面配木色与墨绿，标题用宋体，延续原来的书架风格。"},
    {"id": "fluent", "name": "云母", "full": "云母 · Windows 11", "en": "Fluent / Mica", "dark": False,
     "dots": ("#dbe5f4", "#ffffff", "#005fb8"),
     "desc": "淡彩云母背景、白色圆角内容区和分组卡片，像 Windows 11 自带的软件。"},
    {"id": "mono", "name": "素白", "full": "素白 · 荧光笔", "en": "Mono & Highlighter", "dark": False,
     "dots": ("#ffffff", "#191919", "#fff0a0"),
     "desc": "黑白为主、大量留白，当前句和重点用荧光黄标出，最不累眼。"},
    {"id": "academy", "name": "书院", "full": "书院 · 墨绿金", "en": "Academy Green", "dark": False,
     "dots": ("#1b3730", "#fbf8f1", "#c89b3c"),
     "desc": "深墨绿导航配象牙色内容，金色点缀选中项和重点，沉稳有分量。"},
    {"id": "night", "name": "夜读", "full": "夜读 · 深色台灯", "en": "Night Reading", "dark": True,
     "dots": ("#121317", "#1f2128", "#e8b15a"),
     "desc": "深灰分层配琥珀色，像台灯只照亮正在看的内容，适合晚上复习。"},
    {"id": "aurora", "name": "极光", "full": "极光 · 毛玻璃", "en": "Aurora Glass", "dark": False,
     "dots": ("#c6d3ff", "#f7cdee", "#6d5dfc"),
     "desc": "渐变背景上浮着半透明玻璃面板，紫色渐变按钮，轻盈新潮。"},
]
THEME_IDS = {theme["id"] for theme in THEMES}
DEFAULT_THEME = "paper"
DARK_THEME = "night"  # 跟随系统深色模式时换成的主题


def ui_prefs(request: Request) -> dict[str, Any]:
    """这次请求的界面偏好（来自 Cookie，取值不认识时用默认值）。"""
    cookies = request.cookies
    theme = cookies.get("lec_theme", DEFAULT_THEME)
    return {
        "theme": theme if theme in THEME_IDS else DEFAULT_THEME,
        "autodark": cookies.get("lec_autodark", "1") != "0",
        "glass": cookies.get("lec_glass", "1") != "0",
        "shelf_view": "books" if cookies.get("lec_shelf_view") == "books" else "table",
        "side": cookies.get("lec_side", "1") != "0",
    }


def spine_var(course_id: int) -> str:
    """课程的书脊颜色：按课程编号轮换六种，各主题自己定义（深色主题用更亮的颜色），和 views.SPINE_COLORS 的顺序一致。"""
    return f"var(--sp{(course_id - 1) % 6 + 1})"
