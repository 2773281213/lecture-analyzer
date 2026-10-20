"""桌面程序：用 pywebview（Windows 下是 WebView2）打开一个独立窗口，里面是本机的网页书架。

- 书库：按 LEC_CONFIG、当前目录的 config.toml、记下的书库文件夹查找（config.find_config_path）；都没有时先显示
  首次设置页：新建书库（默认"文档\\课程书架"）或选择已有的书库文件夹，新建的接着打开设置页填写模型和密钥；
- 网页服务只监听 127.0.0.1，端口由系统分配，在后台线程里运行；关窗后停止，后台任务随之结束（下次启动标为中断）；
- 原生的选文件、选文件夹对话框通过 JS 接口（DesktopApi）提供，网页里的 app.js 调用；
- 同一时间只运行一个书架：再次启动时把已经打开的窗口切到前台；
- 关窗时还有没结束的后台任务，先确认；
- 没有 WebView2 运行时的电脑（少数 Windows 10），改用系统浏览器打开书架，并提示下载地址。
"""

from __future__ import annotations

import ctypes
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from typing import Any

from . import apphome
from .books import VIDEO_SUFFIXES
from .config import ConfigError, find_config_path, load_config
from .documents import SUPPORTED_SUFFIXES

WINDOW_TITLE = "课程书架"
APP_ID = "LectureAnalyzer.Desktop"  # Windows 任务栏分组和单实例互斥体用的名字
WEBVIEW2_URL = "https://developer.microsoft.com/microsoft-edge/webview2/"
WEBVIEW2_CLIENT = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"  # WebView2 运行时在 EdgeUpdate 里的编号
ICON = Path(__file__).parent / "web" / "static" / "app.ico"
AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma"}
ARCHIVE_SUFFIXES = {".zip", ".rar", ".7z"}
SERVER_START_SECONDS = 30


def _filter(label: str, suffixes: set[str]) -> tuple[str, str]:
    patterns = ";".join(f"*{suffix}" for suffix in sorted(suffixes))
    return f"{label} ({patterns})", "所有文件 (*.*)"


FILE_TYPES = {
    "media": _filter("录音和录像", AUDIO_SUFFIXES | VIDEO_SUFFIXES),
    "documents": _filter("课件和资料", SUPPORTED_SUFFIXES | ARCHIVE_SUFFIXES),
}


# ---------- Windows 的几个小功能（ctypes） ----------

MB_OK, MB_OKCANCEL = 0x0, 0x1
MB_ICONWARNING, MB_ICONINFORMATION = 0x30, 0x40
MB_DEFBUTTON2, MB_TOPMOST = 0x100, 0x40000
IDOK = 1
ERROR_ALREADY_EXISTS = 183
SW_RESTORE = 9
_mutex: Any = None  # 单实例互斥体，进程结束前一直持有


def ask(message: str) -> bool:
    """确定 / 取消对话框，默认按钮是"取消"。"""
    flags = MB_OKCANCEL | MB_ICONWARNING | MB_DEFBUTTON2 | MB_TOPMOST
    return ctypes.windll.user32.MessageBoxW(None, message, WINDOW_TITLE, flags) == IDOK


def inform(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(None, message, WINDOW_TITLE, MB_OK | MB_ICONINFORMATION | MB_TOPMOST)


def already_running() -> bool:
    """同一时间只运行一个书架（命名互斥体）；已经在运行时把那个窗口切到前台。"""
    global _mutex
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    _mutex = kernel32.CreateMutexW(None, False, APP_ID)
    if ctypes.get_last_error() != ERROR_ALREADY_EXISTS:
        return False
    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = ctypes.c_void_p
    window = user32.FindWindowW(None, WINDOW_TITLE)
    if window:
        user32.ShowWindow(ctypes.c_void_p(window), SW_RESTORE)
        user32.SetForegroundWindow(ctypes.c_void_p(window))
    return True


def has_webview2() -> bool:
    """按微软文档的方法检查 WebView2 运行时：EdgeUpdate 里这个客户端的版本号 pv 存在且不是 0.0.0.0。"""
    import winreg

    places = (
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_CLIENT}"),
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_CLIENT}"),
        (winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_CLIENT}"),
    )
    for root, path in places:
        try:
            with winreg.OpenKey(root, path) as key:
                version = str(winreg.QueryValueEx(key, "pv")[0])
        except OSError:
            continue
        if version and version != "0.0.0.0":
            return True
    return False


def _log_to_file() -> None:
    """用 pythonw 启动时没有控制台，输出写进 %APPDATA%\\LectureAnalyzer\\desktop.log，便于排查问题。"""
    if sys.stdout is not None and sys.stderr is not None:
        return
    path = apphome.settings_file().with_name("desktop.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = stream


# ---------- 网页服务 ----------


class ServerThread:
    """在后台线程里运行网页服务（uvicorn）：只监听本机，端口由系统分配（直接交给 uvicorn 已绑定的套接字，没有抢端口的空档）。"""

    def __init__(self, app: Any) -> None:
        import uvicorn

        self.app = app
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.socket.getsockname()[1]}/"
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.socket]}, name="lec-web",
                                       daemon=True)

    def start(self) -> str:
        self.thread.start()
        deadline = time.monotonic() + SERVER_START_SECONDS
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("书架的网页服务没能启动，详情见 desktop.log")
            time.sleep(0.05)
        return self.url

    def stop(self) -> None:
        """停止服务：先执行应用的关闭流程（结束后台任务进程），再退出线程。"""
        self.server.should_exit = True
        self.thread.join(timeout=20)

    def active_jobs(self) -> int:
        from .jobs import JobStore
        from .library import Library

        with Library(self.app.state.cfg.data_dir / "lecture.db") as library:
            return len(JobStore(library).active_jobs())


# ---------- 网页能调用的本机功能 ----------


class DesktopApi:
    """网页通过 window.pywebview.api 调用的本机功能：选文件、打开文件夹、首次设置。

    下划线开头的属性不会暴露给网页（pywebview 的规则）。
    """

    def __init__(self, app: DesktopApp) -> None:
        self._app = app

    def pick(self, mode: str, types: str = "") -> list[str]:
        """选择文件（files 多个、file 一个）或文件夹（folder）；取消时返回空列表。"""
        import webview

        window = self._app.window
        if mode == "folder":
            chosen = window.create_file_dialog(webview.FileDialog.FOLDER)
        elif mode in ("files", "file"):
            chosen = window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=mode == "files",
                                               file_types=FILE_TYPES.get(types, ()))
        else:
            return []
        return [str(path) for path in chosen or ()]

    def open_path(self, path: str) -> bool:
        """在资源管理器里打开文件夹。只接受已经存在的文件夹，不打开文件（免得运行程序）。"""
        folder = Path(str(path))
        if not folder.is_absolute() or not folder.is_dir():
            return False
        os.startfile(folder)  # noqa: S606 —— 打开的是文件夹
        return True

    # ---------- 首次设置 ----------

    def default_home(self) -> str:
        return str(apphome.default_home())

    def choose_home(self) -> str:
        chosen = self.pick("folder")
        return chosen[0] if chosen else ""

    def use_home(self, path: str, create: bool) -> dict[str, str]:
        """新建（create）或打开已有的书库文件夹，记下位置并打开书架；出错时返回 {"error": 说明}。"""
        text = str(path).strip()
        home = Path(text).expanduser() if text else None
        if home is None or not home.is_absolute():
            return {"error": "请选择或填写一个完整的文件夹位置"}
        if not create and not apphome.is_library_folder(home):
            return {"error": "这个文件夹里没有书库（找不到 config.toml）。要新建书库，请用上面的「新建书库」。"}
        try:
            config = apphome.create_home(home) if create else home / apphome.CONFIG_NAME
            apphome.remember_home(home)
            self._app.open_library(config, welcome=create)
        except (OSError, ValueError, RuntimeError) as exc:  # 文件夹不能写、配置有误、服务没能启动
            return {"error": str(exc)}
        return {"error": ""}


# ---------- 窗口 ----------


def setup_page() -> str:
    """首次设置页（本地 HTML，样式与书架一致）。"""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    web = Path(__file__).parent / "web"
    env = Environment(loader=FileSystemLoader(web / "templates"), autoescape=select_autoescape())
    style = (web / "static" / "style.css").read_text(encoding="utf-8")
    return env.get_template("setup.html").render(style=style, default_home=str(apphome.default_home()))


class DesktopApp:
    def __init__(self) -> None:
        self.window: Any = None
        self.server: ServerThread | None = None

    def start_server(self, config: Path) -> str:
        from .web import create_app

        self.server = ServerThread(create_app(load_config(config)))
        return self.server.start()

    def open_library(self, config: Path, welcome: bool) -> None:
        """首次设置完成：启动网页服务，窗口转到书架（新建的书库先到设置页）。"""
        url = self.start_server(config)
        self.window.load_url(url + ("settings?welcome=1" if welcome else ""))

    def confirm_close(self) -> bool:
        """关窗前：还有没结束的后台任务时先确认；返回 False 表示不关。"""
        runner = self.server.app.state.jobs if self.server else None
        count = self.server.active_jobs() if runner is not None and runner.owner else 0  # 只有本进程执行的任务会中断
        if not count:
            return True
        return ask(f"还有 {count} 个后台任务没有完成。关闭书架会中断它们（已完成的步骤不会丢失，再次提交会接着做）。\n\n"
                   "确定要关闭吗？")

    def run(self, config: Path | None) -> None:
        import webview

        webview.settings["ALLOW_DOWNLOADS"] = True  # 考前报告的 Markdown 等可以下载
        options = {"js_api": DesktopApi(self), "width": 1280, "height": 860, "min_size": (900, 600),
                   "text_select": True, "zoomable": True, "background_color": "#F5EFE4"}
        if config is not None:
            self.window = webview.create_window(WINDOW_TITLE, self.start_server(config), **options)
        else:
            self.window = webview.create_window(WINDOW_TITLE, html=setup_page(), **options)
        self.window.events.closing += self.confirm_close
        try:
            webview.start(icon=str(ICON) if ICON.is_file() else None)
        finally:
            if self.server is not None:
                self.server.stop()


def run_in_browser(config: Path | None) -> None:
    """没有 WebView2 时：用系统浏览器打开书架；关掉提示框即退出。"""
    if config is None:
        inform(f"第一次使用需要 Microsoft Edge WebView2 运行时，请先从下面的地址下载安装：\n{WEBVIEW2_URL}")
        return
    app = DesktopApp()
    url = app.start_server(config)
    webbrowser.open(url)
    inform(f"书架已在浏览器里打开：{url}\n\n这台电脑没有 WebView2 运行时，无法打开独立窗口（下载地址：{WEBVIEW2_URL}）。\n"
           "关闭这个提示框会退出书架。")
    if app.server is not None:
        app.server.stop()


def find_library(explicit: Path | None = None) -> Path | None:
    try:
        return find_config_path(explicit)
    except ConfigError:
        return None


def main(config: Path | None = None) -> None:
    _log_to_file()
    if os.name == "nt":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)  # 任务栏显示书架的图标
        if already_running():
            return
    try:
        if os.name == "nt" and not has_webview2():
            run_in_browser(find_library(config))
        else:
            DesktopApp().run(find_library(config))
    except Exception as exc:  # 程序边界：没有控制台，出错时弹出说明，详细信息写进 desktop.log
        traceback.print_exc()
        inform(f"书架没能启动：{exc}")


if __name__ == "__main__":
    main()
