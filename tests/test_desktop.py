"""桌面程序：网页调用的本机功能（用测试替身代替真实窗口）、首次设置、网页服务线程、关窗确认、WebView2 检查。"""

import sys
import urllib.request
from types import SimpleNamespace

import pytest
import webview
from webview.util import parse_file_type

from lecture_analyzer import apphome, desktop
from lecture_analyzer.config import load_config
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.jobs import JobStore
from lecture_analyzer.library import Library
from lecture_analyzer.web import create_app


class FakeWindow:
    """记下要求打开的文件对话框，返回预设的选择结果（None 表示取消）。"""

    def __init__(self, result=None):
        self.result = result
        self.dialogs = []

    def create_file_dialog(self, dialog_type, directory="", allow_multiple=False, save_filename="", file_types=()):
        self.dialogs.append((int(dialog_type), allow_multiple, tuple(file_types)))
        return self.result


class FakeApp:
    def __init__(self, window=None, error=None):
        self.window = window
        self.error = error
        self.opened = []

    def open_library(self, config, welcome):
        if self.error:
            raise self.error
        self.opened.append((config, welcome))


def test_pick_uses_native_dialogs():
    window = FakeWindow(("D:\\录课\\第4讲-1.mp4", "D:\\录课\\第4讲-2.mp4"))
    api = desktop.DesktopApi(FakeApp(window))
    assert api.pick("files", "media") == ["D:\\录课\\第4讲-1.mp4", "D:\\录课\\第4讲-2.mp4"]
    assert api.pick("file", "documents") and api.pick("folder") and api.pick("everything") == []
    opening, folder = int(webview.FileDialog.OPEN), int(webview.FileDialog.FOLDER)
    assert [dialog[:2] for dialog in window.dialogs] == [(opening, True), (opening, False), (folder, False)]
    media, documents = window.dialogs[0][2][0], window.dialogs[1][2][0]
    assert "*.m4a" in media and "*.mp4" in media and "*.pptx" in documents and "*.rar" in documents
    for file_type in (*desktop.FILE_TYPES["media"], *desktop.FILE_TYPES["documents"]):
        parse_file_type(file_type)  # pywebview 认得这种写法
    window.result = None  # 取消
    assert api.pick("files", "media") == []


def test_open_path_only_opens_existing_folders(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(desktop.os, "startfile", opened.append, raising=False)
    program = tmp_path / "程序.exe"
    program.write_bytes(b"MZ")
    api = desktop.DesktopApi(FakeApp())
    assert not api.open_path(str(program))  # 不打开文件，免得运行程序
    assert not api.open_path(str(tmp_path / "没有")) and not api.open_path("相对路径")
    assert api.open_path(str(tmp_path)) and opened == [tmp_path]


def test_first_run_creates_or_opens_a_library(tmp_path):
    app = FakeApp(FakeWindow())
    api = desktop.DesktopApi(app)
    assert api.default_home() == str(apphome.default_home())
    assert "完整的文件夹位置" in api.use_home("相对路径", True)["error"]
    assert "找不到 config.toml" in api.use_home(str(tmp_path / "空的"), False)["error"]
    home = tmp_path / "我的书库"
    assert api.use_home(f"  {home}  ", True) == {"error": ""}
    assert (home / "config.toml").is_file() and (home / "data").is_dir()
    assert apphome.saved_home() == home.resolve()
    assert api.use_home(str(home), False) == {"error": ""}
    assert app.opened == [(home / "config.toml", True), (home / "config.toml", False)]  # 新建的先到设置页
    failing = desktop.DesktopApi(FakeApp(FakeWindow(), error=RuntimeError("书架的网页服务没能启动")))
    assert failing.use_home(str(home), False) == {"error": "书架的网页服务没能启动"}


def test_setup_page_and_find_library(tmp_path, monkeypatch, example_config):
    html = desktop.setup_page()
    assert "新建书库" in html and "使用已有的书库" in html and "--paper" in html  # 样式直接嵌入
    assert str(apphome.default_home()) in html
    empty = tmp_path / "别处"
    empty.mkdir()
    monkeypatch.chdir(empty)
    assert desktop.find_library() is None  # 没有书库：先显示首次设置
    assert desktop.find_library(example_config) == example_config.resolve()


def test_server_thread_serves_and_close_asks_about_jobs(example_config, monkeypatch):
    cfg = load_config(example_config)
    app = desktop.DesktopApp()
    assert app.confirm_close()  # 还没打开书库
    server = desktop.ServerThread(create_app(cfg))
    app.server = server
    url = server.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 不走系统代理
    with opener.open(url, timeout=10) as response:
        assert response.status == 200 and "课程书架" in response.read().decode("utf-8")
    assert app.confirm_close()  # 没有后台任务时直接关
    with Library(cfg.data_dir / "lecture.db") as library:
        JobStore(library).add(library.add_course("病毒学", CourseSettings()), "rebuild", {})
    questions = []
    monkeypatch.setattr(desktop, "ask", lambda message: questions.append(message) or False)
    assert not app.confirm_close() and "还有 1 个后台任务没有完成" in questions[0]
    server.stop()
    assert not server.thread.is_alive()
    with Library(cfg.data_dir / "lecture.db") as library:
        assert JobStore(library).active_jobs()[0].state == "queued"  # 只有正在运行的任务才会被结束


@pytest.mark.parametrize(("versions", "expected"), [({}, False), ({"HKLM": "0.0.0.0"}, False),
                                                    ({"HKCU": "141.0.3537.71"}, True)])
def test_webview2_detection(monkeypatch, versions, expected):
    keys = {"HKLM": 1, "HKCU": 2}

    class Key:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def open_key(root, path):
        name = "HKCU" if root == keys["HKCU"] else "HKLM"
        if name not in versions or "WOW6432Node" not in path and name == "HKLM":
            raise OSError("没有这个注册表项")
        return Key(name)

    fake = SimpleNamespace(HKEY_LOCAL_MACHINE=keys["HKLM"], HKEY_CURRENT_USER=keys["HKCU"], OpenKey=open_key,
                           QueryValueEx=lambda key, name: (versions[key.name], 1))
    monkeypatch.setitem(sys.modules, "winreg", fake)
    assert desktop.has_webview2() is expected
