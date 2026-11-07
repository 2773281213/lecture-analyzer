"""工作台界面和主题：Cookie 里的界面偏好、外壳（导航栏 / 侧栏 / 内容 / 详情栏 / 状态栏）、六种主题。"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from lecture_analyzer.library import Library
from lecture_analyzer.pipeline.integrate import match_signal_note
from lecture_analyzer.pipeline.run import LectureRunner
from lecture_analyzer.schemas import LectureNotes, Signal
from lecture_analyzer.web import create_app
from lecture_analyzer.web.themes import THEMES, spine_var, ui_prefs

BASE = "http://127.0.0.1"


def _processed_client(env) -> TestClient:
    """先处理完这一讲（假转写、假模型），再给一个带界面的客户端。"""
    with Library(env.cfg.data_dir / "lecture.db") as library:
        LectureRunner(env.cfg, library, library.lecture_by_id(env.lecture_id), report=lambda _: None).run()
    return TestClient(create_app(env.cfg), base_url=BASE, follow_redirects=False)


def _request(**cookies: str) -> SimpleNamespace:
    """界面偏好只读 request.cookies，用这个假请求就够了。"""
    return SimpleNamespace(cookies=dict(cookies))


def _signal(name: str) -> Signal:
    """一条考点信号，只有知识点名称和匹配结果有关。"""
    return Signal(type="exam_hint", knowledge_point=name, strength=2, quote="原话", segment_ids=[1], slide_pages=[])


def _html_tag(page: str) -> str:
    """<html ...> 开始标签：主题、跟随系统深色、毛玻璃这些开关都在它上面。"""
    start = page.index("<html")
    return page[start:page.index(">", start) + 1]


def _app_tag(page: str) -> str:
    """外壳「<div class="app ...">」的开始标签：has-side / side-off 这些状态在它上面。"""
    start = page.index('<div class="app')
    return page[start:page.index(">", start) + 1]


def test_ui_prefs_defaults_when_no_cookie():
    """没有任何 Cookie 时：纸墨主题、跟随系统深色、毛玻璃、侧栏都开着，书架用列表。"""
    prefs = ui_prefs(_request())
    assert prefs["theme"] == "paper"
    assert prefs["autodark"] is True and prefs["glass"] is True and prefs["side"] is True
    assert prefs["shelf_view"] == "table"


def test_ui_prefs_unknown_theme_falls_back_to_paper():
    """不认识的 lec_theme 当作纸墨，认识的原样用。"""
    assert ui_prefs(_request(lec_theme="neon"))["theme"] == "paper"
    assert ui_prefs(_request(lec_theme="night"))["theme"] == "night"


def test_ui_prefs_zero_cookies_turn_switches_off():
    """各个 Cookie 写 "0" 就是把对应的开关关掉。"""
    prefs = ui_prefs(_request(lec_autodark="0", lec_glass="0", lec_side="0"))
    assert prefs["autodark"] is False
    assert prefs["glass"] is False
    assert prefs["side"] is False
    assert prefs["shelf_view"] == "table"  # 和这三个开关无关


def test_ui_prefs_shelf_view_books():
    """lec_shelf_view 只有 "books" 是书脊视图，其他值都是列表。"""
    assert ui_prefs(_request(lec_shelf_view="books"))["shelf_view"] == "books"
    assert ui_prefs(_request(lec_shelf_view="列表"))["shelf_view"] == "table"


def test_spine_var_cycles_every_six_courses():
    """书脊颜色按课程编号轮换六种：第 1 门和第 7 门同色。"""
    assert spine_var(1) == spine_var(7) == "var(--sp1)"
    assert spine_var(2) == "var(--sp2)"


def test_shell_renders_theme_from_cookie(m1_env):
    """主题在服务端就渲染在 <html> 上（打开页面不会先闪一下默认配色），不认识的值回落到纸墨。"""
    client = _processed_client(m1_env)
    client.cookies.set("lec_theme", "night")
    assert 'data-theme="night"' in _html_tag(client.get("/").text)
    client.cookies.set("lec_theme", "neon")  # 不认识的主题
    assert 'data-theme="paper"' in _html_tag(client.get("/").text)


def test_shell_has_autodark_by_default_and_can_turn_it_off(m1_env):
    """默认跟随系统深色（有 data-autodark），lec_autodark=0 时没有。"""
    client = _processed_client(m1_env)
    assert "data-autodark" in _html_tag(client.get("/").text)
    client.cookies.set("lec_autodark", "0")
    assert "data-autodark" not in _html_tag(client.get("/").text)


def test_shell_turns_glass_off_from_cookie(m1_env):
    """lec_glass=0 时 <html> 带 data-glass="off"。"""
    client = _processed_client(m1_env)
    assert "data-glass" not in _html_tag(client.get("/").text)
    client.cookies.set("lec_glass", "0")
    assert 'data-glass="off"' in _html_tag(client.get("/").text)


def test_theme_menu_lists_every_theme(m1_env):
    """主题菜单里每种主题都有一个按钮，挨个数在 THEMES 里。"""
    page = _processed_client(m1_env).get("/").text
    assert 'id="theme-menu"' in page
    for theme in THEMES:
        assert f'data-set-theme="{theme["id"]}"' in page


def test_main_pages_all_render_inside_the_shell(m1_env):
    """主要页面都用外壳渲染；没有侧栏或详情栏内容的页面（出错页）也不能报 500。"""
    client = _processed_client(m1_env)
    course_id, lecture_id = m1_env.course_id, m1_env.lecture_id
    urls = [
        "/",
        f"/courses/{course_id}",
        f"/lectures/{lecture_id}",
        f"/courses/{course_id}/review",
        f"/courses/{course_id}/qa",
        f"/courses/{course_id}/settings",
        "/courses/new",
        "/jobs",
        "/settings",
    ]
    for url in urls:
        page = client.get(url)
        assert page.status_code == 200, url
        assert 'class="app' in page.text, url
    missing = client.get("/courses/999")
    assert missing.status_code == 404 and 'class="app' in missing.text  # 出错页也走外壳


def test_shelf_has_table_rows_and_detail_panels(m1_env):
    """书架首页：课程表每门课一行、右栏每门课一块详情。"""
    page = _processed_client(m1_env).get("/").text
    assert 'class="table" id="shelf-table"' in page
    assert f'data-select="course-{m1_env.course_id}"' in page
    assert f'data-panel="course-{m1_env.course_id}"' in page


def test_course_page_has_lecture_table_and_flow(m1_env):
    """课程页：讲次表每讲一行，右栏是选中那一讲的详情，里面有处理流程。"""
    page = _processed_client(m1_env).get(f"/courses/{m1_env.course_id}").text
    assert 'id="lecture-table"' in page
    assert f'data-select="lecture-{m1_env.lecture_id}"' in page
    start = page.index(f'data-panel="lecture-{m1_env.lecture_id}"')
    panel_tag = page[page.rindex("<section", 0, start):page.index(">", start) + 1]
    assert "hidden" not in panel_tag  # 唯一的这一讲就是选中的
    assert 'class="flow"' in page


def test_lecture_page_has_timeline_transcript_and_toc(m1_env):
    """讲的阅读页：右栏时间轴、知识点色条、转写和跟随播放；侧栏是本讲目录和讲次翻页。"""
    page = _processed_client(m1_env).get(f"/lectures/{m1_env.lecture_id}").text
    assert 'id="timeline"' in page and "data-total=" in page
    assert 'class="tl-band"' in page
    assert 'id="transcript"' in page and 'id="follow-playback"' in page
    assert 'href="#kp-1"' in page
    assert "第 3 讲" in page and "/ 共 1 讲" in page
    assert "is-disabled" in page  # 只有一讲：上一讲、下一讲都没有


def test_settings_page_previews_every_theme(m1_env):
    """设置页的外观部分：六种主题各一张卡片。"""
    page = _processed_client(m1_env).get("/settings").text
    assert 'id="appearance"' in page
    assert page.count('class="theme-card"') == len(THEMES)


def test_shelf_view_cookie_switches_to_books(m1_env):
    """lec_shelf_view=books 时首页内容区换成书脊视图。"""
    client = _processed_client(m1_env)
    assert "shelf-view-table" in client.get("/").text
    client.cookies.set("lec_shelf_view", "books")
    assert "shelf-view-books" in client.get("/").text


def test_side_cookie_collapses_the_side_column(m1_env):
    """lec_side=0 时外壳带上 side-off（侧栏收起来，内容区变宽）。"""
    client = _processed_client(m1_env)
    assert "side-off" not in _app_tag(client.get(f"/courses/{m1_env.course_id}").text)
    client.cookies.set("lec_side", "0")
    assert "side-off" in _app_tag(client.get(f"/courses/{m1_env.course_id}").text)


def test_lecture_page_signals_before_outline_and_knowledge(m1_env):
    """讲的阅读页先看到考点信号，再是大纲和知识点；侧栏目录里考点信号也排在最前。"""
    page = _processed_client(m1_env).get(f"/lectures/{m1_env.lecture_id}").text
    assert page.index('id="signals"') < page.index('id="outline"') < page.index('id="knowledge"')
    assert page.index('href="#signals"') < page.index('href="#outline"')


def test_signal_table_links_to_the_lecture_knowledge_point(m1_env):
    """信号表格里的知识点名称链到本讲对应的知识点卡片（夹具的整合结果把所有信号都归到第 1 个知识点）。"""
    page = _processed_client(m1_env).get(f"/lectures/{m1_env.lecture_id}").text
    assert '<a class="kp-link" href="#kp-1">阿司匹林的剂量分段</a>' in page


def test_match_signal_note_by_name():
    """信号写的知识点名称按名称匹配到本讲第几个知识点：完全相同、略有出入都能对上，毫不相干返回 None。"""
    notes = LectureNotes.model_validate({
        "title": "药理学",
        "outline": ["阿司匹林的剂量分段", "受体激动药"],
        "knowledge_points": [
            {"name": "阿司匹林的剂量分段", "summary_md": "小于 75 mg 抗血凝", "slide_pages": [], "segment_ids": [3],
             "minutes": 1.5},
            {"name": "受体激动药", "summary_md": "剂量与效应", "slide_pages": [], "segment_ids": [1], "minutes": 2},
        ],
        "examples": [],
        "homework": [],
        "signals": [],
    })
    assert match_signal_note(_signal("受体激动药"), notes) == 2
    assert match_signal_note(_signal("受体激动药有哪些"), notes) == 2
    assert match_signal_note(_signal("病毒命名与分类"), notes) is None


def test_signal_priority_levels():
    """单条信号的重要程度和知识点的星级同一套标准：提示考试、考试范围、强度 3 的强调是 3，
    强调、易错是 2，说不考是 0，其余（推导、回顾、例题等）是 1。"""
    from lecture_analyzer.pipeline.review import signal_priority

    assert signal_priority("exam_hint", 3) == 3
    assert signal_priority("exam_hint", 1) == 3
    assert signal_priority("scope_include", 2) == 3
    assert signal_priority("emphasis", 3) == 3
    assert signal_priority("emphasis", 2) == 2
    assert signal_priority("emphasis", 1) == 2
    assert signal_priority("common_mistake", 1) == 2
    assert signal_priority("scope_exclude", 3) == 0
    assert signal_priority("worked_example", 2) == 1
    assert signal_priority("recap", 1) == 1
    assert signal_priority("derivation", 3) == 1


def test_lecture_page_shows_only_key_signals_and_examples(m1_env, monkeypatch):
    """讲的阅读页只显示绝对重点、次重点和例题：一般信号不显示，只在表格下面提一句还有几条。"""
    from dataclasses import replace

    client = _processed_client(m1_env)
    page = client.get(f"/lectures/{m1_env.lecture_id}").text
    # 夹具的两条信号（提示考试强度 3、强调强度 1）都是重点：星级标签都在，没有隐藏的条数
    assert 'title="绝对重点"' in page and 'title="次重点"' in page
    assert "另有" not in page

    # 再补一条回顾信号和一条例题信号：回顾的不显示，例题的照常显示
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        original = library.signals(m1_env.lecture_id)
    recap_quote, example_quote = "回顾一下上节课讲的内容", "比如这道题：求半衰期"
    recap = replace(original[0], signal=original[0].signal.model_copy(
        update={"type": "recap", "quote": recap_quote}))
    example = replace(original[0], signal=original[0].signal.model_copy(
        update={"type": "worked_example", "quote": example_quote}))
    stored_signals = Library.signals

    def with_extra(self, lecture_id: int):
        found = stored_signals(self, lecture_id)
        return found + [recap, example] if lecture_id == m1_env.lecture_id else found

    monkeypatch.setattr(Library, "signals", with_extra)
    page = client.get(f"/lectures/{m1_env.lecture_id}").text
    assert "另有 1 条一般信号" in page  # 只看回顾这一条没显示
    assert "考点信号 3 条" in page  # 状态栏：两条重点 + 一条例题
    assert 'title="绝对重点"' in page and 'title="次重点"' in page
    table = page[page.index('id="signals"'):page.index('id="outline"')]
    assert example_quote in table and recap_quote not in table
    assert recap_quote not in page  # 回顾那条哪儿都不显示
