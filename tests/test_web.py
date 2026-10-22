from fastapi.testclient import TestClient

from lecture_analyzer.course import CourseSettings
from lecture_analyzer.library import Library
from lecture_analyzer.pipeline.run import LectureRunner
from lecture_analyzer.web import create_app
from lecture_analyzer.web.app import render_markdown

BASE = "http://127.0.0.1"


def _processed_client(env) -> TestClient:
    with Library(env.cfg.data_dir / "lecture.db") as library:
        LectureRunner(env.cfg, library, library.lecture_by_id(env.lecture_id), report=lambda _: None).run()
    return TestClient(create_app(env.cfg), base_url=BASE, follow_redirects=False)


def _post(client: TestClient, url: str, **fields):
    token = client.app.state.csrf_token
    return client.post(url, data={"csrf": token, **fields})


def test_shelf_shows_course_books(m1_env):
    page = _processed_client(m1_env).get("/")
    assert page.status_code == 200
    assert "药理学" in page.text and "1 讲" in page.text and "已上架 1" in page.text
    assert f'href="/courses/{m1_env.course_id}"' in page.text and "复习 0/1 个知识点" in page.text


def test_course_page_lists_lectures_and_knowledge_points(m1_env):
    page = _processed_client(m1_env).get(f"/courses/{m1_env.course_id}")
    assert page.status_code == 200
    assert "第 3 讲" in page.text and "量效关系" in page.text
    assert "药理学总论" in page.text and "阿司匹林的剂量分段" in page.text
    assert f'href="/lectures/{m1_env.lecture_id}?t=2000"' in page.text  # 知识点出处第 3 句从 2 秒开始
    assert f'href="/courses/{m1_env.course_id}/review"' in page.text


def test_lecture_page_has_notes_memory_player_and_forms(m1_env):
    client = _processed_client(m1_env)
    page = client.get(f"/lectures/{m1_env.lecture_id}")
    assert page.status_code == 200
    assert "阿司匹林的剂量分段" in page.text and 'data-t="2000"' in page.text
    assert f'src="/media/{m1_env.lecture_id}/1"' in page.text and 'id="parts-data"' in page.text
    assert "首次讲" in page.text and "K1 阿司匹林的剂量分段" in page.text
    assert "未找到原话" in page.text and "通过" in page.text
    assert client.app.state.csrf_token in page.text  # 表单带着令牌


def test_multi_part_lecture_plays_as_one(m1_env):
    # M4 验收：多节的讲像被分成了几段——原生控件只显示当前这一节，转写里插着「第 N 节」
    client = _processed_client(m1_env)
    assert 'id="player-controls"' not in client.get(f"/lectures/{m1_env.lecture_id}").text  # 一节：原生控件就是整讲
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        library.set_part_duration(m1_env.lecture_id, 1, 2_688_512)
        library.add_part(m1_env.lecture_id, m1_env.media)
        unknown = client.get(f"/lectures/{m1_env.lecture_id}").text
        library.set_part_duration(m1_env.lecture_id, 2, 2_678_720)
    assert 'id="player-controls"' not in unknown  # 还有节的时长未知（抽音频之前）：保留原生控件逐节播放
    page = client.get(f"/lectures/{m1_env.lecture_id}").text
    assert 'id="player-controls"' in page and 'max="5367232"' in page and "/ 1:29:27" in page  # 整讲一条进度条
    assert '"offset": 2688512' in page and "第 2 节" not in page  # 第 2 节接在整讲时间轴上，转写不插分节线
    assert 'class="pc-full"' not in page  # 音频没有全屏


def test_old_book_link_redirects(m1_env):
    response = _processed_client(m1_env).get(f"/books/{m1_env.lecture_id}")
    assert response.status_code == 301 and response.headers["location"] == f"/lectures/{m1_env.lecture_id}"


def test_media_parts_support_range_requests(m1_env):
    client = _processed_client(m1_env)
    response = client.get(f"/media/{m1_env.lecture_id}/1", headers={"Range": "bytes=0-99"})
    assert response.status_code == 206 and len(response.content) == 100
    assert client.get(f"/media/{m1_env.lecture_id}/2").status_code == 404


def test_unknown_pages_are_404(m1_env):
    client = _processed_client(m1_env)
    assert client.get("/lectures/999").status_code == 404
    assert client.get("/courses/999").status_code == 404
    assert client.get("/media/999/1").status_code == 404


def test_missing_media_file_is_404(m1_env):
    client = _processed_client(m1_env)
    m1_env.media.unlink()
    assert client.get(f"/media/{m1_env.lecture_id}/1").status_code == 404


def test_json_api(m1_env):
    client = _processed_client(m1_env)
    lecture = client.get(f"/api/lectures/{m1_env.lecture_id}").json()
    assert lecture["schema_version"] == 3 and lecture["notes"]["title"] == "量效关系"
    book = client.get(f"/api/courses/{m1_env.course_id}").json()
    assert book["course"]["name"] == "药理学" and book["knowledge_points"][0]["key"] == 1


def test_foreign_host_is_rejected(m1_env):
    """防 DNS rebinding：Host 不是本机时拒绝，连只读页面也不给。"""
    _processed_client(m1_env)
    evil = TestClient(create_app(m1_env.cfg), base_url="http://attacker.example")
    assert evil.get("/").status_code == 400
    assert TestClient(create_app(m1_env.cfg), base_url="http://localhost:8765").get("/").status_code == 200


def test_writes_require_token_and_small_body(m1_env):
    client = _processed_client(m1_env)
    url = f"/lectures/{m1_env.lecture_id}/tags"
    assert client.post(url, data={"tag": "期末重点"}).status_code == 403
    assert client.post(url, data={"tag": "期末重点", "csrf": "猜的令牌"}).status_code == 403
    assert _post(client, url, tag="期末重点", padding="x" * 5000).status_code == 413
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        assert library.tags(m1_env.lecture_id) == []


def test_tags_can_be_added_and_removed(m1_env):
    client = _processed_client(m1_env)
    added = _post(client, f"/lectures/{m1_env.lecture_id}/tags", tag="期末重点")
    assert added.status_code == 303 and added.headers["location"] == f"/lectures/{m1_env.lecture_id}#organize"
    course_page = client.get(f"/courses/{m1_env.course_id}?tag=期末重点")
    assert "量效关系" in course_page.text
    assert "没有带这个标签的讲" in client.get(f"/courses/{m1_env.course_id}?tag=已复习").text
    bad = _post(client, f"/lectures/{m1_env.lecture_id}/tags", tag="x" * 30)
    assert bad.status_code == 400 and "标签" in bad.text
    _post(client, f"/lectures/{m1_env.lecture_id}/tags/remove", tag="期末重点")
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        assert library.tags(m1_env.lecture_id) == []


def test_review_progress_toggle(m1_env):
    client = _processed_client(m1_env)
    url = f"/courses/{m1_env.course_id}/reviewed"
    response = _post(client, url, name="阿司匹林的剂量分段", key="1", reviewed="1")
    assert response.status_code == 303 and response.headers["location"] == f"/courses/{m1_env.course_id}#k-1"
    assert "复习 1/1 个知识点" in client.get("/").text
    _post(client, url, name="阿司匹林的剂量分段", key="1", reviewed="0", back="review")
    assert "复习 0/1 个知识点" in client.get("/").text
    assert _post(client, url, name="", key="1", reviewed="1").status_code == 400


def test_course_order_and_lecture_move(m1_env):
    client = _processed_client(m1_env)
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        other = library.add_course("病毒学", CourseSettings(subject="药理学"))
    assert _post(client, f"/courses/{other.id}/position", direction="up").status_code == 303
    page = client.get("/").text
    assert page.index(f'href="/courses/{other.id}"') < page.index(f'href="/courses/{m1_env.course_id}"')
    assert _post(client, f"/courses/{other.id}/position", direction="sideways").status_code == 400

    moved = _post(client, f"/lectures/{m1_env.lecture_id}/move", course_id=str(other.id), seq="")
    assert moved.status_code == 303
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        lecture = library.lecture_by_id(m1_env.lecture_id)
        assert lecture.course_id == other.id and lecture.seq == 3
        assert lecture.step_state("integrate") == "" and lecture.step_state("extract") == "done"  # 学科相同，只需重新整合
    assert (m1_env.cfg.data_dir / "books" / "病毒学" / "第03讲" / "lecture.json").is_file()
    assert not (m1_env.cfg.data_dir / "books" / "药理学" / "第03讲").exists()
    assert _post(client, f"/lectures/{m1_env.lecture_id}/move", course_id="abc").status_code == 400


def test_markdown_protects_math_and_blocks_raw_html():
    html = str(render_markdown("公式 $a*b*c$ 和 **重点**\n\n<script>alert(1)</script>"))
    assert "$a*b*c$" in html and "<strong>重点</strong>" in html
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_markdown_bold_next_to_cjk_after_punctuation():
    html = str(render_markdown("최종 목적지가 **장(소장·대장)**인 바이러스, **육량체(hexamer)**가 생긴다"))
    assert "<strong>장(소장·대장)</strong>인" in html and "<strong>육량체(hexamer)</strong>가" in html
    assert "**" not in html


def test_markdown_bold_fix_does_not_unescape_html():
    html = str(render_markdown("**<b>x</b>(y)**가"))
    assert "<b>" not in html


def test_markdown_keeps_plain_numbers():
    """公式占位符不能和正文里的数字混淆。"""
    html = str(render_markdown("小于 75 mg 时 $x_1$，大于 300 mg 时 $x_2$"))
    assert "75 mg" in html and "300 mg" in html and "$x_1$" in html and "$x_2$" in html


def test_empty_library_shows_help(example_config):
    from lecture_analyzer.config import load_config

    page = TestClient(create_app(load_config(example_config)), base_url=BASE).get("/")
    assert page.status_code == 200 and "书架还是空的" in page.text


def test_review_shows_must_points_and_folds_the_rest(m1_env, monkeypatch):
    """总复习每章只展开 ★★★ 必考知识点，其余折叠在章末；目录里非必考的条目淡化显示。"""
    from dataclasses import replace

    from lecture_analyzer.web import views

    real = views.review_book

    def with_minor_point(library, course):
        chapters = real(library, course)
        first = chapters[0]
        minor = replace(first.entries[0], key=99, name="肝药酶诱导", priority=1)
        return [replace(first, entries=[*first.entries, minor]), *chapters[1:]]

    monkeypatch.setattr(views, "review_book", with_minor_point)
    page = _processed_client(m1_env).get(f"/courses/{m1_env.course_id}/review").text
    folded = page.index('<details class="rest-points">')
    assert page.index('id="k-1"') < folded < page.index('id="k-99"')
    assert "其余 1 个知识点（★★ 及以下" in page and 'class="toc-rest"><a href="#k-99"' in page
    focus = _processed_client(m1_env).get(f"/courses/{m1_env.course_id}/review?focus=1").text
    assert "肝药酶诱导" not in focus and "rest-points" not in focus  # 「只看重点」照旧只显示 ★★★
