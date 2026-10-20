import json

import pytest

from lecture_analyzer.asr.base import Segment
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.documents import DocPage
from lecture_analyzer.llm import ImagePart, TextPart
from lecture_analyzer.pipeline.extract import build_system_prompt, extract_lecture, render_transcript
from lecture_analyzer.prompts import load_prompt
from lecture_analyzer.pipeline.page_digest import collect_terms, digest_pages, needs_image
from lecture_analyzer.schemas import PageDigest

from .conftest import FakeProvider, make_profile

PNG = b"\x89PNG-fake"


def _page(no: int, text: str = "这一页的文字内容足够长，" * 10, has_images: bool = False) -> DocPage:
    return DocPage(no, text, PNG, has_images)


def test_needs_image_policies():
    plain = _page(1)
    assert needs_image(plain, "auto") is False
    assert needs_image(plain, "always") is True
    assert needs_image(_page(2, has_images=True), "auto") is True
    assert needs_image(_page(3, text="短"), "auto") is True
    assert needs_image(_page(4, text="公式 a=b，∑x，∫y dx，√z " * 5), "auto") is True
    assert needs_image(_page(5, has_images=True), "never") is False
    assert needs_image(DocPage(6, "", None, True), "always") is False


def _digest_json(page_nos, terms=("加速度",)):
    pages = [{"page_no": n, "title": f"第{n}页", "content_md": f"内容{n}", "terms": list(terms)} for n in page_nos]
    return json.dumps({"pages": pages}, ensure_ascii=False)


def test_digest_pages_batches_and_sends_images_selectively():
    pages = [_page(n, has_images=(n == 2)) for n in range(1, 8)]
    provider = FakeProvider([_digest_json(range(1, 7)), _digest_json([7])], make_profile(vision=True))
    digests, usage = digest_pages(provider, pages, batch_size=6)
    assert [d.page_no for d in digests] == list(range(1, 8))
    assert len(provider.calls) == 2 and usage.input_tokens == 20
    images = [p for p in provider.calls[0]["parts"] if isinstance(p, ImagePart)]
    assert len(images) == 1  # 只有第 2 页带图


def test_digest_pages_fills_missing_pages_from_text():
    pages = [_page(1), _page(2, text="第二页原文")]
    provider = FakeProvider([_digest_json([1])], make_profile(vision=True))
    digests, _ = digest_pages(provider, pages)
    assert digests[1] == PageDigest(page_no=2, title="", content_md="第二页原文", terms=[])


def test_collect_terms_dedupes_in_order():
    digests = [
        PageDigest(page_no=1, title="", content_md="", terms=["加速度", "力", " "]),
        PageDigest(page_no=2, title="", content_md="", terms=["力", "动量"]),
    ]
    assert collect_terms(digests) == ["加速度", "力", "动量"]


def test_render_transcript_numbers_from_one():
    text = render_transcript([Segment(0, 1000, "第一句"), Segment(61_000, 62_000, "둘째", "ko")])
    assert text == "[S1][00:00] 第一句\n[S2][01:01][ko] 둘째"


def test_system_prompt_follows_course_settings():
    chinese = build_system_prompt(CourseSettings(subject="药理学", teaching="zh", note_language="zh"))
    assert "药理学课程" in chinese and "中文授课" in chinese and "一律用中文书写" in chinese
    assert "现场翻译" not in chinese and "{{" not in chinese

    interpreted = build_system_prompt(CourseSettings(subject="病毒学", teaching="ko+zh", note_language="ko"))
    assert "一律用中文书写" in interpreted and "合并成一条" in interpreted  # 笔记一律中文
    assert "老师用韩语讲的术语要译成准确的中文" in interpreted
    assert "[ko] 句子是老师的韩语原话" in interpreted and "[zh] 句子通常是现场翻译成的中文" in interpreted

    english = build_system_prompt(CourseSettings(subject="Biology", teaching="en,fr", interpreted_to="ja", note_language="en"))
    assert "老师用英语、法语授课，现场有人翻译成日语" in english and "一律用中文书写" in english
    assert "[en]、[fr] 句子是老师的英语、法语原话" in english and "{{" not in english


def test_load_prompt_requires_all_placeholders():
    with pytest.raises(KeyError, match="subject"):
        load_prompt("page_digest")


def test_extract_lecture_builds_parts_and_clamps_strength():
    notes = {
        "title": "牛顿定律",
        "outline": ["回顾", "第二定律"],
        "knowledge_points": [
            {"name": "第二定律", "summary_md": "$F=ma$", "slide_pages": [1], "segment_ids": [2], "minutes": 12.5}
        ],
        "examples": [],
        "homework": ["习题 2.3"],
        "signals": [
            {
                "type": "exam_hint", "knowledge_point": "第二定律", "strength": 5,
                "quote": "这个期末必考", "segment_ids": [2], "slide_pages": [1],
            }
        ],
    }
    provider = FakeProvider([json.dumps(notes, ensure_ascii=False)])
    digest = PageDigest(page_no=1, title="牛顿第二定律", content_md="$F=ma$", terms=[])
    segments = [Segment(0, 1000, "上节课讲了"), Segment(1000, 2000, "这个期末必考")]
    result = extract_lecture(provider, segments, [digest], CourseSettings(subject="药理学"))
    assert result.output.signals[0].strength == 3
    texts = [p.text for p in provider.calls[0]["parts"] if isinstance(p, TextPart)]
    assert texts[0].startswith("## 课件\n[P1] 牛顿第二定律")
    assert "[S2][00:01] 这个期末必考" in texts[1]
    assert "quote 必须逐字照抄" in provider.calls[0]["system"]


def test_lecture_without_slides_drops_slide_signals():
    signal = {"knowledge_point": "第二定律", "strength": 1, "quote": "这个课件上没有", "segment_ids": [1],
              "slide_pages": []}
    notes = {"title": "牛顿定律", "outline": [], "knowledge_points": [], "examples": [], "homework": [],
             "signals": [{**signal, "type": kind} for kind in ("off_slide", "skipped_slide", "emphasis")]}
    provider = FakeProvider([json.dumps(notes, ensure_ascii=False)])
    result = extract_lecture(provider, [Segment(0, 1000, "这个课件上没有")], [], CourseSettings(subject="药理学"))
    assert [s.type for s in result.output.signals] == ["emphasis"]
    assert "不要输出 off_slide、skipped_slide" in provider.calls[0]["parts"][0].text
