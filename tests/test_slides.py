import json

import pytest

from lecture_analyzer.asr.base import Segment
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.library import Library, LibraryError, PageRef
from lecture_analyzer.materials import Material, MaterialStore
from lecture_analyzer.pipeline.run import LectureRunner, RunOptions
from lecture_analyzer.pipeline.slides import (
    MatchFile, course_terms, describe_pages, fill_gaps, match_lecture, page_spans, parse_page_spec,
    relevant_files, render_toc, resolve_ranges, sample_transcript,
)
from lecture_analyzer.schemas import PageDigest, PageRange

from .conftest import FakeProvider, make_profile


def _material(material_id: int, title: str, pages: int, kind: str = "slides") -> Material:
    return Material(id=material_id, course_id=1, kind=kind, title=title, path="", source="", position=material_id,
                    page_count=pages)


def _digests(count: int, terms: tuple[str, ...] = ()) -> list[PageDigest]:
    return [PageDigest(page_no=n, title=f"标题{n}", content_md=f"内容{n}", terms=list(terms)) for n in range(1, count + 1)]


FILES = [MatchFile(1, _material(11, "ch1-介绍", 21), _digests(21)), MatchFile(2, _material(12, "ch2-分类", 27), _digests(27))]


def test_resolve_ranges_validates_and_fills_gaps():
    ranges = [PageRange(file=1, start=5, end=3), PageRange(file=1, start=7, end=8), PageRange(file=1, start=19, end=40),
              PageRange(file=2, start=1, end=1), PageRange(file=9, start=1, end=2), PageRange(file=2, start=30, end=31)]
    refs = resolve_ranges(ranges, FILES)
    by_material = {mid: [r.page_no for r in refs if r.material_id == mid] for mid in (11, 12)}
    assert by_material[11] == [3, 4, 5, 6, 7, 8, 19, 20, 21]  # 5→7 相隔 1 页补上；8→19 太远不补；超出的页截掉
    assert by_material[12] == [1]  # 不存在的文件和超出范围的区间丢弃


def test_fill_gaps_and_page_spans():
    assert fill_gaps([1, 2, 6, 11]) == [1, 2, 3, 4, 5, 6, 11]
    assert page_spans([5, 1, 2, 3, 8, 9]) == "1–3、5、8–9"
    assert describe_pages([PageRef(11, 1), PageRef(11, 2), PageRef(12, 4)], {11: "ch1", 12: "ch2"}) == \
        "「ch1」第 1–2 页；「ch2」第 4 页"


def test_parse_page_spec():
    materials = [_material(3, "ch1", 21), _material(4, "ch2", 27)]
    assert parse_page_spec("3:2-3, 4", materials)[:3] == [PageRef(3, 2), PageRef(3, 3), PageRef(4, 1)]
    assert len(parse_page_spec("4", materials)) == 27
    for bad, message in (("9:1-2", "不存在"), ("3:5-40", "超出范围"), ("3:a-b", "格式不对")):
        with pytest.raises(ValueError, match=message):
            parse_page_spec(bad, materials)


def test_toc_sample_and_term_prefilter():
    toc = render_toc(FILES)
    assert "## [F1] ch1-介绍（课件，共 21 页）" in toc and "[F2-27] 标题27" in toc
    segments = [Segment(i * 1000, i * 1000 + 900, "句子" * 20) for i in range(1000)]
    sample = sample_transcript(segments, limit=2000)
    assert len(sample) <= 2600 and sample.startswith("[00:00]")
    files = [MatchFile(1, _material(1, "a", 1), _digests(1, ("干扰素",))),
             MatchFile(2, _material(2, "b", 1), _digests(1, ("逆转录酶", "整合酶"))),
             MatchFile(3, _material(3, "c", 1), _digests(1, ("抗体",)))]
    spoken = [Segment(0, 1000, "今天讲逆转录酶和整合酶，还有抗体")]
    assert [f.number for f in relevant_files(files, spoken, keep=2)] == [2, 3]


def test_match_lecture_uses_model_ranges():
    provider = FakeProvider([json.dumps({"ranges": [{"file": 2, "start": 3, "end": 5}], "note": "讲分类"})])
    result = match_lecture(provider, FILES, [Segment(0, 1000, "바이러스 분류")], CourseSettings(teaching="ko+zh"))
    assert result.refs == [PageRef(12, 3), PageRef(12, 4), PageRef(12, 5)] and result.note == "讲分类"
    call = provider.calls[0]
    assert "老师用韩语授课" in call["system"] and "[F1-1] 标题1" in call["parts"][0].text


# ---------- 与单讲流程的衔接 ----------


def _add_slides(env, pages: int = 4, digested: bool = True) -> int:
    with Library(env.cfg.data_dir / "lecture.db") as library:
        store = MaterialStore(library)
        material = store.add(library.course_by_id(env.course_id), "slides", "第二章 药动学", "第二章.pdf")
        store.set_file(material.id, f"materials/{material.id}/x.pdf", pages)
        if digested:
            _save_digests(store, material.id, pages)
        return material.id


def _save_digests(store: MaterialStore, material_id: int, pages: int) -> None:
    digests = [PageDigest(page_no=n, title=f"药动学{n}", content_md=f"第{n}页内容", terms=["阿司匹林", f"术语{n}"])
               for n in range(1, pages + 1)]
    store.save_pages(material_id, digests)
    store.set_step(material_id, "digest", "done")


def _run(env, **options):
    messages: list[str] = []
    with Library(env.cfg.data_dir / "lecture.db") as library:
        LectureRunner(env.cfg, library, library.lecture_by_id(env.lecture_id), RunOptions(**options),
                      report=messages.append).run()
        return library.lecture_by_id(env.lecture_id), messages


def _extract_calls(env):
    return [p for p in env.provider_calls if p.calls and p.calls[0]["schema"].__name__ == "LectureNotes"]


def test_slides_gate_blocks_until_choice(m1_env):
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        library.update_course(m1_env.course_id, slides="unset")
    with pytest.raises(LibraryError, match="是否提供课件"):
        _run(m1_env)


def test_matched_pages_feed_extraction_and_rematch_without_change(m1_env):
    m1_env.routes["SlideMatch"] = [json.dumps({"ranges": [{"file": 1, "start": 2, "end": 3}], "note": "讲药动学"})]
    material_id = _add_slides(m1_env)
    lecture, _ = _run(m1_env)
    assert "「第二章 药动学」第 2–3 页" in lecture.status["slides"]["detail"]
    slides_part = _extract_calls(m1_env)[0].calls[0]["parts"][0].text
    assert "[P1]（第二章 药动学 第 2 页） 药动学2" in slides_part and "[P2]（第二章 药动学 第 3 页）" in slides_part
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        assert library.note_pages(lecture.id) == [PageRef(material_id, 2), PageRef(material_id, 3)]
        assert [row["step"] for row in library.cost_summary()] == ["extract", "integrate", "match"]

    with Library(m1_env.cfg.data_dir / "lecture.db") as library:  # 又导入了别的课件：重新匹配，结果没变
        library.reset_steps(lecture.id, ["slides"])
    lecture, messages = _run(m1_env)
    assert "不需要重新整理笔记" in lecture.status["slides"]["detail"]
    assert len(_extract_calls(m1_env)) == 1 and any(m.startswith("跳过「整理笔记」") for m in messages)


def test_manual_pages_skip_matching(m1_env):
    material_id = _add_slides(m1_env, pages=6)
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        MaterialStore(library).set_lecture_pages(m1_env.lecture_id, [PageRef(material_id, 5)], "manual")
    lecture, _ = _run(m1_env)
    assert lecture.status["slides"]["detail"].startswith("手动指定：「第二章 药动学」第 5 页")
    assert not [p for p in m1_env.provider_calls if p.calls and p.calls[0]["schema"].__name__ == "SlideMatch"]


def test_manual_pages_are_digested_before_extraction(m1_env, monkeypatch):
    """手动指定（或添加讲时附带）的课件还没识别时，先识别再整理笔记。"""
    from lecture_analyzer.pipeline import run as run_module

    material_id = _add_slides(m1_env, pages=3, digested=False)
    seen: list[list[int]] = []

    def fake_digest(cfg, library, materials, report):
        seen.append([m.id for m in materials])
        for material in materials:
            _save_digests(MaterialStore(library), material.id, material.page_count)
        return list(materials)

    monkeypatch.setattr(run_module, "digest_materials", fake_digest)
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        MaterialStore(library).set_lecture_pages(m1_env.lecture_id, [PageRef(material_id, 2)], "manual")
    _run(m1_env)
    assert seen == [[material_id]]
    slides_part = _extract_calls(m1_env)[0].calls[0]["parts"][0].text
    assert "[P1]（第二章 药动学 第 2 页） 药动学2" in slides_part


def test_hotword_order_does_not_depend_on_the_process(m1_env):
    # M4 验收：同一页的术语按集合遍历，顺序随进程变化，转写缓存每次都被当成热词变了而作废
    terms = [f"术语{n}" for n in range(30)]
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        store = MaterialStore(library)
        material = store.add(library.course_by_id(m1_env.course_id), "slides", "第一章", "第一章.pdf")
        store.set_file(material.id, f"materials/{material.id}/x.pdf", 1)
        store.save_pages(material.id, [PageDigest(page_no=1, title="", content_md="", terms=[*terms, " 术语3 "])])
        assert course_terms(library, library.course_by_id(m1_env.course_id)) == terms


def test_transcription_hotwords_come_from_course_slides(m1_env, monkeypatch):
    _add_slides(m1_env, pages=3)
    m1_env.routes["SlideMatch"] = [json.dumps({"ranges": [], "note": "对不上"})]
    seen: list[list[str]] = []
    original = m1_env.asr.transcribe
    monkeypatch.setattr(m1_env.asr, "transcribe", lambda wav, hotwords=(): seen.append(list(hotwords)) or original(wav))
    _run(m1_env)
    assert seen[0][0] == "阿司匹林"  # 出现在每一页的术语排在最前
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        assert course_terms(library, library.course_by_id(m1_env.course_id), limit=2) == ["阿司匹林", "术语1"]
