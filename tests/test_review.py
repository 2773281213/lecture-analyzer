import json

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from lecture_analyzer.course import CourseSettings
from lecture_analyzer.knowledge import CourseMemory, HistorySignal, KnowledgePoint, KPAppearance
from lecture_analyzer.library import Course, Library
from lecture_analyzer.pipeline.integrate import integrate_through
from lecture_analyzer.pipeline.review import (
    ChapterInput, ReviewContext, build_system_prompt, generate_review, kp_priority, plan_review,
    render_review_markdown, resolve_review, review_book,
)
from lecture_analyzer.pipeline.run import LectureRunner
from lecture_analyzer.schemas import ReviewChapter
from lecture_analyzer.web import create_app

from .conftest import RoutingProvider, make_profile
from .test_integrate import RESULTS, _context, course_env  # noqa: F401 —— 复用整合测试的课程


def _signal(kind, strength=2, seq=1):
    return HistorySignal(key=1, lecture_id=1, seq=seq, type=kind, strength=strength, quote="q", segment_ids=[1])


def test_priority_rules():
    assert kp_priority([_signal("exam_hint", 1)], 1) == 3
    assert kp_priority([_signal("emphasis", 3)], 1) == 3
    assert kp_priority([_signal("emphasis", 2)], 1) == 2
    assert kp_priority([_signal("common_mistake", 1)], 1) == 2
    assert kp_priority([], 2) == 2  # 在两讲以上出现
    assert kp_priority([_signal("recap", 1)], 1) == 1
    assert kp_priority([_signal("scope_exclude", 3)], 3) == 0
    assert kp_priority([_signal("scope_exclude", 2)], 3) == 2  # 较弱的"不考"不整体排除
    assert kp_priority([_signal("scope_exclude", 3), _signal("exam_hint", 2)], 1) == 3


def _chapter(overview, entries):
    return json.dumps({"overview_md": overview, "entries": [
        {"kp": kp, "summary_md": summary, "sources": sources, "emphasis": emphasis, "examples": []}
        for kp, summary, sources, emphasis in entries
    ]}, ensure_ascii=False)


def _integrated(course_env):  # noqa: F811
    library, course, ids = course_env
    ctx, _ = _context(library, [RESULTS[1], RESULTS[2], RESULTS[3]])
    integrate_through(ctx, course, library.lecture_by_id(ids[3]))
    return library, course, ids


def test_generate_review_by_chapter_with_validated_references(course_env):  # noqa: F811
    library, course, ids = _integrated(course_env)
    planned = plan_review(library, course)
    assert [p.input.chapter for p in planned] == ["命名", "分类", "形态"] and all(p.needed for p in planned)
    responses = [
        _chapter("命名一章", [("K1", "按宿主命名", ["L1N1"], [])]),
        _chapter("分类一章", [("K2", "ICTV 按复制方式分类", ["L1N2", "L9N9"], ["Q1", "Q7"]), ("K9", "不在本章", [], [])]),
        _chapter("形态一章", [("K4", "衣壳对称", ["L3N2"], ["Q1"])]),
    ]
    provider = RoutingProvider({"ReviewChapter": responses}, make_profile())
    notes = generate_review(ReviewContext(library, provider), course, planned)
    assert notes[1] == "分类：2 个知识点（1 个由程序补全）"  # K3 卫星病毒模型漏写
    assert "[Q1] K2 · 强调·强度2 · 第1讲 00:00：「复制方式很重要」" in provider.calls[1]["parts"][0].text

    chapters = {c.chapter: c for c in review_book(library, course)}
    classify = chapters["分类"]
    assert classify.state == "ready" and classify.overview_md == "分类一章"
    k2, k3 = classify.entries
    assert (k2.name, k2.priority, k2.summary_md, k2.generated) == ("ICTV 分类", 2, "ICTV 按复制方式分类", True)
    assert [s["seq"] for s in k2.sources] == [1]  # 无效的 L9N9 被丢弃
    assert [q["quote"] for q in k2.quotes] == ["复制方式很重要", "昨天学过分类"]  # 引用的在前，其余按强度
    assert not k3.generated and "**第2讲**：卫星病毒的要点" in k3.summary_md  # 漏写的用各讲笔记原文
    assert chapters["形态"].entries[0].quotes[0]["quote"] == "衣壳一定要记住"

    markdown = render_review_markdown(course, list(chapters.values()))
    assert "### ICTV 分类 ★★" in markdown and "- 强调（第1讲 00:00）：「复制方式很重要」" in markdown

    # 材料没变：不需要重新生成；改了第 3 讲的笔记：只有「形态」一章待更新
    assert not any(p.needed for p in plan_review(library, course))
    notes3 = library.notes(ids[3])
    notes3.knowledge_points[1].summary_md = "衣壳对称：二十面体与螺旋"
    library.save_notes(ids[3], notes3, "fake", "fake-model")
    assert [p.input.chapter for p in plan_review(library, course) if p.needed] == ["形态"]
    assert {c.chapter: c.state for c in review_book(library, course)}["形态"] == "stale"
    with Library(library.path) as other:
        assert [row["step"] for row in other.cost_summary(course)] == ["integrate", "review"]


def test_review_text_has_no_citations_or_repeated_title():
    # M4 验收：要点里留着「（见 Q6、Q7）」「（Q7，重点）」这类编号（页面不显示编号），开头还重复知识点名称
    def kp(key, name):
        return KnowledgePoint(key=key, name=name, chapter="总论", aliases=(),
                              appearances=(KPAppearance(lecture_id=1, seq=1, note_index=key, role="first", minutes=5),))

    chapter = ChapterInput("总论", [kp(3, "基因组的极性（正链、负链）"), kp(4, "第V类：负极性单链RNA病毒的增殖策略"),
                                  kp(5, "药物作用机制"), kp(6, "噬菌体药物与噬菌体裂解酶")], [],
                           sources={"L1N3": (1, 3)}, quotes={1: 11, 2: 12}, examples={1: (1, 0)})
    entries = [
        ("K3", "## 基因组的极性\n\n- 正链（Q1，重点）与 mRNA 同向（见 Q1、Q2）\n- 双向性（具体操作，见例题 E1）；"
               "垂直传播（种子传毒也算，见 K3。）\n- 雌二醇（E2）、大肠杆菌（K12）、（见第 3 讲）、（L1N3 之外）照常保留"),
        ("K4", "**第V类：负极性单链RNA病毒**\n\n- 几乎都有包膜"),
        ("K5", "### 1. 崩解\n\n片剂先崩解"),
        ("K6", "**噬菌体药物（活菌药思路）**\n\n用噬菌体杀菌"),
    ]
    output = ReviewChapter.model_validate({"overview_md": "抓住极性（见 K3）。", "entries": [
        {"kp": key, "summary_md": md, "sources": ["L1N3"], "emphasis": ["Q1"], "examples": []} for key, md in entries]})
    stored = resolve_review(output, chapter)
    assert stored.overview_md == "抓住极性。"
    assert [entry.summary_md for entry in stored.entries] == [
        "- 正链（重点）与 mRNA 同向\n- 双向性（具体操作）；垂直传播（种子传毒也算）\n"
        "- 雌二醇（E2）、大肠杆菌（K12）、（见第 3 讲）、（L1N3 之外）照常保留",  # 没有这个编号的、不只是编号的保留
        "- 几乎都有包膜",
        "### 1. 崩解\n\n片剂先崩解",  # 小节标题不是知识点名称，保留
        "**噬菌体药物（活菌药思路）**\n\n用噬菌体杀菌",  # 知识点的前半部分，也是小节标题
    ]
    system = build_system_prompt(Course(id=1, name="病毒学", settings=CourseSettings(subject="病毒学"), asr=None))
    assert "也不要写 K、L、Q、E 编号" in system and "不要用知识点名称作开头的标题" in system


def test_review_progress_survives_rebuild_by_name(course_env):  # noqa: F811
    library, course, ids = _integrated(course_env)
    library.set_reviewed(course.id, "ictv分类", True)  # 网页提交的是归一化后的名称
    assert [e.reviewed for c in review_book(library, course) for e in c.entries if e.key == 2] == [True]
    memory = CourseMemory(library)
    memory.reset(course.id)
    for seq in (1, 2, 3):
        memory.apply(course, ids[seq], memory.log(ids[seq]))
    assert [e.reviewed for c in review_book(library, course) for e in c.entries if e.key == 2] == [True]


def test_review_page_and_command(m1_env, monkeypatch):
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        LectureRunner(m1_env.cfg, library, library.lecture_by_id(m1_env.lecture_id), report=lambda _: None).run()
    client = TestClient(create_app(m1_env.cfg), base_url="http://127.0.0.1")
    page = client.get(f"/courses/{m1_env.course_id}/review")
    assert page.status_code == 200 and "还没有生成复习讲义" in page.text
    assert "阿司匹林的剂量分段" in page.text and "这个期末一定会考" in page.text and "★★★" in page.text
    assert "阿司匹林的剂量分段" in client.get(f"/courses/{m1_env.course_id}/review?focus=1").text

    from lecture_analyzer import cli

    review_json = _chapter("药理学总论导语", [("K1", "**三段剂量**", ["L3N1"], ["Q1"])])
    monkeypatch.setattr("lecture_analyzer.pipeline.review.create_provider",  # 命令行与后台任务共用 review_context
                        lambda profile: RoutingProvider({"ReviewChapter": [review_json]}, profile))
    args = ["--config", str(m1_env.cfg.root / "config.toml")]
    result = CliRunner().invoke(cli.app, ["review", "药理学", *args])
    assert result.exit_code == 0, result.output
    assert "将调用模型生成 1 章" in result.output and "药理学总论：1 个知识点" in result.output
    assert "复习讲义已是最新" in CliRunner().invoke(cli.app, ["review", "药理学", *args]).output
    exported = (m1_env.cfg.data_dir / "books" / "药理学" / "总复习.md").read_text(encoding="utf-8")
    assert "药理学总论导语" in exported and "### 阿司匹林的剂量分段 ★★★" in exported
    page = client.get(f"/courses/{m1_env.course_id}/review").text
    assert "药理学总论导语" in page and "<strong>三段剂量</strong>" in page
