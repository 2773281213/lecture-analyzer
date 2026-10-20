"""外语授课的课程：用原文理解、用中文总结——名称和正文全中文（外语原词只进别名、不显示）、原话先显示中文意思、
现场翻译以原话为准、问答用中文回答。"""

import json
import sqlite3

from fastapi.testclient import TestClient

from lecture_analyzer.course import CourseSettings
from lecture_analyzer.knowledge import CourseMemory
from lecture_analyzer.library import Course, Library
from lecture_analyzer.library_schema import MIGRATIONS
from lecture_analyzer.pipeline import extract, integrate, qa, review
from lecture_analyzer.pipeline.report_render import evidence_text
from lecture_analyzer.knowledge import KnowledgePoint, KPAppearance
from lecture_analyzer.render import quote_html, zh_aliases
from lecture_analyzer.schemas import ReviewChapter
from lecture_analyzer.textnorm import strip_foreign_asides
from lecture_analyzer.schemas import LectureNotes, Signal
from lecture_analyzer.web import create_app

from .conftest import LECTURE_NOTES, RoutingProvider, integration_json
from .test_pipeline_run import _run
from .test_web import BASE

KOREAN = CourseSettings(subject="病毒学", teaching="ko", interpreted_to="zh", note_language="zh")
CHINESE = CourseSettings(subject="药理学", teaching="zh", note_language="zh")


def _course(settings: CourseSettings) -> Course:
    return Course(id=1, name="病毒学", settings=settings, asr=None)


def test_term_rule_is_all_chinese():
    # M4 修订：分析结果和知识点分析必须全中文，不在括号里附外语原词
    for settings in (KOREAN, CHINESE):
        assert "一律用中文书写" in settings.term_rule and "不要在括号里附任何外语原词" in settings.term_rule
        assert "老师讲课时直接说的英语术语也一样" in settings.term_rule  # M4 验收：笔记里仍括注老师说的英语术语
    assert KOREAN.foreign_languages == ["ko"] and "老师用韩语讲的术语要译成准确的中文" in KOREAN.term_rule
    assert CHINESE.foreign_languages == [] and "讲的术语要译成" not in CHINESE.term_rule
    assert "老师用韩语、英语讲的术语" in CourseSettings(teaching="ko,en").term_rule
    assert CourseSettings(teaching="zh-tw").foreign_languages == []  # 同一种语言的变体不算外语
    old_korean_notes = CourseSettings(teaching="ko", interpreted_to="zh", note_language="ko")
    assert old_korean_notes.note_language == "zh" and old_korean_notes.foreign_languages == ["ko"]


def test_aliases_and_quotes_are_shown_in_chinese():
    assert zh_aliases(["朊病毒", "프리온", "prion", "DNA 回旋酶"]) == ["朊病毒", "DNA 回旋酶"]  # 外语原词只在后台用
    assert quote_html("이건 기말에 꼭 나와요", "这个期末一定会考") == (
        '<span class="quote-meaning">这个期末一定会考</span>'
        '<details class="quote-original"><summary>原文</summary>이건 기말에 꼭 나와요</details>')
    assert quote_html("期末一定会考 <b>") == "「期末一定会考 &lt;b&gt;」"  # 原话本来就是中文时照常显示


def test_prompts_read_original_and_summarize_in_note_language():
    system = extract.build_system_prompt(KOREAN)
    assert "不要先翻译再整理" in system and "老师用韩语讲的术语要译成准确的中文" in system
    assert "以原话为准" in system and "不要只根据翻译整理" in system  # 现场翻译只作辅助
    assert "meaning：quote 不是中文时，用中文写出这句原话的意思" in system
    assert "讲的术语要译成" not in extract.build_system_prompt(CHINESE)
    for module in (integrate, review):
        assert "老师用韩语讲的术语要译成准确的中文" in module.build_system_prompt(_course(KOREAN))
        assert "讲的术语要译成" not in module.build_system_prompt(_course(CHINESE))
    assert "原文说法只放在这里" in integrate.build_system_prompt(_course(KOREAN))  # 外语原词放进别名


def test_foreign_asides_are_stripped():
    # M4 验收：收紧提示词后笔记里仍有「（virus）」「（두창）」这类旁注，保存前按规则清理
    cases = {
        "病毒（virus）是分子寄生物": "病毒是分子寄生物", "天花（두창）": "天花", "朊病毒 (Prion) 很特殊": "朊病毒 很特殊",
        "大肠杆菌（E. coli）": "大肠杆菌", "人类免疫缺陷病毒（Human Immunodeficiency Virus, HIV）": "人类免疫缺陷病毒",
        "DNA 聚合酶（DNA pol）": "DNA 聚合酶",
        # 中外文混排的括号去掉外语条目（M4 验收：清理后仍有「（又称痘疮、마마）」「（类似日食，称 eclipse）」）
        "天花病毒（又称痘疮、마마）": "天花病毒（又称痘疮）", "核衣壳（nucleocapsid，也有人称核蛋白）": "核衣壳（也有人称核蛋白）",
        "（类似日食，称 eclipse）": "（类似日食）", "病毒（英文：virus）": "病毒",
    }
    kept = ["丁型肝炎病毒（HDV）", "信使 RNA（mRNA）", "选项（A）", "（DNA RNA）", "T4 噬菌体（T4）", "（见第 3 讲）",
            "剂量（1,000 mg）", "（DNA，RNA）",
            r"平均浓度 $C_{ss}(avg)$ 与 $\sin(\theta)$", r"$$\ln(dose) = k$$ 和 `f(word)`", "见[链接](page)"]
    assert {text: strip_foreign_asides(text) for text in cases} == cases
    assert [strip_foreign_asides(text) for text in kept] == kept  # 缩写、选项、公式、代码、链接、中文括号不动


def test_saved_analysis_has_no_foreign_asides(m1_env):
    notes = json.loads(json.dumps(LECTURE_NOTES, ensure_ascii=False))
    notes["title"] = "量效关系（dose-response）"
    notes["knowledge_points"][0]["name"] = "阿司匹林（aspirin）的剂量分段"
    notes["knowledge_points"][0]["summary_md"] = "小于 $75$ mg：**抗血凝**（anticoagulation），$f(x)$ 段（DNA）"
    notes["signals"][0]["meaning"] = "这个期末一定会考（시험）"
    m1_env.routes["LectureNotes"] = [json.dumps(notes, ensure_ascii=False)]
    m1_env.routes["IntegrationResult"] = [integration_json(notes, chapter="药理学总论（general pharmacology）")]
    _run(m1_env)
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        saved = library.notes(m1_env.lecture_id)
        tree = CourseMemory(library).tree(library.course_by_id(m1_env.course_id))
    assert saved.title == "量效关系" and saved.knowledge_points[0].name == "阿司匹林的剂量分段"
    assert saved.knowledge_points[0].summary_md == "小于 $75$ mg：**抗血凝**，$f(x)$ 段（DNA）"
    assert (saved.signals[0].quote, saved.signals[0].meaning) == ("这个期末一定会考", "这个期末一定会考")  # 原话不动
    assert (tree[0].name, tree[0].chapter) == ("阿司匹林的剂量分段", "药理学总论")

    kp = KnowledgePoint(key=1, name="朊病毒", chapter="总论", aliases=("프리온",),
                        appearances=(KPAppearance(lecture_id=1, seq=1, note_index=1, role="first", minutes=5),))
    output = ReviewChapter.model_validate({"overview_md": "本章（chapter）导语", "entries": [
        {"kp": "K1", "summary_md": "朊病毒（prion）没有核酸", "sources": [], "emphasis": [], "examples": []}]})
    stored = review.resolve_review(output, review.ChapterInput("总论", [kp], []))
    assert (stored.overview_md, stored.entries[0].summary_md) == ("本章导语", "朊病毒没有核酸")


def test_signal_meaning_is_required_for_models_but_optional_in_old_notes():
    assert "meaning" in Signal.model_json_schema()["required"]  # 严格结构化输出时模型必须给出
    old = json.loads(json.dumps(LECTURE_NOTES))  # M4 之前保存的笔记没有 meaning
    assert all(s.meaning == "" for s in LectureNotes.model_validate(old).signals)


def test_meaning_is_stored_and_shown(m1_env):
    notes = json.loads(json.dumps(LECTURE_NOTES, ensure_ascii=False))
    notes["signals"][0]["meaning"] = "期末一定会考"  # 假设原话是外语，这里是笔记语言的意思
    m1_env.routes["LectureNotes"] = [json.dumps(notes, ensure_ascii=False)]
    m1_env.routes["IntegrationResult"] = [integration_json(notes, chapter="药理学总论")]
    _run(m1_env)
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        checked = library.signals(m1_env.lecture_id)
        history = CourseMemory(library).history_signals(library.course_by_id(m1_env.course_id))
    assert checked[0].signal.meaning == "期末一定会考" and checked[1].signal.meaning == ""
    assert history[0].meaning == "期末一定会考"
    client = TestClient(create_app(m1_env.cfg), base_url=BASE)
    page = client.get(f"/lectures/{m1_env.lecture_id}").text
    assert '<span class="quote-meaning">期末一定会考</span><details class="quote-original"><summary>原文</summary>' in page
    view = {"id": "Q1", "quote": "이건 기말에 꼭 나와요", "meaning": "这个期末一定会考", "label": "提示考试", "seq": 2,
            "t": 61000}
    assert evidence_text(view) == "「这个期末一定会考」（提示考试，第2讲 01:01）"  # 悬停提示只写中文意思


def test_v4_adds_meaning_column(tmp_path):
    path = tmp_path / "lecture.db"
    conn = sqlite3.connect(path)
    for version in (1, 2, 3):
        conn.executescript(MIGRATIONS[version])
    conn.close()
    with Library(path) as library:
        columns = {row["name"] for row in library.conn.execute("PRAGMA table_info(signal)")}
    assert "meaning" in columns


def test_qa_answers_in_note_language(m1_env, monkeypatch):
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        library.update_course(m1_env.course_id, teaching="ko", interpreted_to="zh", note_language="zh")
    _run(m1_env)
    routes = {"QARetrieval": [json.dumps({"knowledge_points": ["K1"], "lectures": []})],
              "QAAnswer": [json.dumps({"found": True, "answer_md": "三段剂量（three doses） [N1]",
                                       "supplement_md": "补充（extra）"}, ensure_ascii=False)]}
    created: list[RoutingProvider] = []

    def fake(profile):
        created.append(RoutingProvider(routes, profile.model_copy(update={"model": "fake"})))
        return created[-1]

    monkeypatch.setattr(qa, "create_provider", fake)
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        result = qa.answer_question(m1_env.cfg, library, course, "아스피린 용량은 어떻게 나뉘나요?", report=lambda _: None)
    assert (result["answer_md"], result["supplement_md"]) == ("三段剂量 [N1]", "补充")  # 外语旁注保存前去掉
    seen = {call["schema"].__name__: call["system"] for provider in created for call in provider.calls}
    assert "用中文回答（问题用其他语言提出时也一样" in seen["QAAnswer"]  # 用韩语提问也用中文回答
    assert "老师用韩语讲的术语要译成准确的中文" in seen["QAAnswer"] and "不要抄写原文" in seen["QAAnswer"]
