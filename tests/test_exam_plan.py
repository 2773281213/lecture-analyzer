"""考试方案：先弄清这门课怎么考（题型分值、答题要求、考试范围），再按它分析知识点。"""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from lecture_analyzer.asr.base import Segment
from lecture_analyzer.cli import app as cli_app
from lecture_analyzer.config import ScoringConfig
from lecture_analyzer.exam_plans import (
    QUESTION_TYPES, ExamPlan, ExamPlanStore, PlanEvidence, PlanQuestionType, normalize_type, plan_block,
)
from lecture_analyzer.job_handlers import run_job
from lecture_analyzer.job_runner import JobRunner
from lecture_analyzer.jobs import JobStore
from lecture_analyzer.library import Library
from lecture_analyzer.library_schema import MIGRATIONS
from lecture_analyzer.pipeline import exam_plan as exam_plan_module
from lecture_analyzer.pipeline import extract, review
from lecture_analyzer.pipeline.exam_plan import AwaitingExamPlan, to_plan, transcript_excerpt
from lecture_analyzer.pipeline.materials import ITEM_TYPES
from lecture_analyzer.pipeline.memory import merge_notes
from lecture_analyzer.pipeline.organize import move_lecture
from lecture_analyzer.pipeline.report import build_plan, effective_top_n, generate_report, save_report
from lecture_analyzer.pipeline.run import LectureRunner, RunOptions
from lecture_analyzer.pipeline.scoring import score_course, type_fit_parts
from lecture_analyzer.schemas import ExamPlanOutput, LectureNotes
from lecture_analyzer.web import create_app

from .conftest import LECTURE_NOTES, make_wav
from .test_report import REVIEW, _fake_models
from .test_web import BASE, _post

PLAN_OUTPUT = {
    "exam_form": "闭卷笔试", "duration_minutes": 120, "total_points": 100,
    "components": [{"name": "期末考试", "percent": 70, "source": "user", "quote": "期末占 70%", "segment_ids": []}],
    "question_types": [
        {"type": "名词解释", "count": 5, "points_each": 4, "total_points": 0, "note": "", "source": "user",
         "quote": "名词解释 5 题每题 4 分", "segment_ids": []},
        {"type": "简答题", "count": 4, "points_each": 10, "total_points": 0, "note": "", "source": "lecture",
         "quote": "这个期末一定会考", "segment_ids": [2]},
        {"type": "A1 型选择题", "count": 40, "points_each": 1, "total_points": 0, "note": "", "source": "lecture",
         "quote": "选择题四十道", "segment_ids": [1]},
    ],
    "answer_rules": [{"text": "简答题要写出作用机制和临床应用", "question_type": "简答题", "source": "user", "quote": "",
                      "segment_ids": []}],
    "scope": [{"kind": "exclude", "text": "药物经济学不考", "source": "user", "quote": "", "segment_ids": []}],
    "not_mentioned": ["考试范围"],
}
DB = "lecture.db"


def _db(env):
    return env.cfg.data_dir / DB


def _set(env, **fields):
    with Library(_db(env)) as library:
        return library.update_course(env.course_id, **fields)


def _course(env):
    with Library(_db(env)) as library:
        return library.course_by_id(env.course_id)


def _with_plan_route(env, notes=None):
    env.routes["ExamPlanOutput"] = [json.dumps(PLAN_OUTPUT, ensure_ascii=False)]
    if notes is not None:
        env.routes["LectureNotes"] = [json.dumps(notes, ensure_ascii=False)]


def _run(env, lecture_id=None, **options):
    messages: list[str] = []
    with Library(_db(env)) as library:
        lecture = library.lecture_by_id(lecture_id or env.lecture_id)
        LectureRunner(env.cfg, library, lecture, RunOptions(**options), report=messages.append).run()
        return library.lecture_by_id(lecture.id), messages


def _calls(env, schema):
    return [call for provider in env.provider_calls for call in provider.calls if call["schema"] is not None
            and call["schema"].__name__ == schema]


def _typed_notes():
    notes = json.loads(json.dumps(LECTURE_NOTES))
    notes["knowledge_points"][0]["question_types"] = ["简答题", "名词解释", "论述题"]  # 论述题不在方案里
    return notes


# ---------- 题型、合计与指纹 ----------


def test_question_types_follow_material_item_types():
    assert all(ITEM_TYPES[key] == label for key, label in QUESTION_TYPES.items())


@pytest.mark.parametrize("name,key", [
    ("名词解释", "term"), ("单项选择题", "choice"), ("A1 型题", "choice"), ("B型题", "choice"), ("问答题", "short"),
    ("简述题", "short"), ("病案分析", "case"), ("论述题", "essay"), ("是非题", "judge"), ("作文", "other"),
])
def test_normalize_type(name, key):
    assert normalize_type(name) == key


def test_totals_and_shares_are_computed_by_code_and_fingerprint_ignores_sources():
    plan = ExamPlan(question_types=[
        PlanQuestionType(name="名词解释", count=5, points_each=4, total_points=99),  # 有题数和分值时不用填写的合计
        PlanQuestionType(name="病例分析", total_points=30),
        PlanQuestionType(name="选择题", count=50),  # 没说分值
    ]).computed()
    assert [(t.key, t.total_points, t.share) for t in plan.question_types] == [
        ("term", 20, 0.4), ("case", 30, 0.6), ("choice", 0, 0)]
    moved = plan.model_copy(update={"question_types": [
        plan.question_types[0].model_copy(update={"evidence": PlanEvidence(source="lecture", quote="另一句")}),
        *plan.question_types[1:]]})
    assert moved.fingerprint() == plan.fingerprint()
    assert plan.match_type("名词解释").name == "名词解释" and plan.match_type("A2 型题").name == "选择题"
    assert plan.match_type("论述题") is None
    assert plan_block(None) == "" and plan_block(ExamPlan()) == ""


# ---------- 转写摘录与原话核对 ----------


def test_transcript_excerpt_keeps_the_start_and_keyword_windows():
    segments = [Segment(i * 30_000, i * 30_000 + 20_000, f"第{i}句讲药理") for i in range(240)]  # 两小时
    segments[180] = Segment(180 * 30_000, 180 * 30_000 + 20_000, "期末考试是闭卷，名词解释五道")
    items = transcript_excerpt(segments, lead_minutes=1, window_minutes=1)
    assert len(items) == 2
    lead, window = items
    assert "[S1]" in lead.full and "[S2]" in lead.full and "[S3]" not in lead.full
    assert "[S179]" in window.full and "[S181]" in window.full and "[S183]" in window.full
    assert "[S178]" not in window.full and window.full.endswith("……")
    assert lead.keep > window.keep > 0 and lead.short == "" == window.short  # 超出预算时先去掉关键词段
    assert transcript_excerpt([], 40, 2) == []


def test_classroom_quotes_are_checked_against_the_transcript(m1_env):
    from .conftest import LECTURE_SEGMENTS

    with Library(_db(m1_env)) as library:
        lecture = library.lecture_by_id(m1_env.lecture_id)
    output = ExamPlanOutput.model_validate(PLAN_OUTPUT)
    output.question_types[1].segment_ids = [3]  # 编号偏了一句：前后放宽一句仍核对得上
    output.answer_rules[0].source = "lecture"
    output.answer_rules[0].quote = "大家记住阿司匹林的三段剂量"
    output.answer_rules[0].segment_ids = [1]  # 编号错得太远：全文查找后改到原话实际所在的句子
    plan = to_plan(output, lecture, LECTURE_SEGMENTS)
    short, choice = plan.question_types[1].evidence, plan.question_types[2].evidence
    assert short.verified and short.seq == 3 and short.t == 1000
    assert not choice.verified and choice.quote == "选择题四十道"  # 编造的原话保留并标出，由用户判断
    rule = plan.answer_rules[0].evidence
    assert rule.verified and rule.segment_ids == [3] and rule.t == 2000
    assert plan.question_types[0].evidence.source == "user" and plan.question_types[0].evidence.verified


# ---------- 分析与关卡 ----------


def test_gate_analyzes_then_stops_before_extract_and_confirmed_plan_drives_notes(m1_env):
    _set(m1_env, exam_plan="unset")
    _with_plan_route(m1_env, _typed_notes())
    with pytest.raises(AwaitingExamPlan, match="请到课程页核对"):
        _run(m1_env)
    with Library(_db(m1_env)) as library:
        lecture = library.lecture_by_id(m1_env.lecture_id)
        course = library.course_by_id(m1_env.course_id)
        stored = ExamPlanStore(library).get(course)
        steps = {row["step"] for row in library.cost_summary(course)}
    assert lecture.step_state("transcribe") == "done" and lecture.step_state("extract") not in ("failed", "done")
    assert course.exam_plan == "draft" and "exam_plan" in steps
    assert [t.name for t in stored.plan.question_types] == ["名词解释", "简答题", "A1 型选择题"]
    assert [t.share for t in stored.plan.question_types] == [0.2, 0.4, 0.4]
    [call] = _calls(m1_env, "ExamPlanOutput")
    text = call["parts"][0].text
    assert "课堂转写摘录（第 3 讲）" in text and "这个期末一定会考" in text and "（没有提供）" in text
    assert not _calls(m1_env, "LectureNotes")  # 停在整理笔记之前

    with Library(_db(m1_env)) as library:
        ExamPlanStore(library).confirm(library.course_by_id(m1_env.course_id))
    lecture, _ = _run(m1_env)
    assert lecture.step_state("extract") == "done"
    system = _calls(m1_env, "LectureNotes")[-1]["system"]
    assert "本课程的考试方案" in system and "名词解释，5 题 × 4 分 = 20 分，占 20%" in system
    assert "［简答题］简答题要写出作用机制和临床应用" in system and "question_types 填它适合出成的题型" in system
    with Library(_db(m1_env)) as library:
        course = library.course_by_id(m1_env.course_id)
        scoring = score_course(library, course, ScoringConfig())
    [item] = scoring.ranked
    assert item.types == ["名词解释", "简答题"]  # 归到方案的题型，按方案顺序；论述题不在方案里
    [fit] = [part for part in item.parts if part.label.startswith("题型匹配")]
    assert fit.points == 3.6 and "占 60%" in fit.label  # 6 × (0.2 + 0.4)


def test_auto_confirm_continues_without_stopping(m1_env):
    _set(m1_env, exam_plan="unset")
    _with_plan_route(m1_env)
    m1_env.cfg.exam_plan.auto_confirm = True
    lecture, messages = _run(m1_env)
    assert lecture.step_state("book") == "done" and _course(m1_env).exam_plan == "confirmed"
    assert any("自动采用考试方案" in m for m in messages)


def test_source_lecture_must_be_transcribed_first(m1_env):
    _set(m1_env, exam_plan="unset")
    _with_plan_route(m1_env)
    with Library(_db(m1_env)) as library:
        library.add_lecture(library.course_by_id(m1_env.course_id), 1, [make_wav(m1_env.tmp_path / "第一讲.wav")])
    with pytest.raises(AwaitingExamPlan, match="第 1 讲还没有转写"):
        _run(m1_env)
    assert not _calls(m1_env, "ExamPlanOutput")

    with Library(_db(m1_env)) as library:  # 填了考试说明就可以不等第 1 讲：只用说明分析
        ExamPlanStore(library).save_inputs(library.course_by_id(m1_env.course_id), None, "闭卷，名词解释 5 题每题 4 分")
    with pytest.raises(AwaitingExamPlan, match="请到课程页核对"):
        _run(m1_env)
    text = _calls(m1_env, "ExamPlanOutput")[-1]["parts"][0].text
    assert "闭卷，名词解释 5 题每题 4 分" in text and "课堂转写摘录\n（没有）" in text


def test_prompts_are_unchanged_without_a_confirmed_plan(m1_env):
    course = _course(m1_env)
    settings = course.settings
    assert extract.build_system_prompt(settings) == extract.build_system_prompt(settings, "", "")
    assert "考试方案" not in extract.build_system_prompt(settings)
    assert review.build_system_prompt(course) == review.build_system_prompt(course, "")
    with Library(_db(m1_env)) as library:
        ExamPlanStore(library).save_draft(course, ExamPlan.model_validate({"exam_form": "闭卷"}), "m")
        draft = library.course_by_id(course.id)
        assert ExamPlanStore(library).active(draft) is None  # 草稿不进入提示词
        assert review.course_system_prompt(library, draft) == review.build_system_prompt(draft)


def test_old_notes_without_question_types_and_merged_windows():
    old = json.loads(json.dumps(LECTURE_NOTES))
    assert LectureNotes.model_validate(old).knowledge_points[0].question_types == []
    a, b = _typed_notes(), json.loads(json.dumps(LECTURE_NOTES))
    b["knowledge_points"][0]["question_types"] = ["选择题", "简答题"]
    merged = merge_notes([LectureNotes.model_validate(a), LectureNotes.model_validate(b)])
    assert merged.knowledge_points[0].question_types == ["简答题", "名词解释", "论述题", "选择题"]


def test_type_fit_parts():
    assert type_fit_parts(["简答题"], None, ScoringConfig()) == []
    plan = ExamPlan(question_types=[PlanQuestionType(name="简答题", count=4, points_each=10),
                                    PlanQuestionType(name="病例分析", total_points=60)]).computed()
    [part] = type_fit_parts(["简答题", "病例分析"], plan, ScoringConfig(question_type_fit=5))
    assert part.points == 5 and "占 100%" in part.label
    assert type_fit_parts(["简答题"], plan, ScoringConfig(question_type_fit=0)) == []


# ---------- 考前报告按题型分组 ----------


def test_report_is_grouped_by_question_type(m1_env, monkeypatch):
    _set(m1_env, exam_plan="unset")
    _with_plan_route(m1_env, _typed_notes())
    m1_env.cfg.exam_plan.auto_confirm = True
    _run(m1_env)
    ranking = {"ranking": [{"kp": "K1", "reason": "老师说期末一定会考", "question_types": ["简答", "论述题"],
                            "evidence": ["Q1"]}]}
    created = _fake_models(monkeypatch, {"KPRanking": [json.dumps(ranking, ensure_ascii=False)],
                                         "ReviewChapter": [json.dumps(REVIEW, ensure_ascii=False)]})
    with Library(_db(m1_env)) as library:
        course = library.course_by_id(m1_env.course_id)
        plan = build_plan(m1_env.cfg, library, course)
        result = generate_report(m1_env.cfg, library, plan, report=lambda _: None)
        report_id, path = save_report(library, m1_env.cfg.data_dir, plan, result)
    assert result["version"] == 3 and result["plan_form"] == "闭卷笔试"
    assert [row["name"] for row in result["plan_table"]] == ["名词解释", "简答题", "A1 型选择题"]
    assert result["points"][0]["types"] == ["简答题"] and result["points"][0]["group"] == "简答题"
    assert result["groups"] == [{"type": "名词解释", "ranks": []}, {"type": "简答题", "ranks": [1]},
                                {"type": "A1 型选择题", "ranks": []}]
    rank_calls = [c for p in created for c in p.calls if c["schema"].__name__ == "KPRanking"]
    assert "本课程的考试方案" in rank_calls[0]["system"] and "最可能考的 24 个" in rank_calls[0]["system"]
    assert "适合题型：名词解释、简答题" in rank_calls[0]["parts"][0].text
    review_calls = [c for p in created for c in p.calls if c["schema"].__name__ == "ReviewChapter"]
    assert "名词解释写成一两句" in review_calls[0]["system"] and "适合题型：" in review_calls[0]["parts"][0].text
    markdown = path.read_text(encoding="utf-8")
    assert "## 题型与分值（考试方案）：闭卷笔试" in markdown and "## 简答题（4 题 × 10 分 = 40 分，占 40%）" in markdown
    assert "### 1. 阿司匹林的剂量分段 ★★★" in markdown and "（这种题型没有排进前列的考点）" in markdown
    page = TestClient(create_app(m1_env.cfg), base_url=BASE).get(f"/reports/{report_id}").text
    assert "题型与分值（考试方案）" in page and "简答题要写出作用机制和临床应用" in page
    assert page.index('id="g-2"') < page.index('id="p-1"') and "这种题型没有排进前列的考点" in page
    assert (path.parent / "考试方案.md").read_text(encoding="utf-8").startswith("# 药理学 · 考试方案（已确认）")


def test_effective_top_n():
    assert effective_top_n(15, None) == 15
    plan = ExamPlan(question_types=[PlanQuestionType(name="名词解释", count=5), PlanQuestionType(name="选择题", count=60),
                                    PlanQuestionType(name="病例分析")])
    assert effective_top_n(15, plan) == 8 + 10 + 3  # 题数 × 1.5、每种最多 10 个、没写题数算 3 个
    assert effective_top_n(30, plan) == 30


# ---------- 网页 ----------


def _client(env):
    app = create_app(env.cfg)
    app.state.jobs = JobRunner(env.cfg, _db(env), command=lambda job_id: ["unused"])  # 只提交、不执行
    return TestClient(app, base_url=BASE, follow_redirects=False)


def _plan_form(**extra):
    form = {
        "exam_form": "闭卷", "duration_minutes": "120", "total_points": "",
        "qt_ref": ["0", "1", ""], "qt_name": ["名词解释", "", "病例分析"], "qt_key": ["term", "short", "other"],
        "qt_count": ["6", "4", ""], "qt_points": ["4", "10", ""], "qt_total": ["", "", "30"], "qt_note": ["", "", ""],
        "rule_ref": ["0", ""], "rule_type": ["简答题", ""], "rule_text": ["简答题要写出作用机制和临床应用", "用规范术语"],
        "scope_ref": [""], "scope_kind": ["focus"], "scope_text": ["第三章是重点"],
        "comp_ref": [""], "comp_name": [""], "comp_percent": [""],
    }
    return form | extra


def test_course_page_card_save_confirm_and_skip(m1_env):
    _run(m1_env)  # 先按没有方案整理过一讲
    _set(m1_env, exam_plan="unset")
    client = _client(m1_env)
    page = client.get(f"/courses/{m1_env.course_id}").text
    assert 'id="exam-plan"' in page and "分析考试方案" in page and "不分析，直接填写" in page

    analyzed = _post(client, f"/courses/{m1_env.course_id}/exam-plan/analyze", lecture=str(m1_env.lecture_id),
                     user_text="闭卷，名词解释 5 题")
    with Library(_db(m1_env)) as library:
        job = JobStore(library).latest(1)[0]
        stored = ExamPlanStore(library).get(library.course_by_id(m1_env.course_id))
    assert analyzed.headers["location"] == f"/jobs/{job.id}" and job.kind == "exam_plan"
    assert job.params == {"lecture": m1_env.lecture_id} and stored.user_text == "闭卷，名词解释 5 题"

    with Library(_db(m1_env)) as library:  # 模拟分析结果
        course = library.course_by_id(m1_env.course_id)
        ExamPlanStore(library).save_draft(course, to_plan(ExamPlanOutput.model_validate(PLAN_OUTPUT),
                                                          library.lecture_by_id(m1_env.lecture_id),
                                                          library.segments(m1_env.lecture_id)).computed(), "m")
    page = client.get(f"/courses/{m1_env.course_id}").text
    assert "考试方案已分析好" in page and "原话未核对上" in page and "保存并确认" in page

    saved = _post(client, f"/courses/{m1_env.course_id}/exam-plan/save", action="save", **_plan_form())
    assert saved.headers["location"] == f"/courses/{m1_env.course_id}#exam-plan"
    with Library(_db(m1_env)) as library:
        course = library.course_by_id(m1_env.course_id)
        plan = ExamPlanStore(library).get(course).plan
    assert course.exam_plan == "draft"
    assert [(t.name, t.key, t.total_points) for t in plan.question_types] == [
        ("名词解释", "term", 24), ("简答题", "short", 40), ("病例分析", "case", 30)]
    assert plan.question_types[0].evidence.source == "user"  # 改了题数：记为用户填写
    assert plan.question_types[1].evidence.source == "lecture"  # 名称清空时按归类补上，其余没改：保留课堂出处
    assert plan.answer_rules[0].evidence.source == "user" and [s.kind for s in plan.scope] == ["focus"]
    assert plan.not_mentioned == []  # 填了考试范围：不再算"没有提到"

    confirmed = _post(client, f"/courses/{m1_env.course_id}/exam-plan/save", action="confirm", **_plan_form())
    with Library(_db(m1_env)) as library:
        job = JobStore(library).latest(1)[0]
    assert confirmed.headers["location"] == f"/jobs/{job.id}"
    assert job.kind == "run" and job.params == {"force": ["extract"], "lectures": []}  # 已整理过的讲按方案重新整理
    assert _course(m1_env).exam_plan == "confirmed"

    again = _post(client, f"/courses/{m1_env.course_id}/exam-plan/save", action="confirm", **_plan_form())
    assert again.headers["location"] == f"/courses/{m1_env.course_id}#exam-plan"  # 内容没变：不重新处理
    with Library(_db(m1_env)) as library:
        assert JobStore(library).latest(1)[0].id == job.id
    assert "内容没有变化" in client.get(f"/courses/{m1_env.course_id}").text

    assert _post(client, f"/courses/{m1_env.course_id}/exam-plan/save", action="save",
                 **_plan_form(qt_count=["x", "", ""])).status_code == 400
    skipped = _post(client, f"/courses/{m1_env.course_id}/exam-plan/skip")
    assert skipped.status_code == 303 and _course(m1_env).exam_plan == "skipped"


def test_run_job_waits_for_the_plan(m1_env):
    _set(m1_env, exam_plan="unset")
    _with_plan_route(m1_env)
    with Library(_db(m1_env)) as library:
        job = JobStore(library).add(library.course_by_id(m1_env.course_id), "run", {"lectures": [], "force": []})
    run_job(m1_env.cfg, _db(m1_env), job.id)
    with Library(_db(m1_env)) as library:
        done = JobStore(library).get(job.id)
    assert done.state == "done" and done.result["awaiting_exam_plan"]
    assert done.result["url"] == f"/courses/{m1_env.course_id}#exam-plan" and "请到课程页核对" in done.result["summary"]


# ---------- 迁移、命令行、移动讲 ----------


def test_migration_from_v4_adds_the_plan(tmp_path):
    path = tmp_path / DB
    conn = sqlite3.connect(path)
    for version in range(1, 5):
        conn.executescript(MIGRATIONS[version])
    conn.execute("INSERT INTO course (name, subject, teaching, note_language, created_at) VALUES ('病毒学', '病毒学',"
                 " 'zh', 'zh', '2026-10-01')")
    conn.commit()
    conn.close()
    with Library(path) as library:
        [course] = library.courses()
        assert course.exam_plan == "unset" and ExamPlanStore(library).get(course) is None
    assert (tmp_path / f"{DB}.bak-v4").is_file()


def test_cli_plan_commands(m1_env, monkeypatch):
    runner = CliRunner()
    config = ["--config", str(m1_env.cfg.source)]
    _set(m1_env, exam_plan="unset")
    _with_plan_route(m1_env)
    with Library(_db(m1_env)) as library:  # 先转写（停在整理笔记之前）
        try:
            LectureRunner(m1_env.cfg, library, library.lecture_by_id(m1_env.lecture_id), report=lambda _: None).run()
        except AwaitingExamPlan:
            pass
    shown = runner.invoke(cli_app, ["plan", "show", "药理学", *config])
    assert shown.exit_code == 0 and "待确认" in shown.output and "名词解释" in shown.output
    assert "原话未核对上" in shown.output
    run = runner.invoke(cli_app, ["run", "药理学", *config])
    assert run.exit_code == 0 and "请到课程页核对" in run.output and "lec plan confirm" in run.output
    confirmed = runner.invoke(cli_app, ["plan", "confirm", "药理学", *config])
    assert confirmed.exit_code == 0 and "已确认" in confirmed.output and _course(m1_env).exam_plan == "confirmed"
    skipped = runner.invoke(cli_app, ["plan", "skip", "药理学", *config])
    assert skipped.exit_code == 0 and _course(m1_env).exam_plan == "skipped"


def test_moving_a_lecture_to_a_course_with_another_plan_rewrites_notes(m1_env):
    from lecture_analyzer.course import CourseSettings

    _run(m1_env)
    with Library(_db(m1_env)) as library:
        target = library.add_course("药理学（二）", CourseSettings(subject="药理学"))
        target = library.update_course(target.id, slides="none")
        store = ExamPlanStore(library)
        store.save_draft(target, ExamPlan(exam_form="开卷"), "m")
        target, _ = store.confirm(library.course_by_id(target.id))
        result = move_lecture(library, m1_env.cfg.data_dir, library.lecture_by_id(m1_env.lecture_id), target)
        moved = library.lecture_by_id(m1_env.lecture_id)
    assert result.rerun_from == "extract" and moved.step_state("extract") != "done"
