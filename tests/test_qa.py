"""问答：检索相关知识点 → 代码整理证据（含笔记要点 N、转写片段 T）→ 回答，以及作为后台任务连续追问。"""

import json

import pytest

from lecture_analyzer.asr.base import Segment
from lecture_analyzer.conversations import QAStore, QATurn
from lecture_analyzer.course import CourseSettings
from lecture_analyzer.job_handlers import run_job
from lecture_analyzer.jobs import JobStore
from lecture_analyzer.library import Library, LibraryError
from lecture_analyzer.pipeline import qa as qa_module
from lecture_analyzer.pipeline.qa import answer_question, clean_citations, transcript_excerpt

from .conftest import RoutingProvider
from .test_pipeline_run import _run


def _fake_models(monkeypatch, retrieval: dict, answer: dict) -> list[RoutingProvider]:
    routes = {"QARetrieval": [json.dumps(retrieval, ensure_ascii=False)],
              "QAAnswer": [json.dumps(answer, ensure_ascii=False)]}
    created: list[RoutingProvider] = []

    def fake(profile):
        provider = RoutingProvider(routes, profile.model_copy(update={"model": f"fake-{profile.name}"}))
        created.append(provider)
        return provider

    monkeypatch.setattr(qa_module, "create_provider", fake)
    return created


def _inputs(created: list[RoutingProvider]) -> tuple[str, str]:
    """（检索的输入，回答的输入）。"""
    calls = [call for provider in created for call in provider.calls]
    by_schema = {call["schema"].__name__: call["parts"][0].text for call in calls}
    return by_schema["QARetrieval"], by_schema["QAAnswer"]


def test_answer_with_evidence_and_citations(m1_env, monkeypatch):
    _run(m1_env)
    created = _fake_models(
        monkeypatch, {"knowledge_points": ["K1", "K99", "k1"], "lectures": [3, 9]},
        {"found": True, "answer_md": "小于 75 mg 抗血凝 [N1][T1, Q1]，老师说期末会考 [K1][X9]",
         "supplement_md": "大剂量还可以解热镇痛 [N1]"},
    )
    history = [QATurn(1, m1_env.course_id, None, 1, "阿司匹林是什么药？", {"answer_md": "解热镇痛药", "kps": [
        {"key": 1, "name": "阿司匹林的剂量分段"}]}, ["fake"], "2026-10-15T10:00:00")]
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        result = answer_question(m1_env.cfg, library, course, "  三段剂量分别是什么？ ", None, history,
                                 report=lambda _: None)
        costs = {row["step"] for row in library.cost_summary(course)}
    assert result["answer_md"] == "小于 75 mg 抗血凝 [N1][T1][Q1]，老师说期末会考"  # 不存在的编号去掉
    assert result["supplement_md"] == "大剂量还可以解热镇痛"  # 补充部分不写证据编号
    assert result["found"] and result["kps"] == [{"key": 1, "name": "阿司匹林的剂量分段"}]
    assert result["lectures"] == [3] and result["scope"] is None and result["question"] == "三段剂量分别是什么？"
    assert set(result["evidence"]) == {"N1", "T1", "Q1"}  # 只保留回答引用的证据
    note, excerpt, quote = (result["evidence"][key] for key in ("N1", "T1", "Q1"))
    assert note["lecture_id"] == m1_env.lecture_id and "抗血凝" in note["summary_md"] and note["t"] == 2000
    assert excerpt["text"] == "这个期末一定会考， 大家记住阿司匹林的三段剂量。" and excerpt["t"] == 1000
    assert quote["quote"] == "这个期末一定会考"
    models = {"retrieve": m1_env.cfg.profile_for_step("qa_retrieve").model, "answer": "claude-opus-5-5"}
    assert result["models"] == models  # 记录配置里的型号（与其他步骤一致）
    assert {"qa_retrieve", "qa"} <= costs

    retrieval_input, answer_input = _inputs(created)
    assert "[K1] 阿司匹林的剂量分段｜药理学总论｜第 3 讲" in retrieval_input
    assert "涉及知识点：K1" in retrieval_input and "整门课（共 1 讲）" in retrieval_input
    assert "[T1] K1 · 第3讲 00:01：这个期末一定会考" in answer_input
    assert "「这个期末一定会考」" in answer_input and "编造的一句话" not in answer_input  # 只用校验通过的原话
    assert "第 3 讲「量效关系」：回顾受体；阿司匹林的三段剂量" in answer_input  # 针对整讲提问时带上大纲


def test_lecture_scope_and_nothing_found(m1_env, monkeypatch):
    _run(m1_env)
    created = _fake_models(monkeypatch, {"knowledge_points": [], "lectures": []},
                           {"found": False, "answer_md": "课程内容里没有提到。", "supplement_md": ""})
    with Library(m1_env.cfg.data_dir / "lecture.db") as library:
        course = library.course_by_id(m1_env.course_id)
        lecture = library.lecture_by_id(m1_env.lecture_id)
        result = answer_question(m1_env.cfg, library, course, "期中考试在哪天？", lecture, report=lambda _: None)
        assert result["scope"] == {"lecture_id": lecture.id, "seq": 3}
        assert not result["found"] and result["evidence"] == {} and result["kps"] == []
        _, answer_input = _inputs(created)
        assert "只限第 3 讲「量效关系」" in answer_input and "（没有找到相关的知识点）" in answer_input

        other = library.add_course("病毒学", CourseSettings())
        with pytest.raises(LibraryError, match="不属于"):
            answer_question(m1_env.cfg, library, other, "问题", lecture, report=lambda _: None)
        with pytest.raises(LibraryError, match="还没有整合进课程记忆"):
            answer_question(m1_env.cfg, library, other, "问题", report=lambda _: None)
        with pytest.raises(LibraryError, match="不能为空"):
            answer_question(m1_env.cfg, library, course, "   ", report=lambda _: None)


def test_qa_jobs_continue_a_thread(m1_env, monkeypatch):
    _run(m1_env)
    created = _fake_models(monkeypatch, {"knowledge_points": ["K1"], "lectures": []},
                           {"found": True, "answer_md": "三段剂量 [N1]", "supplement_md": ""})
    db = m1_env.cfg.data_dir / "lecture.db"
    for question in ("阿司匹林的剂量怎么分？", "那大剂量呢？"):
        with Library(db) as library:
            course = library.course_by_id(m1_env.course_id)
            job = JobStore(library).add(course, "qa", {"thread": 1, "question": question,
                                                       "lecture_id": m1_env.lecture_id})
        run_job(m1_env.cfg, db, job.id)
    with Library(db) as library:
        course = library.course_by_id(m1_env.course_id)
        finished = JobStore(library).get(job.id)
        turns = QAStore(library).thread(course, 1)
    assert finished.state == "done", finished.error
    assert finished.result["url"] == f"/courses/{m1_env.course_id}/qa?thread=1#qa-{turns[-1].id}"
    assert [t.question for t in turns] == ["阿司匹林的剂量怎么分？", "那大剂量呢？"]
    assert turns[1].lecture_id == m1_env.lecture_id and turns[1].models[1] == "claude-opus-5-5"
    follow_up_retrieval = [call["parts"][0].text for provider in created for call in provider.calls
                           if call["schema"].__name__ == "QARetrieval"][-1]
    assert "问：阿司匹林的剂量怎么分？" in follow_up_retrieval  # 追问时带上之前的对话


def test_clean_citations_and_excerpt():
    text, cited = clean_citations("见 [n1]、[T2，Q3] 和 [K4] [M 5][x]", {"N1", "T2", "M5"})
    assert text == "见 [N1]、[T2] 和  [M5][x]" and cited == ["N1", "T2", "M5"]
    segments = [Segment(i * 1000, i * 1000 + 900, f"第{i}句话。") for i in range(10)]
    excerpt, start = transcript_excerpt(segments, 5, limit=12)
    assert excerpt == "第3句话。 第4句话。" and start == 3000  # 从前一句开始，到字数上限为止
    assert transcript_excerpt(segments, 1, limit=3)[0] == "第0句话。"  # 至少取一句
