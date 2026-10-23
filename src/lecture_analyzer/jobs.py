"""后台任务表：在书架里处理讲、导入和识别资料、重建课程记忆、下载本地模型、生成考前报告和总复习、问答。

- 任务记在 job 表里（类型、参数、状态、进度说明、结果、错误），网页轮询 /api/jobs/{编号} 显示进度；
- 每个任务在独立的子进程里执行（执行器见 job_runner.py，各类任务的处理见 job_handlers.py）；
- 两条队列：重任务（转写占显卡、大量调用模型）一个个排队执行；轻任务（报告、总复习、问答）另开一条，
  不会被正在跑的转写挡住；
- 状态按"排队中 → 进行中 → 完成 / 失败 / 已取消"单向变化：取消之后子进程再写回的进度和结果都不生效；
- 暂停：排队中或进行中 → 已暂停（进行中的子进程被结束，之后写回的进度和结果同样不生效）；
  继续：已暂停 → 排队中，按原来的编号排队，已完成的步骤、已识别的页不重做；
- 重试：失败或已取消的任务按同样的参数重新提交一个新任务（旧任务留在列表里，看得出失败原因）；
- 应用启动时，上次没执行完的任务（应用被关闭）标为失败；已暂停的任务保持暂停。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .library import Course, Library, LibraryError


@dataclass(frozen=True)
class JobKind:
    label: str
    lane: str  # heavy：占显卡或大量调用模型，排队一个个执行；light：报告、问答，另开一条队列


JOB_KINDS: dict[str, JobKind] = {
    "run": JobKind("处理讲", "heavy"),
    "rebuild": JobKind("重建课程记忆", "heavy"),
    "exam_plan": JobKind("分析考试方案", "heavy"),  # 与处理讲排在同一队列：要读处理讲刚写好的转写
    "import_material": JobKind("导入资料", "heavy"),
    "process_material": JobKind("处理资料", "heavy"),
    "download_models": JobKind("下载本地模型", "heavy"),
    "report": JobKind("生成考前报告", "light"),
    "review": JobKind("生成 / 更新总复习", "light"),
    "qa": JobKind("问答", "light"),
}
LANES = ("heavy", "light")
ACTIVE_STATES = ("queued", "running")
FINISHED_STATES = ("done", "failed", "cancelled")
RETRYABLE_STATES = ("failed", "cancelled")
STATE_LABELS = {"queued": "排队中", "running": "进行中", "paused": "已暂停", "done": "完成", "failed": "失败",
                "cancelled": "已取消"}
INTERRUPTED = "书架程序在任务完成前被关闭，请重新提交（已完成的步骤不会重做）"
MAX_PROGRESS_CHARS = 500


@dataclass(frozen=True)
class Job:
    id: int
    course_id: int | None  # 下载本地模型这类与课程无关的任务为空
    kind: str
    params: dict[str, Any]
    state: str
    progress: str
    result: dict[str, Any]
    error: str
    created_at: str
    updated_at: str

    @property
    def label(self) -> str:
        info = JOB_KINDS.get(self.kind)
        return info.label if info else self.kind

    @property
    def finished(self) -> bool:
        """完成、失败或已取消（已暂停的任务还没结束，可以继续）。"""
        return self.state in FINISHED_STATES

    @property
    def paused(self) -> bool:
        return self.state == "paused"

    @property
    def retryable(self) -> bool:
        return self.state in RETRYABLE_STATES


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _params_json(params: dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True)  # 排好序，相同参数的任务可以直接比较


class JobStore:
    def __init__(self, library: Library) -> None:
        self.conn = library.conn

    def add(self, course: Course | None, kind: str, params: dict[str, Any]) -> Job:
        if kind not in JOB_KINDS:
            raise LibraryError(f"未知的任务类型：{kind}")
        now = _now()
        cursor = self.conn.execute(
            "INSERT INTO job (course_id, kind, params, state, created_at, updated_at) VALUES (?, ?, ?, 'queued', ?, ?)",
            (course.id if course else None, kind, _params_json(params), now, now),
        )
        self.conn.commit()
        return self.get(int(cursor.lastrowid))

    def get(self, job_id: int) -> Job:
        row = self.conn.execute("SELECT * FROM job WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise LibraryError(f"没有编号为 {job_id} 的任务")
        return _job(row)

    # ---------- 状态变化（只在当前状态符合时生效） ----------

    def start(self, job_id: int) -> bool:
        """排队中 → 进行中；任务已被取消时返回 False。"""
        return self._change("UPDATE job SET state = 'running', progress = '开始…', updated_at = ?"
                            " WHERE id = ? AND state = 'queued'", (_now(), job_id))

    def report(self, job_id: int, message: str) -> None:
        self._change("UPDATE job SET progress = ?, updated_at = ? WHERE id = ? AND state = 'running'",
                     (message[:MAX_PROGRESS_CHARS], _now(), job_id))

    def finish(self, job_id: int, result: dict[str, Any]) -> bool:
        return self._change("UPDATE job SET state = 'done', progress = '完成', result = ?, updated_at = ?"
                            " WHERE id = ? AND state = 'running'",
                            (json.dumps(result, ensure_ascii=False), _now(), job_id))

    def fail(self, job_id: int, error: str) -> bool:
        return self._change("UPDATE job SET state = 'failed', error = ?, updated_at = ?"
                            " WHERE id = ? AND state IN ('queued', 'running')", (error, _now(), job_id))

    def cancel(self, job_id: int) -> bool:
        """排队中、进行中或已暂停 → 已取消（进度说明保留，看得出停在哪里）；已经结束的任务返回 False。"""
        return self._change("UPDATE job SET state = 'cancelled', updated_at = ?"
                            " WHERE id = ? AND state IN ('queued', 'running', 'paused')", (_now(), job_id))

    def pause(self, job_id: int) -> bool:
        """排队中或进行中 → 已暂停（进度说明保留）；其他状态返回 False。"""
        return self._change("UPDATE job SET state = 'paused', updated_at = ?"
                            " WHERE id = ? AND state IN ('queued', 'running')", (_now(), job_id))

    def resume(self, job_id: int) -> bool:
        """已暂停 → 排队中；不是暂停状态时返回 False。"""
        return self._change("UPDATE job SET state = 'queued', updated_at = ? WHERE id = ? AND state = 'paused'",
                            (_now(), job_id))

    def fail_interrupted(self) -> int:
        """把上次没执行完的任务标为失败（应用重启后它们不会再继续）。"""
        cursor = self.conn.execute(
            "UPDATE job SET state = 'failed', error = ?, updated_at = ? WHERE state IN ('queued', 'running')",
            (INTERRUPTED, _now()),
        )
        self.conn.commit()
        return cursor.rowcount

    def _change(self, sql: str, values: tuple[Any, ...]) -> bool:
        cursor = self.conn.execute(sql, values)
        self.conn.commit()
        return cursor.rowcount == 1

    # ---------- 查询 ----------

    def recent(self, course: Course, limit: int = 5) -> list[Job]:
        rows = self.conn.execute("SELECT * FROM job WHERE course_id = ? ORDER BY id DESC LIMIT ?", (course.id, limit))
        return [_job(row) for row in rows]

    def of_kind(self, course: Course, kind: str) -> list[Job]:
        """这门课某一类型的全部任务，按提交顺序。"""
        rows = self.conn.execute("SELECT * FROM job WHERE course_id = ? AND kind = ? ORDER BY id", (course.id, kind))
        return [_job(row) for row in rows]

    def latest(self, limit: int = 50) -> list[Job]:
        """所有任务，最新的在前（任务页）。"""
        rows = self.conn.execute("SELECT * FROM job ORDER BY id DESC LIMIT ?", (limit,))
        return [_job(row) for row in rows]

    def active(self, course: Course | None, kind: str, params: dict[str, Any]) -> Job | None:
        """同一门课、同一类型、同样参数、还没结束的任务（避免重复提交）。"""
        row = self.conn.execute(
            "SELECT * FROM job WHERE course_id IS ? AND kind = ? AND params = ? AND state IN ('queued', 'running')"
            " ORDER BY id LIMIT 1",
            (course.id if course else None, kind, _params_json(params)),
        ).fetchone()
        return _job(row) if row else None

    def paused_jobs(self) -> list[Job]:
        rows = self.conn.execute("SELECT * FROM job WHERE state = 'paused' ORDER BY id")
        return [_job(row) for row in rows]

    def retry_candidates(self) -> list[Job]:
        """「全部重试」要重新提交的任务：同一门课、同一类型、同样参数的任务只看最新的一个，它失败或被取消时才算
        （后来已经重新提交并完成、或者还在排队/进行中/暂停的，不再重复提交）。按提交顺序。"""
        rows = self.conn.execute(
            "SELECT * FROM job WHERE id IN (SELECT MAX(id) FROM job GROUP BY course_id, kind, params)"
            " AND state IN ('failed', 'cancelled') ORDER BY id")
        return [_job(row) for row in rows]

    def active_jobs(self, course: Course | None = None) -> list[Job]:
        """还没结束的任务（不给课程时为全部）。"""
        if course is None:
            rows = self.conn.execute("SELECT * FROM job WHERE state IN ('queued', 'running') ORDER BY id")
        else:
            rows = self.conn.execute(
                "SELECT * FROM job WHERE course_id = ? AND state IN ('queued', 'running') ORDER BY id", (course.id,))
        return [_job(row) for row in rows]

    def next_queued(self, lane: str) -> Job | None:
        """这条队列里最早提交、还在排队的任务。"""
        rows = self.conn.execute("SELECT * FROM job WHERE state = 'queued' ORDER BY id")
        return next((_job(row) for row in rows if row["kind"] in JOB_KINDS and JOB_KINDS[row["kind"]].lane == lane),
                    None)


def ensure_idle(library: Library, course: Course, action: str) -> None:
    """删除、移动之前：这门课不能有还没结束的任务（任务进程正在读写这些数据；暂停的任务继续后也会读写）。"""
    store = JobStore(library)
    jobs = store.active_jobs(course) + [job for job in store.paused_jobs() if job.course_id == course.id]
    if jobs:
        names = "、".join(f"#{job.id} {job.label}" for job in sorted(jobs, key=lambda job: job.id)[:3])
        raise LibraryError(f"「{course.name}」还有没结束的后台任务（{names}），请等它们完成或取消后再{action}"
                           "（书架程序意外关闭留下的任务，重新打开一次书架就会清掉）")


def _job(row: Any) -> Job:
    return Job(
        id=row["id"], course_id=row["course_id"], kind=row["kind"], params=json.loads(row["params"]),
        state=row["state"], progress=row["progress"], result=json.loads(row["result"] or "{}"), error=row["error"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )
