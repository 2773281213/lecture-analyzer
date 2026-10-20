"""后台任务执行器：每个任务一个子进程；重任务、轻任务各一条队列，各自按提交顺序一个个执行。

- 子进程是 `python -m lecture_analyzer.worker <配置文件> <任务编号>`：转写占满显卡甚至崩溃都不影响界面，
  每个任务结束后显存全部释放；
- 排队以任务表为准：提交时只唤醒队列，队列线程从表里取最早的排队任务；
- 取消：排队中的直接标为已取消；进行中的先标为已取消，再结束整个子进程树（含 ffmpeg 等）。
  "开始执行"和"取消"在同一把锁里完成，不会出现标了取消、进程却没被结束的情况；
- 子进程没写结果就退出（崩溃、被结束）时标为失败，失败原因附上日志的最后几行；
  日志在 data/work/jobs/，只保留失败的任务的；
- 关闭书架时结束所有子进程，任务标为失败；重新提交会接着做（已完成的步骤、已识别的页不重做）；
- 同一个书库同一时间只有一个执行器（锁文件 data/work/jobs/runner.lock，进程退出时系统自动释放）：
  例如开着桌面程序又运行 `lec serve`，后打开的只提交任务，由先打开的执行，也不会把先打开的那个正在执行的任务
  当成上次没执行完的；在后打开的书架里取消任务（只改任务表），执行器两秒内发现并结束进程。
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import traceback
from contextlib import suppress
from collections.abc import Callable
from pathlib import Path
from typing import IO

from .config import AppConfig
from .jobs import INTERRUPTED, LANES, JobStore
from .library import Library

POLL_SECONDS = 2.0  # 队列线程没被唤醒时，隔这么久查一次任务表
CANCEL_POLL_SECONDS = 2.0  # 任务运行期间隔这么久看一次是否在别处被取消
LOG_TAIL_LINES = 8
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # Windows：子进程不弹出控制台窗口

Command = Callable[[int], list[str]]


def worker_command(cfg: AppConfig) -> Command:
    """子进程的命令行：用同一个 Python 运行 worker，并传入配置文件（子进程重新读取配置和密钥）。"""
    config = cfg.source or cfg.root / "config.toml"
    return lambda job_id: [sys.executable, "-m", "lecture_analyzer.worker", str(config), str(job_id)]


def log_path(data_dir: Path, job_id: int) -> Path:
    return data_dir / "work" / "jobs" / f"job-{job_id}.log"


class RunnerLock:
    """书库的执行器锁：拿到锁的进程负责执行这个书库的后台任务。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file: IO[bytes] | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+b")
        try:
            _lock(file)
        except OSError:  # 别的进程拿着锁
            file.close()
            return False
        self.file = file
        return True

    def release(self) -> None:
        if self.file is None:
            return
        with suppress(OSError):
            _unlock(self.file)
        self.file.close()
        self.file = None


if os.name == "nt":
    import msvcrt

    def _lock(file: IO[bytes]) -> None:
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(file: IO[bytes]) -> None:
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(file: IO[bytes]) -> None:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(file: IO[bytes]) -> None:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def kill_tree(process: subprocess.Popen[bytes]) -> None:
    """结束子进程及它启动的进程（ffmpeg、解压程序等）。"""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, check=False,
                       creationflags=NO_WINDOW)
    else:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _tail(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return ("：\n" + "\n".join(lines[-LOG_TAIL_LINES:])) if lines else ""


class JobRunner:
    def __init__(self, cfg: AppConfig, db_path: Path, command: Command | None = None) -> None:
        self.cfg = cfg
        self.db_path = db_path
        self.command = command or worker_command(cfg)
        self._lock = threading.Lock()
        self._processes: dict[int, subprocess.Popen[bytes]] = {}
        self._wake = {lane: threading.Event() for lane in LANES}
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._owner_lock = RunnerLock(log_path(cfg.data_dir, 0).with_name("runner.lock"))
        self.owner = False  # 是否由本进程执行这个书库的任务

    def start(self) -> None:
        """成为这个书库的执行器：先把上次没执行完的任务标为失败，再启动两条队列。
        别的书架程序已经在执行这个书库的任务时，本进程只提交任务。"""
        if self.owner or not self._owner_lock.acquire():
            return
        self.owner = True
        with Library(self.db_path) as library:
            JobStore(library).fail_interrupted()
        self._threads = [threading.Thread(target=self._loop, args=(lane,), name=f"lec-jobs-{lane}", daemon=True)
                         for lane in LANES]
        for thread in self._threads:
            thread.start()

    def submit(self, job_id: int) -> None:
        """任务已经写进任务表（排队中）：唤醒队列。"""
        for event in self._wake.values():
            event.set()

    def cancel(self, job_id: int) -> bool:
        """取消排队中或进行中的任务；任务已经结束时返回 False。"""
        with self._lock:
            with Library(self.db_path) as library:
                if not JobStore(library).cancel(job_id):
                    return False
            process = self._processes.get(job_id)
        if process is not None:
            kill_tree(process)
        return True

    def running(self) -> list[int]:
        with self._lock:
            return list(self._processes)

    def shutdown(self) -> None:
        """关闭书架：不再取新任务，结束正在运行的子进程（任务标为失败）。"""
        self._stop.set()
        for event in self._wake.values():
            event.set()
        with self._lock:
            processes = dict(self._processes)
            if processes:
                with Library(self.db_path) as library:
                    for job_id in processes:
                        JobStore(library).fail(job_id, INTERRUPTED)
        for process in processes.values():
            kill_tree(process)
        self._owner_lock.release()
        self.owner = False

    # ---------- 队列线程 ----------

    def _loop(self, lane: str) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._next(lane)
                if job_id is None:
                    self._wake[lane].wait(POLL_SECONDS)
                    self._wake[lane].clear()
                    continue
                self._run(job_id)
            except Exception:  # 队列线程不能退出：打印错误，稍后再试
                traceback.print_exc()
                self._stop.wait(POLL_SECONDS)

    def _next(self, lane: str) -> int | None:
        with Library(self.db_path) as library:
            job = JobStore(library).next_queued(lane)
        return job.id if job else None

    def _run(self, job_id: int) -> None:
        log = log_path(self.cfg.data_dir, job_id)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("wb") as output:
            process = self._launch(job_id, output)
            if process is None:
                return
            code = self._wait(job_id, process)
        with self._lock:
            self._processes.pop(job_id, None)
        self._settle(job_id, code, log)

    def _launch(self, job_id: int, output: IO[bytes]) -> subprocess.Popen[bytes] | None:
        """标为进行中并启动子进程；任务已被取消、书架正在关闭时返回 None。"""
        with self._lock, Library(self.db_path) as library:
            store = JobStore(library)
            if self._stop.is_set() or not store.start(job_id):
                return None
            try:
                process = subprocess.Popen(
                    self.command(job_id), stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    creationflags=NO_WINDOW, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
            except OSError as exc:
                store.fail(job_id, f"无法启动任务进程：{exc}")
                return None
            self._processes[job_id] = process
            return process

    def _wait(self, job_id: int, process: subprocess.Popen[bytes]) -> int:
        """等子进程结束；任务在别的书架程序里被取消（只改了任务表）时结束它。"""
        while True:
            try:
                return process.wait(timeout=CANCEL_POLL_SECONDS)
            except subprocess.TimeoutExpired:
                with Library(self.db_path) as library:
                    cancelled = JobStore(library).get(job_id).state == "cancelled"
                if cancelled:
                    kill_tree(process)

    def _settle(self, job_id: int, code: int, log: Path) -> None:
        """子进程退出后：没写结果的（崩溃、被结束）标为失败；只保留失败任务的日志。"""
        with Library(self.db_path) as library:
            store = JobStore(library)
            if store.get(job_id).state == "running":
                store.fail(job_id, f"任务进程意外退出（退出码 {code}）{_tail(log)}")
            state = store.get(job_id).state
        if state != "failed":
            log.unlink(missing_ok=True)
