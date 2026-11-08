"""跨进程的数据锁（locks.py）：同一线程可重入，其他线程、其他进程拿不到；进程退出时系统自动释放。"""

import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from lecture_analyzer.library import LibraryError
from lecture_analyzer.locks import discard, hold, is_free, lecture_lock

HOLDER = """
import sys
from pathlib import Path
from lecture_analyzer.locks import hold
with hold(Path(sys.argv[1])):
    print("held", flush=True)
    sys.stdin.readline()
"""


@contextmanager
def held_elsewhere(path: Path) -> Iterator[None]:
    """在另一个线程里拿着这把锁（模拟后台任务正在处理），退出时放开。"""
    taken, done = threading.Event(), threading.Event()

    def keep() -> None:
        with hold(path):
            taken.set()
            done.wait(30)

    thread = threading.Thread(target=keep, daemon=True)
    thread.start()
    assert taken.wait(10)
    try:
        yield
    finally:
        done.set()
        thread.join(10)


def test_reentrant_in_one_thread_exclusive_across_threads(tmp_path):
    path = lecture_lock(tmp_path, 7)
    assert is_free(path) and not path.exists()  # 查看不会建出锁文件
    with hold(path), hold(path, wait=0):  # 同一线程可以重复拿
        assert path.is_file()
        outcome: list[str] = []

        def other() -> None:
            try:
                with hold(path, wait=0.3, busy="第 7 讲正忙"):
                    outcome.append("拿到了")
            except LibraryError as exc:
                outcome.append(str(exc))
            outcome.append("free" if is_free(path) else "busy")

        thread = threading.Thread(target=other)
        thread.start()
        thread.join(10)
        assert outcome == ["第 7 讲正忙", "busy"]
    assert is_free(path)
    discard(path)
    assert not path.exists()


@contextmanager
def _holder(path: Path) -> Iterator[subprocess.Popen[str]]:
    """另一个进程拿着这把锁，直到往它的标准输入写一行（或它被结束）。"""
    process = subprocess.Popen([sys.executable, "-c", HOLDER, str(path)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "held"
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(10)
        process.stdin.close()
        process.stdout.close()


def _let_go(process: subprocess.Popen[str]) -> None:
    process.stdin.write("\n")
    process.stdin.flush()


def test_other_process_holds_until_it_exits(tmp_path):
    path = lecture_lock(tmp_path, 3)
    with _holder(path) as holder:
        assert not is_free(path)
        with pytest.raises(LibraryError, match="正被别的操作使用"):
            with hold(path, wait=0):
                pass
        waited: list[str] = []
        release = threading.Timer(0.5, _let_go, [holder])
        release.start()
        with hold(path, on_wait=lambda: waited.append("等待")):  # 任务进程一直等，对方放开后拿到
            assert waited == ["等待"]
        release.join()

    with _holder(path) as crashed:
        crashed.kill()  # 拿着锁的进程崩溃：系统自动释放
        crashed.wait(10)
        with hold(path, wait=5):
            pass
