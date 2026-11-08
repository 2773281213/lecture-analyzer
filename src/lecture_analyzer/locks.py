"""跨进程的数据锁：书架程序、任务子进程和命令行同时操作一个书库时，不让两边同时改同一份数据。

后台任务只锁住它此刻正在处理的那一讲、那份资料，其他讲和资料照常可以删除、移动、追加节：
- 讲锁 lecture-<讲的编号>：任务正在处理这一讲（抽音频到上架）；删除、移动这一讲和追加节要先拿到它；
- 资料锁 material-<资料编号>：正在识别、拆题或对应知识点；删除这份资料要先拿到它；
- 资料导入锁 materials-<课程编号>：正在导入资料（新建资料、拆页）；删除这门课的资料要先拿到它；
- 课程记忆锁 memory-<课程编号>：正在重写课程记忆或导出整门课（整合、重建、上架、删除或移动讲）。

锁文件在 data/work/locks/，拿着锁的进程退出时系统自动释放，崩溃也不会留下解不开的锁。
同一个线程可以重复拿同一把锁；同一进程的不同线程之间也互斥。
任务进程等锁不设时限（界面上的操作只占几秒）；界面和命令行的操作只等几秒，拿不到就说明原因、请稍后再试。
加锁的顺序固定为 讲锁 → 资料锁、资料导入锁 → 课程记忆锁，不会互相等死。
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from .library import LibraryError

UI_WAIT = 3.0  # 界面和命令行的操作最多等这么久（例如等任务导出完）
POLL_SECONDS = 0.2


def lecture_lock(data_dir: Path, lecture_id: int) -> Path:
    return _lock_dir(data_dir) / f"lecture-{lecture_id}.lock"


def material_lock(data_dir: Path, material_id: int) -> Path:
    return _lock_dir(data_dir) / f"material-{material_id}.lock"


def materials_lock(data_dir: Path, course_id: int) -> Path:
    return _lock_dir(data_dir) / f"materials-{course_id}.lock"


def memory_lock(data_dir: Path, course_id: int) -> Path:
    return _lock_dir(data_dir) / f"memory-{course_id}.lock"


def _lock_dir(data_dir: Path) -> Path:
    return data_dir / "work" / "locks"


@contextmanager
def hold(
    path: Path, wait: float | None = None, busy: str = "", on_wait: Callable[[], None] | None = None
) -> Iterator[None]:
    """拿着锁执行一段代码。wait 为 None 时一直等（先调用 on_wait 说明在等什么）；
    否则最多等 wait 秒，拿不到时抛出 LibraryError(busy)。"""
    if not _acquire(path, 0):
        if wait is None:
            if on_wait is not None:
                on_wait()
            _acquire(path, None)
        elif wait <= 0 or not _acquire(path, wait):
            raise LibraryError(busy or "数据正被别的操作使用，请稍后再试")
    try:
        yield
    finally:
        _release(path)


def is_free(path: Path) -> bool:
    """这把锁现在有没有被别的进程或线程拿着（显示页面时用，判断完锁可能随即被拿走）。"""
    if not path.exists():
        return True
    if not _acquire(path, 0):
        return False
    _release(path)
    return True


def discard(path: Path) -> None:
    """删除讲或资料之后，删掉它的锁文件（还有进程开着它时删不掉，留着也无妨）。"""
    with suppress(OSError):
        path.unlink(missing_ok=True)


# ---------- 进程内的登记：可重入、线程之间互斥 ----------


@dataclass
class _Entry:
    guard: threading.RLock = field(default_factory=threading.RLock)
    depth: int = 0
    file: IO[bytes] | None = None


_registry_guard = threading.Lock()
_entries: dict[str, _Entry] = {}


def _entry(path: Path) -> _Entry:
    key = os.path.normcase(os.path.abspath(path))
    with _registry_guard:
        return _entries.setdefault(key, _Entry())


def _acquire(path: Path, wait: float | None) -> bool:
    """wait：0 为只试一次，None 为一直等，其余为最多等几秒。"""
    entry = _entry(path)
    deadline = None if wait is None else time.monotonic() + wait
    if wait is None:
        entry.guard.acquire()
    elif wait > 0:
        if not entry.guard.acquire(timeout=wait):
            return False
    elif not entry.guard.acquire(blocking=False):
        return False
    if entry.depth == 0:
        file = _lock_file(path, deadline)
        if file is None:
            entry.guard.release()
            return False
        entry.file = file
    entry.depth += 1
    return True


def _release(path: Path) -> None:
    entry = _entry(path)
    entry.depth -= 1
    if entry.depth == 0 and entry.file is not None:
        with suppress(OSError):
            unlock_file(entry.file)
        entry.file.close()
        entry.file = None
    entry.guard.release()


def _lock_file(path: Path, deadline: float | None) -> IO[bytes] | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        file = path.open("a+b")  # 每次重新打开：锁文件被删掉后不会锁在已删除的文件上
        try:
            lock_file(file)
            return file
        except OSError:  # 别的进程拿着锁
            file.close()
        if deadline is not None and time.monotonic() >= deadline:
            return None
        time.sleep(POLL_SECONDS)


# ---------- 系统的文件锁（job_runner 的执行器锁也用它们） ----------

if os.name == "nt":
    import msvcrt

    def lock_file(file: IO[bytes]) -> None:
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)

    def unlock_file(file: IO[bytes]) -> None:
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def lock_file(file: IO[bytes]) -> None:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def unlock_file(file: IO[bytes]) -> None:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
