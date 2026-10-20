"""测试用的任务子进程：按任务参数报告进度、等一会儿，然后完成、失败或直接崩溃。

用法：python job_stub.py <书库> <任务编号>（由 test_jobs.py 交给 JobRunner 启动）
"""

import os
import sys
import time
from pathlib import Path

from lecture_analyzer.jobs import JobStore
from lecture_analyzer.library import Library


def main(db_path: str, job_id: int) -> None:
    with Library(Path(db_path)) as library:
        store = JobStore(library)
        params = store.get(job_id).params
        store.report(job_id, "子进程已启动")
        time.sleep(params.get("sleep", 0))
        outcome = params.get("outcome", "done")
        if outcome == "crash":
            print("模拟崩溃", flush=True)
            os._exit(3)
        if outcome == "fail":
            store.fail(job_id, "模拟失败")
            return
        store.finish(job_id, {"url": "/done"})


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
