"""后台任务的子进程入口：python -m lecture_analyzer.worker <配置文件> <任务编号>（由 job_runner.py 启动）。"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or not args[1].isdigit():
        print("用法：python -m lecture_analyzer.worker <配置文件> <任务编号>", file=sys.stderr)
        return 2
    from .config import load_config
    from .job_handlers import run_job

    cfg = load_config(Path(args[0]))
    run_job(cfg, cfg.data_dir / "lecture.db", int(args[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
