"""M0：生成人工校对初稿，或计算各转写结果相对校对稿的字错率。

示例：
  # 1. 取 05:00–15:00 生成初稿，人工改成参考稿
  uv run python experiments/cer.py draft experiments/output/asr/funasr_paraformer.json --start 05:00 --end 15:00 --out experiments/output/ref.txt
  # 2. 计算各方案在同一窗口内的字错率
  uv run python experiments/cer.py score --ref experiments/output/ref.txt --start 05:00 --end 15:00 experiments/output/asr/*.json
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

from common import console, print_rows

from lecture_analyzer.asr import load_segments, segments_text
from lecture_analyzer.textnorm import cer
from lecture_analyzer.timecode import format_ms, parse_timecode


_LINE_TAGS = re.compile(r"^\s*(\[[^\]]*\]\s*)+")  # 行首的 [mm:ss][ko] 之类标记


def draft(args: argparse.Namespace) -> None:
    start, end = parse_timecode(args.start), parse_timecode(args.end)
    segments = [
        s for s in load_segments(args.segments)
        if start <= s.start_ms < end and (args.lang is None or s.lang == args.lang)
    ]
    lines = [f"[{format_ms(s.start_ms)}]{f'[{s.lang}]' if s.lang else ''} {s.text}" for s in segments]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"已写出 {len(segments)} 句（{format_ms(start)}–{format_ms(end)}）到 {args.out}，请逐句听录音修改")


def read_reference(path: Path) -> str:
    """读取校对稿，去掉每行开头的时间和语种标记（校对时可以保留它们方便定位）。"""
    return "\n".join(_LINE_TAGS.sub("", line) for line in path.read_text(encoding="utf-8").splitlines())


def score(args: argparse.Namespace) -> None:
    start, end = parse_timecode(args.start), parse_timecode(args.end)
    reference = read_reference(args.ref)
    paths = [Path(p) for pattern in args.hypotheses for p in glob.glob(pattern)]  # Windows 的 shell 不展开通配符
    rows = []
    for path in sorted(paths):
        if path.name == "summary.json":
            continue
        hypothesis = segments_text(load_segments(path), start, end, lang=args.lang)
        rows.append({"转写结果": path.stem, "字错率": cer(reference, hypothesis), "字数": len(hypothesis)})
    scope = f"，只算 {args.lang}" if args.lang else ""
    print_rows(f"字错率（{format_ms(start)}–{format_ms(end)}{scope}，越低越好）", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="转写字错率工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_draft = sub.add_parser("draft", help="导出某个时间窗口的文字，作为人工校对初稿")
    p_draft.add_argument("segments", type=Path)
    p_draft.add_argument("--start", required=True, help="开始时间 mm:ss")
    p_draft.add_argument("--end", required=True, help="结束时间 mm:ss")
    p_draft.add_argument("--out", type=Path, required=True)
    p_draft.add_argument("--lang", help="只导出该语种的句子（如 ko）")
    p_draft.set_defaults(func=draft)

    p_score = sub.add_parser("score", help="计算字错率")
    p_score.add_argument("hypotheses", nargs="+", help="转写结果 JSON，可用通配符")
    p_score.add_argument("--ref", type=Path, required=True, help="人工校对后的参考稿")
    p_score.add_argument("--start", required=True)
    p_score.add_argument("--end", required=True)
    p_score.add_argument("--lang", help="只比较该语种的句子（如 ko），用于韩语 + 翻译的课分别评估")
    p_score.set_defaults(func=score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
