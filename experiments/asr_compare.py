"""M0：用同一段录音对比几种转写方案的速度和结果（字错率用 cer.py 计算）。

示例：
  # 中文课：取 30:00 起的 15 分钟
  uv run python experiments/asr_compare.py 药理学3.m4a --start 30:00 --minutes 15 \
      --backend funasr:paraformer --backend funasr:sensevoice --backend qwen3:1.7b
  # 韩语 + 现场翻译的课
  uv run python experiments/asr_compare.py 病毒学2-1.mp4 --start 20:00 --minutes 15 \
      --backend funasr:sensevoice --backend qwen3:1.7b --backend qwen3:0.6b --out experiments/output/asr_virology
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from common import OUTPUT_ROOT, add_course_args, console, course_settings, print_rows, read_lines, write_json
from rich.markup import escape

from lecture_analyzer.asr import create_backend, save_segments
from lecture_analyzer.config import load_config
from lecture_analyzer.media import extract_audio, wav_duration_ms
from lecture_analyzer.timecode import format_ms, parse_timecode


def main() -> None:
    parser = argparse.ArgumentParser(description="对比转写方案的速度和结果")
    parser.add_argument("media", type=Path, help="录音或录像文件")
    parser.add_argument("--backend", action="append", required=True,
                        help="可重复：funasr:paraformer、funasr:sensevoice、qwen3:1.7b、qwen3:0.6b、llm:<模型档案>")
    parser.add_argument("--hotwords", type=Path, help="热词文件，每行一个（llm_trial.py digest 会生成 terms.txt）")
    parser.add_argument("--start", default="00:00", help="从哪里开始截取，mm:ss 或 h:mm:ss")
    parser.add_argument("--minutes", type=float, help="截取多少分钟，节省时间和费用；不填则到结尾")
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT / "asr")
    parser.add_argument("--config", type=Path)
    add_course_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    settings = course_settings(args)
    start_ms = parse_timecode(args.start)
    wav = extract_audio(args.media, args.out / "input.wav", start_seconds=start_ms / 1000,
                        max_seconds=args.minutes * 60 if args.minutes else None)
    audio_seconds = wav_duration_ms(wav) / 1000
    hotwords = read_lines(args.hotwords)
    console.print(f"截取 {args.start} 起 {audio_seconds / 60:.1f} 分钟，热词 {len(hotwords)} 个")

    rows = []
    for spec in args.backend:
        backend = create_backend(cfg, spec, work_dir=args.out / "work", settings=settings)
        try:
            backend.warm_up()
            started = time.perf_counter()
            segments = backend.transcribe(wav, hotwords)
            elapsed = time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001 —— 实验脚本：一个方案失败不影响其他方案，记下原因继续
            console.print(f"[red]{escape(f'{spec} 失败：{type(exc).__name__}: {exc}')}[/red]")
            rows.append({"方案": spec, "耗时(秒)": None, "实时率": None, "句数": None, "字数": None,
                         "语种(句数)": f"失败：{type(exc).__name__}", "费用(美元)": None})
            continue
        finally:
            backend.close()  # 依次运行多个本地模型时释放显存

        # 时间戳换算回原文件的时间，方便对照录音
        segments = [replace(s, start_ms=s.start_ms + start_ms, end_ms=s.end_ms + start_ms) for s in segments]
        stem = spec.replace(":", "_")
        save_segments(segments, args.out / f"{stem}.json")
        text = "\n".join(f"[{format_ms(s.start_ms)}]{f'[{s.lang}]' if s.lang else ''} {s.text}" for s in segments)
        (args.out / f"{stem}.txt").write_text(text, encoding="utf-8")

        usage = getattr(backend, "usage", None)  # 只有大模型转写有用量
        cost = backend.provider.profile.estimate_cost(usage) if usage is not None else None  # type: ignore[attr-defined]
        langs = Counter(s.lang for s in segments if s.lang)
        rows.append({
            "方案": spec,
            "耗时(秒)": round(elapsed, 1),
            "实时率": round(elapsed / audio_seconds, 3),
            "句数": len(segments),
            "字数": sum(len(s.text) for s in segments),
            "语种(句数)": " ".join(f"{lang}:{n}" for lang, n in langs.most_common()) or "-",
            "费用(美元)": cost,
        })

    print_rows("转写对比（实时率 = 耗时 / 音频时长，越小越快）", rows)
    write_json(args.out / "summary.json", rows)
    console.print(f"结果已写入 {args.out}")


if __name__ == "__main__":
    main()
