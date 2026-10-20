"""M0：在多个模型上试跑页面识别和单课抽取，比较效果、token 和费用。

示例：
  # 课件页面识别（生成术语表 terms.txt，可作为转写热词）
  uv run python experiments/llm_trial.py digest --slides 第3讲.pptx --models gemini,claude
  # 单课抽取（用上一步某个模型的识别结果作为课件输入，保证各模型对比公平）
  uv run python experiments/llm_trial.py extract --transcript experiments/output/asr/funasr_paraformer.json \
      --digest experiments/output/llm/digest_gemini.json --models claude,gemini
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from common import OUTPUT_ROOT, add_course_args, console, course_settings, print_rows, write_json
from rich.markup import escape

from lecture_analyzer.asr import load_segments
from lecture_analyzer.config import AppConfig, load_config
from lecture_analyzer.course import build_settings
from lecture_analyzer.documents import load_pages
from lecture_analyzer.llm import LLMError, Usage, create_provider
from lecture_analyzer.pipeline.extract import extract_lecture, render_slides, render_transcript
from lecture_analyzer.pipeline.page_digest import collect_terms, digest_pages, needs_image
from lecture_analyzer.render import render_notes
from lecture_analyzer.schemas import PageDigest
from lecture_analyzer.textnorm import normalize_zh


def digest(args: argparse.Namespace, cfg: AppConfig) -> None:
    pages = load_pages(args.slides, args.out / "work")
    image_pages = sum(needs_image(page, args.policy) for page in pages)
    rows = []
    for name in args.models:
        profile = cfg.profile(name)
        if not profile.vision:
            console.print(f"[yellow]跳过 {name}：该档案不支持看图[/yellow]")
            continue
        started = time.perf_counter()
        try:
            digests, usage = digest_pages(
                create_provider(profile), pages, subject=args.subject, policy=args.policy,
                effort=cfg.effort_for_step("page_digest"),
            )
        except LLMError as exc:
            console.print(f"[red]{escape(f'{name} 失败：{exc}')}[/red]")
            continue
        elapsed = time.perf_counter() - started
        write_json(args.out / f"digest_{name}.json", [d.model_dump() for d in digests])
        (args.out / f"digest_{name}.md").write_text(render_slides(digests), encoding="utf-8")
        terms = collect_terms(digests)
        (args.out / f"terms_{name}.txt").write_text("\n".join(terms), encoding="utf-8")
        if not (args.out / "terms.txt").exists():  # 第一个成功的模型的术语表作为默认热词
            (args.out / "terms.txt").write_text("\n".join(terms), encoding="utf-8")
        rows.append(_usage_row(name, elapsed, usage, profile.estimate_cost(usage)) | {
            "页数": len(pages), "带图页": image_pages, "术语数": len(terms),
        })
    print_rows("页面识别对比", rows)
    write_json(args.out / "digest_summary.json", rows)


def extract(args: argparse.Namespace, cfg: AppConfig) -> None:
    segments = load_segments(args.transcript)
    digests = [PageDigest(**item) for item in json.loads(args.digest.read_text(encoding="utf-8"))] if args.digest else []
    settings = course_settings(args) or build_settings(subject=args.subject, note_language=args.notes)
    input_chars = len(render_slides(digests)) + len(render_transcript(segments))
    transcript_norm = normalize_zh("".join(s.text for s in segments))
    rows = []
    for name in args.models:
        profile = cfg.profile(name)
        started = time.perf_counter()
        try:
            result = extract_lecture(
                create_provider(profile), segments, digests, settings, effort=cfg.effort_for_step("extract")
            )
        except LLMError as exc:
            console.print(f"[red]{escape(f'{name} 失败：{exc}')}[/red]")
            continue
        elapsed = time.perf_counter() - started
        notes, usage = result.output, result.usage
        write_json(args.out / f"notes_{name}.json", notes.model_dump())
        (args.out / f"notes_{name}.md").write_text(render_notes(notes, segments), encoding="utf-8")
        quoted = [s for s in notes.signals if s.quote.strip()]
        hits = sum(normalize_zh(s.quote) in transcript_norm for s in quoted)
        rows.append(_usage_row(name, elapsed, usage, profile.estimate_cost(usage)) | {
            "知识点": len(notes.knowledge_points),
            "考点信号": len(notes.signals),
            "原话命中": f"{hits}/{len(quoted)}",
            "token/字": round(usage.input_tokens / max(input_chars, 1), 2),
        })
    print_rows("单课抽取对比（token/字 用于填写模型档案的 tokens_per_char）", rows)
    write_json(args.out / "extract_summary.json", rows)


def _usage_row(name: str, elapsed: float, usage: Usage, cost: float | None) -> dict[str, Any]:
    return {
        "模型": name,
        "耗时(秒)": round(elapsed, 1),
        "输入token": usage.input_tokens,
        "输出token": usage.output_tokens,
        "费用(美元)": cost,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="多模型试跑")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT / "llm")
    sub = parser.add_subparsers(dest="command", required=True)

    p_digest = sub.add_parser("digest", help="课件页面识别")
    p_digest.add_argument("--slides", type=Path, required=True, help="课件或资料（PDF/PPTX/DOCX/图片）")
    p_digest.add_argument("--policy", choices=["auto", "always", "never"], default="auto", help="哪些页面附带图片")
    p_digest.set_defaults(func=digest)

    p_extract = sub.add_parser("extract", help="单课抽取")
    p_extract.add_argument("--transcript", type=Path, required=True, help="转写结果 JSON（asr_compare.py 的输出）")
    p_extract.add_argument("--digest", type=Path, help="课件识别结果 JSON（digest 子命令的输出）")
    p_extract.set_defaults(func=extract)

    for sub_parser in (p_digest, p_extract):
        sub_parser.add_argument("--models", required=True, type=lambda s: [m.strip() for m in s.split(",") if m.strip()],
                                help="逗号分隔的模型档案名，如 claude,gemini")
        add_course_args(sub_parser)

    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.func(args, load_config(args.config))


if __name__ == "__main__":
    main()
