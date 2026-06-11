"""Run baseline, fixed-ratio SnapKV, and adaptive SnapKV comparison."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Baseline vs SnapKV-30% vs Adaptive SnapKV.")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "metrics_adaptive_comparison.csv")
    parser.add_argument("--fixed-ratio", type=float, default=0.3, help="Fixed SnapKV compression ratio.")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--max-ppl-tokens", type=int, default=512)
    parser.add_argument("--ppl-prefill-tokens", type=int, default=128)
    parser.add_argument("--generation-prompt-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-chars", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate.py"),
        "--datasets",
        "wikitext",
        "pg19",
        "--methods",
        "baseline",
        "snapkv",
        "adaptive_snapkv",
        "--max-ppl-tokens",
        str(args.max_ppl_tokens),
        "--ppl-prefill-tokens",
        str(args.ppl_prefill_tokens),
        "--generation-prompt-tokens",
        str(args.generation_prompt_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--min-chars",
        str(args.min_chars),
        "--snapkv-compression-ratio",
        str(args.fixed_ratio),
        "--snapkv-window-size",
        str(args.window_size),
        "--snapkv-kernel-size",
        str(args.kernel_size),
        "--output",
        str(args.output),
    ]

    print("Running baseline + fixed SnapKV + adaptive SnapKV ->", args.output)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
