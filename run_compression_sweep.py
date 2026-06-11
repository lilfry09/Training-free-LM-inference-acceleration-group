"""Run SnapKV compression-ratio sweep experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]


def output_path(ratio: float) -> Path:
    return ROOT / "results" / f"metrics_cr{int(round(ratio * 100)):03d}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SnapKV compression-ratio sweep.")
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    parser.add_argument("--force", action="store_true", help="Re-run experiments even when output CSV exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for ratio in args.ratios:
        out = output_path(ratio)
        if out.exists() and not args.force:
            print(f"Skipping existing {out}")
            continue

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate.py"),
            "--datasets",
            "wikitext",
            "pg19",
            "--methods",
            "baseline",
            "snapkv",
            "--max-ppl-tokens",
            "512",
            "--ppl-prefill-tokens",
            "128",
            "--generation-prompt-tokens",
            "128",
            "--max-new-tokens",
            "64",
            "--snapkv-compression-ratio",
            str(ratio),
            "--snapkv-window-size",
            "64",
            "--output",
            str(out),
        ]
        print(f"Running compression ratio {ratio:.2f} -> {out}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
