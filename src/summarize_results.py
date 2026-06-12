import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build paper-ready comparison CSV/Markdown tables.")
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    return parser.parse_args()


def method_label(result: dict) -> str:
    mode = result["mode"]
    if mode == "kvpress":
        ratio = result.get("kvpress_compression_ratio")
        method = result.get("kvpress_method")
        label = f"kvpress_{method}_r{ratio}"
        if method == "streamingllm":
            label += f"_sink{result.get('kvpress_n_sink')}"
        return label
    if mode == "gqa":
        if result.get("gqa_impl") == "reduced_kv_grouped_attention":
            return f"gqa_reduced_kv{result.get('gqa_kv_heads')}"
        return f"gqa_kv{result.get('gqa_kv_heads')}"
    return mode


def load_metric_files(outputs_dir: Path) -> list[dict]:
    results = []
    for path in sorted(outputs_dir.glob("metrics_*.json")):
        with path.open(encoding="utf-8") as f:
            result = json.load(f)
        if "cache_ppl_metrics" not in result:
            print(f"[skip] {path.name}: missing cache_ppl_metrics; likely an old-format metrics file.")
            continue
        result["_path"] = path.name
        results.append(result)
    return results


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown_table(path: Path, title: str, rows: list[dict], fieldnames: list[str]) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(k)) for k in fieldnames) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_quality_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        chunked = result.get("ppl_metrics") or {}
        cached = result.get("cache_ppl_metrics") or {}
        kv_summary = (result.get("kv_cache_stats") or {}).get("summary") or {}
        rows.append(
            {
                "dataset": result.get("dataset"),
                "split": result.get("split"),
                "method": method_label(result),
                "chunked_ppl": chunked.get("ppl"),
                "chunked_tokens": chunked.get("tokens"),
                "cache_ppl": cached.get("ppl"),
                "cache_tokens": cached.get("tokens"),
                "cache_context_len": cached.get("context_len"),
                "kv_events": kv_summary.get("num_events", 0),
                "kv_avg_compression": kv_summary.get("avg_compression_ratio_actual"),
                "source": result.get("_path"),
            }
        )
    return sorted(rows, key=lambda x: (x["dataset"] or "", x["method"] or ""))


def build_speed_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        for speed in result.get("speed_metrics") or []:
            rows.append(
                {
                    "dataset": result.get("dataset"),
                    "split": result.get("split"),
                    "method": method_label(result),
                    "context_len": speed.get("context_len"),
                    "ttft_sec": speed.get("ttft_sec"),
                    "ttft_sec_std": speed.get("ttft_sec_std"),
                    "tpot_sec": speed.get("tpot_sec"),
                    "tpot_sec_std": speed.get("tpot_sec_std"),
                    "e2e_sec": speed.get("e2e_sec"),
                    "e2e_sec_std": speed.get("e2e_sec_std"),
                    "throughput_tok_per_sec": speed.get("throughput_tok_per_sec"),
                    "throughput_tok_per_sec_std": speed.get("throughput_tok_per_sec_std"),
                    "generated_tokens": speed.get("generated_tokens"),
                    "speed_repeats": speed.get("speed_repeats"),
                    "source": result.get("_path"),
                }
            )
    return sorted(rows, key=lambda x: (x["dataset"] or "", x["method"] or "", x["context_len"] or 0))


def main():
    args = parse_args()
    outputs_dir = Path(args.outputs_dir).resolve()
    results = load_metric_files(outputs_dir)

    quality_fields = [
        "dataset",
        "split",
        "method",
        "chunked_ppl",
        "chunked_tokens",
        "cache_ppl",
        "cache_tokens",
        "cache_context_len",
        "kv_events",
        "kv_avg_compression",
        "source",
    ]
    speed_fields = [
        "dataset",
        "split",
        "method",
        "context_len",
        "ttft_sec",
        "ttft_sec_std",
        "tpot_sec",
        "tpot_sec_std",
        "e2e_sec",
        "e2e_sec_std",
        "throughput_tok_per_sec",
        "throughput_tok_per_sec_std",
        "generated_tokens",
        "speed_repeats",
        "source",
    ]

    quality_rows = build_quality_rows(results)
    speed_rows = build_speed_rows(results)
    write_csv(outputs_dir / "comparison_quality.csv", quality_rows, quality_fields)
    write_csv(outputs_dir / "comparison_speed.csv", speed_rows, speed_fields)
    write_markdown_table(outputs_dir / "comparison_quality.md", "Quality Comparison", quality_rows, quality_fields)
    write_markdown_table(outputs_dir / "comparison_speed.md", "Speed Comparison", speed_rows, speed_fields)
    print(f"[saved] {outputs_dir / 'comparison_quality.csv'}")
    print(f"[saved] {outputs_dir / 'comparison_quality.md'}")
    print(f"[saved] {outputs_dir / 'comparison_speed.csv'}")
    print(f"[saved] {outputs_dir / 'comparison_speed.md'}")


if __name__ == "__main__":
    main()
