import csv
import json
from pathlib import Path


LAYERS = 6
QUERY_HEADS = 8
HEAD_DIM = 64
BYTES_PER_VALUE = 4
BASELINE_TOKENS = 512


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


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


def write_markdown(path: Path, title: str, rows: list[dict], fieldnames: list[str]) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in fieldnames) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def approx_kv_mb(kv_heads: int, tokens: int) -> float:
    values = LAYERS * kv_heads * tokens * HEAD_DIM * 2
    return values * BYTES_PER_VALUE / (1024 * 1024)


def speed_at(result: dict, context_len: int = 512) -> dict:
    for item in result.get("speed_metrics") or []:
        if int(item.get("context_len", -1)) == context_len:
            return item
    return {}


def kvpress_after_len(result: dict) -> int | None:
    summary = (result.get("kv_cache_stats") or {}).get("summary") or {}
    by_before = summary.get("by_before_len") or []
    for item in by_before:
        if int(item.get("before_len", -1)) == BASELINE_TOKENS:
            return int(item.get("after_len_min"))
    return summary.get("after_len_min")


def build_kvpress_rows(outputs_dir: Path) -> list[dict]:
    rows = []
    for ratio in ["0.25", "0.5", "0.75"]:
        path = outputs_dir / f"ablation_streamingllm_r{ratio}_wikitext_test.json"
        if not path.exists():
            continue
        result = load_json(path)
        speed = speed_at(result)
        cache_tokens = kvpress_after_len(result)
        rows.append(
            {
                "dataset": result.get("dataset"),
                "method": "kvpress_streamingllm",
                "compression_ratio": result.get("kvpress_compression_ratio"),
                "cached_ppl": (result.get("cache_ppl_metrics") or {}).get("ppl"),
                "throughput_tok_per_sec": speed.get("throughput_tok_per_sec"),
                "throughput_tok_per_sec_std": speed.get("throughput_tok_per_sec_std"),
                "ttft_sec": speed.get("ttft_sec"),
                "ttft_sec_std": speed.get("ttft_sec_std"),
                "tpot_sec": speed.get("tpot_sec"),
                "tpot_sec_std": speed.get("tpot_sec_std"),
                "cache_tokens": cache_tokens,
                "approx_kv_cache_mb": approx_kv_mb(QUERY_HEADS, cache_tokens) if cache_tokens else None,
                "source": path.name,
            }
        )
    return rows


def build_gqa_rows(outputs_dir: Path) -> list[dict]:
    rows = []
    for kv_heads in [8, 4, 2]:
        path = outputs_dir / f"gqa_sanity_kv{kv_heads}_wikitext_test.json"
        if not path.exists():
            continue
        result = load_json(path)
        speed = speed_at(result)
        rows.append(
            {
                "dataset": result.get("dataset"),
                "kv_heads": kv_heads,
                "kv_head_ratio": result.get("gqa_kv_head_ratio"),
                "chunked_ppl": (result.get("ppl_metrics") or {}).get("ppl"),
                "cached_ppl": (result.get("cache_ppl_metrics") or {}).get("ppl"),
                "throughput_tok_per_sec": speed.get("throughput_tok_per_sec"),
                "throughput_tok_per_sec_std": speed.get("throughput_tok_per_sec_std"),
                "cache_tokens": BASELINE_TOKENS,
                "approx_kv_cache_mb": approx_kv_mb(kv_heads, BASELINE_TOKENS),
                "source": path.name,
            }
        )
    return rows


def build_memory_rows(kvpress_rows: list[dict], gqa_rows: list[dict]) -> list[dict]:
    baseline_mb = approx_kv_mb(QUERY_HEADS, BASELINE_TOKENS)
    rows = [
        {
            "method": "baseline / gqa_8kv",
            "kv_heads": QUERY_HEADS,
            "cache_tokens": BASELINE_TOKENS,
            "approx_kv_cache_mb": baseline_mb,
            "reduction_vs_baseline": "0.0%",
        }
    ]
    for item in gqa_rows:
        if item["kv_heads"] == QUERY_HEADS:
            continue
        mb = item["approx_kv_cache_mb"]
        rows.append(
            {
                "method": f"gqa_{item['kv_heads']}kv",
                "kv_heads": item["kv_heads"],
                "cache_tokens": BASELINE_TOKENS,
                "approx_kv_cache_mb": mb,
                "reduction_vs_baseline": f"{100.0 * (1.0 - mb / baseline_mb):.1f}%",
            }
        )
    for item in kvpress_rows:
        mb = item["approx_kv_cache_mb"]
        rows.append(
            {
                "method": f"streamingllm_r{item['compression_ratio']}",
                "kv_heads": QUERY_HEADS,
                "cache_tokens": item["cache_tokens"],
                "approx_kv_cache_mb": mb,
                "reduction_vs_baseline": f"{100.0 * (1.0 - mb / baseline_mb):.1f}%",
            }
        )
    return rows


def main() -> None:
    outputs_dir = Path("outputs")
    kvpress_rows = build_kvpress_rows(outputs_dir)
    gqa_rows = build_gqa_rows(outputs_dir)
    memory_rows = build_memory_rows(kvpress_rows, gqa_rows)

    kvpress_fields = [
        "dataset",
        "method",
        "compression_ratio",
        "cached_ppl",
        "throughput_tok_per_sec",
        "throughput_tok_per_sec_std",
        "ttft_sec",
        "ttft_sec_std",
        "tpot_sec",
        "tpot_sec_std",
        "cache_tokens",
        "approx_kv_cache_mb",
        "source",
    ]
    gqa_fields = [
        "dataset",
        "kv_heads",
        "kv_head_ratio",
        "chunked_ppl",
        "cached_ppl",
        "throughput_tok_per_sec",
        "throughput_tok_per_sec_std",
        "cache_tokens",
        "approx_kv_cache_mb",
        "source",
    ]
    memory_fields = ["method", "kv_heads", "cache_tokens", "approx_kv_cache_mb", "reduction_vs_baseline"]

    write_csv(outputs_dir / "comparison_kvpress_ablation.csv", kvpress_rows, kvpress_fields)
    write_markdown(outputs_dir / "comparison_kvpress_ablation.md", "KVPress Ratio Ablation", kvpress_rows, kvpress_fields)
    write_csv(outputs_dir / "comparison_gqa_sanity.csv", gqa_rows, gqa_fields)
    write_markdown(outputs_dir / "comparison_gqa_sanity.md", "GQA KV-Head Sanity", gqa_rows, gqa_fields)
    write_csv(outputs_dir / "comparison_memory.csv", memory_rows, memory_fields)
    write_markdown(outputs_dir / "comparison_memory.md", "Approximate KV Cache Memory", memory_rows, memory_fields)

    for name in [
        "comparison_kvpress_ablation",
        "comparison_gqa_sanity",
        "comparison_memory",
    ]:
        print(f"[saved] {outputs_dir / (name + '.csv')}")
        print(f"[saved] {outputs_dir / (name + '.md')}")


if __name__ == "__main__":
    main()
