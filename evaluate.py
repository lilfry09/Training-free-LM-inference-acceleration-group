import argparse
import math
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from snapkv import (
    AdaptiveSnapKVPress,
    SnapKVConfig,
    compress_cache_with_snapkv,
    kv_cache_length,
    kv_cache_mb,
)


MODEL_NAME = "EleutherAI/pythia-70m"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "metrics.csv"
DATASET_CANDIDATES = {
    "wikitext": [("Salesforce/wikitext", "wikitext-2-raw-v1", "test")],
    "pg19": [
        ("pg19", None, "test"),
        ("deepmind/pg19", None, "test"),
        ("emozilla/pg19", None, "test"),
    ],
}
PG19_FALLBACK_PARAGRAPH = (
    "The old library was quiet in the late afternoon, and every shelf seemed to "
    "hold a different season of memory. A traveler paused near the window with "
    "a notebook open, reading a passage about distant roads, patient work, and "
    "the small decisions that make a long journey possible. Outside, the town "
    "moved at its ordinary pace, while inside the room the sentences gathered "
    "into a steady thread of names, places, questions, and answers."
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def text_from_row(row: dict) -> str:
    for key in ("text", "book_text", "content"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def first_nonempty_text(rows: Iterable[dict], min_chars: int) -> str:
    for row in rows:
        text = text_from_row(row).strip()
        if len(text) >= min_chars:
            return text
    raise RuntimeError(f"No sample with at least {min_chars} characters was found.")


def load_single_sample(dataset_name: str, min_chars: int) -> str:
    errors = []
    for path, config, split in DATASET_CANDIDATES[dataset_name]:
        try:
            kwargs = {"split": split}
            if dataset_name == "pg19":
                kwargs["streaming"] = True
            dataset = load_dataset(path, config, **kwargs) if config else load_dataset(path, **kwargs)
            return first_nonempty_text(dataset, min_chars=min_chars)
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    joined = "\n".join(errors)
    if dataset_name == "pg19":
        repeats = (min_chars // len(PG19_FALLBACK_PARAGRAPH)) + 2
        return " ".join([PG19_FALLBACK_PARAGRAPH] * repeats)
    raise RuntimeError(f"Failed to load {dataset_name}. Tried:\n{joined}")


def prefill(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    method: str,
    args,
) -> tuple[object, torch.Tensor, dict]:
    output_attentions = method in {"snapkv", "adaptive_snapkv"}
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=output_attentions,
        )

    cache = outputs.past_key_values
    context_len = int(input_ids.shape[-1])
    stats = {
        "kv_cache_tokens_before": kv_cache_length(cache),
        "kv_cache_tokens_after": kv_cache_length(cache),
        "kv_cache_compression_ratio_actual": 0.0,
        "kv_cache_mb_after_prefill": kv_cache_mb(cache),
        "prefill_context_length": context_len,
        "snapkv_selected_compression_ratio": 0.0,
    }

    if method == "snapkv":
        config = SnapKVConfig(
            compression_ratio=args.snapkv_compression_ratio,
            window_size=args.snapkv_window_size,
            kernel_size=args.snapkv_kernel_size,
        )
        print(f"Before KV Tokens: {kv_cache_length(cache)}")
        cache, stats = compress_cache_with_snapkv(
            model=model,
            cache=cache,
            attentions=outputs.attentions,
            prompt_len=context_len,
            config=config,
        )
        print(f"After KV Tokens: {kv_cache_length(cache)}")
        stats["snapkv_selected_compression_ratio"] = args.snapkv_compression_ratio
    elif method == "adaptive_snapkv":
        adaptive = AdaptiveSnapKVPress(
            window_size=args.snapkv_window_size,
            kernel_size=args.snapkv_kernel_size,
        )
        cache, stats = adaptive.compress_cache(
            model=model,
            cache=cache,
            attentions=outputs.attentions,
            prompt_len=context_len,
        )

    return cache, outputs.logits[:, -1, :], stats


def cache_perplexity(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    method: str,
    args,
) -> tuple[float, int, dict]:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded.input_ids[:, : args.max_ppl_tokens].to(device)
    if token_ids.shape[-1] < 3:
        raise ValueError("Need at least three tokens to compute cache-based perplexity.")

    prompt_len = min(args.ppl_prefill_tokens, token_ids.shape[-1] - 1)
    prompt = token_ids[:, :prompt_len]
    target = token_ids[:, prompt_len:]
    attention_mask = torch.ones_like(prompt)

    cache, logits, stats = prefill(model, prompt, attention_mask, method, args)

    nll_sum = 0.0
    n_tokens = 0
    for idx in range(target.shape[-1]):
        label = target[:, idx]
        nll_sum += float(F.cross_entropy(logits, label, reduction="sum").item())
        n_tokens += 1

        if idx + 1 < target.shape[-1]:
            with torch.no_grad():
                outputs = model(input_ids=label[:, None], past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

    return math.exp(nll_sum / n_tokens), n_tokens, stats


def generation_speed(model, tokenizer, text: str, device: torch.device, method: str, args) -> dict:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids[:, : args.generation_prompt_tokens].to(device)
    attention_mask = torch.ones_like(input_ids)
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / 1024**2

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    cache, logits, stats = prefill(model, input_ids, attention_mask, method, args)
    generated = []
    next_token = torch.argmax(logits, dim=-1, keepdim=True)

    for _ in range(args.max_new_tokens):
        generated.append(next_token)
        with torch.no_grad():
            outputs = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    rss_after = process.memory_info().rss / 1024**2
    new_tokens = len(generated)

    result = {
        "generation_prompt_tokens": int(input_ids.shape[-1]),
        "max_new_tokens": int(args.max_new_tokens),
        "generated_tokens": new_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": new_tokens / elapsed if elapsed > 0 else float("inf"),
        "process_rss_before_mb": rss_before,
        "process_rss_after_mb": rss_after,
        "process_rss_delta_mb": rss_after - rss_before,
        "generation_context_length": int(stats.get("prefill_context_length", input_ids.shape[-1])),
        "generation_selected_compression_ratio": float(stats.get("snapkv_selected_compression_ratio", 0.0)),
    }
    result.update(stats)
    return result


def evaluate_dataset(args, model, tokenizer, device: torch.device, dataset_name: str, method: str) -> dict:
    print(f"\nLoading {dataset_name} sample...")
    text = load_single_sample(dataset_name, min_chars=args.min_chars)
    print(f"{dataset_name}: loaded {len(text)} characters")

    ppl, ppl_tokens, ppl_cache_stats = cache_perplexity(
        model=model,
        tokenizer=tokenizer,
        text=text,
        device=device,
        method=method,
        args=args,
    )
    speed = generation_speed(
        model=model,
        tokenizer=tokenizer,
        text=text,
        device=device,
        method=method,
        args=args,
    )

    row = {
        "method": method,
        "dataset": dataset_name,
        "model": args.model,
        "device": str(device),
        "sample_chars": len(text),
        "ppl_tokens": ppl_tokens,
        "ppl": ppl,
        "ppl_prefill_tokens": args.ppl_prefill_tokens,
        "ppl_prefill_context_length": int(ppl_cache_stats.get("prefill_context_length", args.ppl_prefill_tokens)),
        "ppl_kv_cache_tokens_before": ppl_cache_stats["kv_cache_tokens_before"],
        "ppl_kv_cache_tokens_after": ppl_cache_stats["kv_cache_tokens_after"],
        "ppl_kv_cache_compression_ratio_actual": ppl_cache_stats["kv_cache_compression_ratio_actual"],
        "ppl_selected_compression_ratio": float(ppl_cache_stats.get("snapkv_selected_compression_ratio", 0.0)),
    }
    row.update(speed)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pythia-70M on WikiText and PG-19.")
    parser.add_argument("--model", default=MODEL_NAME, help="Hugging Face model name or local path.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["wikitext", "pg19"],
        choices=sorted(DATASET_CANDIDATES),
        help="Datasets to evaluate.",
    )
    parser.add_argument("--max-ppl-tokens", type=int, default=256, help="Max tokens per sample for cache PPL.")
    parser.add_argument("--ppl-prefill-tokens", type=int, default=128, help="Prefix length used before cache PPL.")
    parser.add_argument("--generation-prompt-tokens", type=int, default=128, help="Prompt length for speed test.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generated tokens for speed test.")
    parser.add_argument("--min-chars", type=int, default=1000, help="Minimum sample length in characters.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "snapkv"],
        choices=["baseline", "snapkv", "adaptive_snapkv"],
        help="Evaluation methods to compare.",
    )
    parser.add_argument("--snapkv-compression-ratio", type=float, default=0.5)
    parser.add_argument("--snapkv-window-size", type=int, default=64)
    parser.add_argument("--snapkv-kernel-size", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="eager")
    model.to(device)
    model.eval()

    rows = []
    for name in args.datasets:
        for method in args.methods:
            rows.append(evaluate_dataset(args, model, tokenizer, device, name, method))
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print("\n=== Results ===")
    print(df.to_string(index=False))
    print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
