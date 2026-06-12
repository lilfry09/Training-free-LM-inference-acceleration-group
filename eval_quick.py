"""Quick, real evaluation for layer-wise KV cache compression."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from layerwise_compression import (
    CompressionConfig,
    LayerWiseKVCompressor,
    cache_token_count,
    format_schedule,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR.parent / "finalproj" / "models" / "pythia-70m"
DEFAULT_TEXT_PATH = PROJECT_DIR.parent / "finalproj" / "datasets" / "pg19_samples" / "test_1.txt"
DEFAULT_OUTPUT = PROJECT_DIR / "results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate layer-wise KV cache compression.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_path", type=Path, default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--ppl_tokens", type=int, default=160)
    parser.add_argument("--prompt_tokens", type=int, default=96)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--min_cache_tokens", type=int, default=16)
    parser.add_argument("--keep_initial_tokens", type=int, default=4)
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def sample_text(text_path: Path) -> str:
    if text_path.exists():
        return text_path.read_text(encoding="utf-8", errors="ignore")

    return (
        "Natural language processing systems often need to process long contexts. "
        "During autoregressive decoding, transformer models cache keys and values "
        "from previous tokens so that each new token can reuse earlier computation. "
        "The cache improves speed, but it grows with sequence length and can become "
        "a memory bottleneck for deployment. Layer-wise compression keeps more "
        "context in shallow layers, compresses middle layers aggressively, and keeps "
        "a moderate amount of context in deep layers. "
    ) * 8


def load_model_and_tokenizer(model_path: Path, device: str, dtype: torch.dtype):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}. "
            "Pass --model_path or run finalproj/scripts/prepare_assets.ps1 first."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def time_generation(
    compressor: LayerWiseKVCompressor,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    repeats: int,
    device: str,
    compress: bool,
) -> tuple[float, dict[str, float]]:
    runs = []
    last_stats: dict[str, float] = {}

    for _ in range(repeats):
        sync_if_needed(device)
        start = time.perf_counter()
        _, last_stats = compressor.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=None,
            compress=compress,
        )
        sync_if_needed(device)
        runs.append(time.perf_counter() - start)

    return sum(runs) / len(runs), last_stats


@torch.no_grad()
def continuation_perplexity(
    compressor: LayerWiseKVCompressor,
    input_ids: torch.Tensor,
    *,
    context_tokens: int,
    eval_tokens: int,
    compress: bool,
) -> tuple[float, int, float]:
    """Score true continuation tokens after a full-context prefill."""

    if input_ids.shape[1] <= context_tokens:
        raise RuntimeError("Not enough tokens for continuation perplexity.")

    actual_eval_tokens = min(eval_tokens, input_ids.shape[1] - context_tokens)
    prompt_ids = input_ids[:, :context_tokens]
    target_ids = input_ids[:, context_tokens : context_tokens + actual_eval_tokens]

    outputs = compressor._forward_step(prompt_ids, None, 0)
    past_key_values = outputs.past_key_values
    if compress:
        past_key_values = compressor._compress(past_key_values)

    logits = outputs.logits[:, -1, :]
    losses = []
    for offset in range(actual_eval_tokens):
        target = target_ids[:, offset]
        losses.append(F.cross_entropy(logits, target, reduction="none"))

        if offset + 1 >= actual_eval_tokens:
            break

        current_input = target.reshape(1, 1)
        absolute_position = context_tokens + offset
        outputs = compressor._forward_step(current_input, past_key_values, absolute_position)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

    mean_nll = torch.cat(losses).mean()
    avg_cache_tokens = 0.0
    if past_key_values is not None:
        avg_cache_tokens = cache_token_count(past_key_values) / compressor.total_layers
    return float(torch.exp(mean_nll).item()), actual_eval_tokens, avg_cache_tokens


def evaluate_method(
    name: str,
    compressor: LayerWiseKVCompressor,
    dense_time: float | None,
    dense_cache_tokens: float | None,
    ppl_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    args: argparse.Namespace,
    device: str,
    compress: bool,
) -> dict[str, object]:
    ppl, scored_tokens, ppl_cache_tokens = continuation_perplexity(
        compressor,
        ppl_ids,
        context_tokens=args.prompt_tokens,
        eval_tokens=args.ppl_tokens - args.prompt_tokens,
        compress=compress,
    )
    gen_time, gen_stats = time_generation(
        compressor,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        repeats=args.repeats,
        device=device,
        compress=compress,
    )

    speedup = 1.0 if dense_time is None else dense_time / gen_time
    avg_cache_tokens = gen_stats["avg_cache_tokens_per_layer"]
    cache_reduction = 0.0
    if dense_cache_tokens:
        cache_reduction = max(0.0, 1.0 - avg_cache_tokens / dense_cache_tokens)

    return {
        "method": name,
        "ppl": round(ppl, 4),
        "generation_time_sec": round(gen_time, 4),
        "tokens_per_sec": round(args.max_new_tokens / gen_time, 4),
        "speedup": round(speedup, 4),
        "avg_cache_tokens_per_layer": round(avg_cache_tokens, 4),
        "ppl_final_cache_tokens_per_layer": round(ppl_cache_tokens, 4),
        "cache_reduction": round(cache_reduction, 4),
        "ppl_scored_tokens": int(scored_tokens),
    }


def main() -> list[dict[str, object]]:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print("=" * 72)
    print("Layer-wise Adaptive KV Cache Compression - Quick Real Evaluation")
    print("=" * 72)
    print(f"Model: {args.model_path}")
    print(f"Text: {args.text_path if args.text_path.exists() else 'built-in fallback'}")
    print(f"Device: {device}, dtype: {dtype}")

    model, tokenizer = load_model_and_tokenizer(args.model_path.resolve(), device, dtype)
    encoded = tokenizer(sample_text(args.text_path.resolve()), return_tensors="pt", truncation=False)
    token_ids = encoded["input_ids"].to(device)
    if token_ids.shape[1] < max(args.ppl_tokens, args.prompt_tokens) + 1:
        raise RuntimeError("Sample text produced too few tokens for the requested evaluation.")

    ppl_ids = token_ids[:, : args.ppl_tokens]
    prompt_ids = token_ids[:, : args.prompt_tokens]

    dense = LayerWiseKVCompressor(
        model,
        CompressionConfig(
            1.0,
            1.0,
            1.0,
            min_tokens=args.min_cache_tokens,
            keep_initial_tokens=args.keep_initial_tokens,
        ),
    )
    uniform = LayerWiseKVCompressor(
        model,
        CompressionConfig(
            0.5,
            0.5,
            0.5,
            min_tokens=args.min_cache_tokens,
            keep_initial_tokens=args.keep_initial_tokens,
        ),
    )
    uniform_67 = LayerWiseKVCompressor(
        model,
        CompressionConfig(
            0.67,
            0.67,
            0.67,
            min_tokens=args.min_cache_tokens,
            keep_initial_tokens=args.keep_initial_tokens,
        ),
    )
    layerwise = LayerWiseKVCompressor(
        model,
        CompressionConfig(
            0.8,
            0.5,
            0.7,
            min_tokens=args.min_cache_tokens,
            keep_initial_tokens=args.keep_initial_tokens,
        ),
    )

    print(f"Layer-wise schedule: {format_schedule(layerwise.describe_schedule())}")

    results = []
    dense_result = evaluate_method(
        "Dense",
        dense,
        None,
        None,
        ppl_ids,
        prompt_ids,
        args,
        device,
        compress=False,
    )
    results.append(dense_result)

    dense_time = float(dense_result["generation_time_sec"])
    dense_cache_tokens = float(dense_result["avg_cache_tokens_per_layer"])

    results.append(
        evaluate_method(
            "Uniform-50%",
            uniform,
            dense_time,
            dense_cache_tokens,
            ppl_ids,
            prompt_ids,
            args,
            device,
            compress=True,
        )
    )
    results.append(
        evaluate_method(
            "Uniform-67%",
            uniform_67,
            dense_time,
            dense_cache_tokens,
            ppl_ids,
            prompt_ids,
            args,
            device,
            compress=True,
        )
    )
    results.append(
        evaluate_method(
            "LayerWise-80/50/70",
            layerwise,
            dense_time,
            dense_cache_tokens,
            ppl_ids,
            prompt_ids,
            args,
            device,
            compress=True,
        )
    )

    payload = {
        "model_path": str(args.model_path.resolve()),
        "device": device,
        "dtype": str(dtype),
        "ppl_tokens": args.ppl_tokens,
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "min_cache_tokens": args.min_cache_tokens,
        "keep_initial_tokens": args.keep_initial_tokens,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n| Method | PPL | Time (s) | Tokens/s | Speedup | Cache Reduction |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row['method']} | {row['ppl']} | {row['generation_time_sec']} | "
            f"{row['tokens_per_sec']} | {row['speedup']}x | {row['cache_reduction']:.1%} |"
        )
    print(f"\nSaved: {args.output_json.resolve()}")
    return results


if __name__ == "__main__":
    main()
