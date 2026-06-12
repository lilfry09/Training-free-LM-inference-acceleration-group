import argparse
import json
import math
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress_adapter import build_press, kvpress_on_gptneox
from methods import choose_attn_impl, enable_gqa


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pythia-70M with baseline/flash/gqa/kvpress variants.")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--model_path", type=str, default="models/pythia-70m")
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "flash", "gqa", "gqa_flash", "kvpress"],
    )
    parser.add_argument("--gqa_kv_heads", type=int, default=2)
    parser.add_argument("--kvpress_method", type=str, default="knorm", choices=["knorm", "streamingllm"])
    parser.add_argument("--kvpress_compression_ratio", type=float, default=0.5)
    parser.add_argument("--kvpress_n_sink", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "pg19"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max_eval_tokens", type=int, default=4096)
    parser.add_argument("--ppl_context_len", type=int, default=512)
    parser.add_argument("--cached_ppl_context_len", type=int, default=512)
    parser.add_argument("--cached_ppl_eval_tokens", type=int, default=256)
    parser.add_argument("--enable_cached_ppl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speed_context_lens", type=str, default="128,512,1024")
    parser.add_argument("--gen_new_tokens", type=int, default=64)
    parser.add_argument("--speed_repeats", type=int, default=3)
    parser.add_argument("--speed_warmup_runs", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output_json", type=str, default="outputs/latest_metrics.json")
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name

def resolve_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_text(project_root: Path, dataset: str, split: str) -> str:
    if dataset == "wikitext":
        split_path = project_root / "datasets" / "wikitext-103-raw-v1" / split
        ds = load_from_disk(str(split_path))
        lines = [x["text"] for x in ds if x["text"] and x["text"].strip()]
        return "\n".join(lines)

    txt_path = project_root / "datasets" / "pg19_samples" / f"{split}_1.txt"
    return txt_path.read_text(encoding="utf-8")


@torch.inference_mode()
def compute_ppl(model, input_ids: torch.Tensor, context_len: int):
    total_nll = 0.0
    total_tokens = 0

    for start in range(0, input_ids.shape[0] - 1, context_len):
        chunk = input_ids[start : start + context_len + 1]
        if chunk.shape[0] < 2:
            continue
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)
        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
        total_nll += float(loss.item())
        total_tokens += int(y.numel())

    ppl = math.exp(total_nll / max(total_tokens, 1))
    return {"ppl": ppl, "nll_sum": total_nll, "tokens": total_tokens}


def _sync_if_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def compute_cached_ppl(
    model,
    input_ids: torch.Tensor,
    context_len: int,
    eval_tokens: int,
    prefill_ctx_factory=None,
):
    if context_len < 1:
        raise ValueError("cached PPL context_len must be positive.")
    if eval_tokens < 1:
        raise ValueError("cached PPL eval_tokens must be positive.")
    if input_ids.shape[0] <= context_len:
        raise RuntimeError("Too few tokens for cached PPL evaluation.")

    actual_eval_tokens = min(eval_tokens, int(input_ids.shape[0] - context_len))
    prompt_ids = input_ids[:context_len].unsqueeze(0)
    target_ids = input_ids[context_len : context_len + actual_eval_tokens]
    prefill_ctx_factory = prefill_ctx_factory or (lambda _: nullcontext())

    with prefill_ctx_factory(model):
        out = model(input_ids=prompt_ids, use_cache=True)

    past_key_values = out.past_key_values
    logits = out.logits[:, -1, :]
    total_nll = 0.0
    total_tokens = 0

    for offset, target_id in enumerate(target_ids):
        target = target_id.reshape(1)
        loss = F.cross_entropy(logits, target, reduction="sum")
        total_nll += float(loss.item())
        total_tokens += 1

        if offset + 1 < actual_eval_tokens:
            current_position = torch.tensor([context_len + offset], device=input_ids.device, dtype=torch.long)
            out = model(
                input_ids=target.reshape(1, 1),
                past_key_values=past_key_values,
                cache_position=current_position,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]

    ppl = math.exp(total_nll / max(total_tokens, 1))
    return {
        "ppl": ppl,
        "nll_sum": total_nll,
        "tokens": total_tokens,
        "context_len": context_len,
        "requested_eval_tokens": eval_tokens,
    }


@torch.inference_mode()
def _speed_run(model, prompt_ids: torch.Tensor, new_tokens: int, device: str, prefill_ctx_factory=None):
    prompt_ids = prompt_ids.unsqueeze(0)
    prefill_ctx_factory = prefill_ctx_factory or (lambda _: nullcontext())

    _sync_if_cuda(device)
    t0 = time.perf_counter()
    with prefill_ctx_factory(model):
        out = model(input_ids=prompt_ids, use_cache=True)
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
    past_key_values = out.past_key_values
    _sync_if_cuda(device)
    t1 = time.perf_counter()
    ttft = t1 - t0

    generated = 1
    decode_start = time.perf_counter()
    for step in range(1, new_tokens):
        current_position = torch.tensor(
            [prompt_ids.shape[-1] + step - 1],
            device=prompt_ids.device,
            dtype=torch.long,
        )
        out = model(
            input_ids=next_token.reshape(1, 1),
            past_key_values=past_key_values,
            cache_position=current_position,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
        generated += 1
    _sync_if_cuda(device)
    t2 = time.perf_counter()

    decode_time = t2 - decode_start
    e2e_time = t2 - t0
    tpot = decode_time / max(generated - 1, 1)
    throughput = generated / max(e2e_time, 1e-8)

    return {
        "ttft_sec": ttft,
        "tpot_sec": tpot,
        "decode_sec": decode_time,
        "e2e_sec": e2e_time,
        "throughput_tok_per_sec": throughput,
        "generated_tokens": generated,
    }


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


@torch.inference_mode()
def speed_test(
    model,
    prompt_ids: torch.Tensor,
    new_tokens: int,
    device: str,
    prefill_ctx_factory=None,
    repeats: int = 3,
    warmup_runs: int = 1,
):
    if repeats < 1:
        raise ValueError("speed repeats must be positive.")
    if warmup_runs < 0:
        raise ValueError("speed warmup runs cannot be negative.")

    for _ in range(warmup_runs):
        _speed_run(model, prompt_ids, new_tokens, device, prefill_ctx_factory=prefill_ctx_factory)

    raw_runs = [
        _speed_run(model, prompt_ids, new_tokens, device, prefill_ctx_factory=prefill_ctx_factory)
        for _ in range(repeats)
    ]
    result = {
        "speed_repeats": repeats,
        "speed_warmup_runs": warmup_runs,
        "raw_runs": raw_runs,
        "generated_tokens": raw_runs[0]["generated_tokens"],
    }
    for key in ["ttft_sec", "tpot_sec", "decode_sec", "e2e_sec", "throughput_tok_per_sec"]:
        avg, std = _mean_and_std([float(run[key]) for run in raw_runs])
        result[key] = avg
        result[f"{key}_std"] = std
    return result


def summarize_kv_cache_stats(events: list[dict]) -> dict:
    if not events:
        return {"summary": {"num_events": 0}, "events": []}

    before_lens = [x["before_len"] for x in events]
    after_lens = [x["after_len"] for x in events]
    ratios = [x["compression_ratio_actual"] for x in events]
    by_before_len = {}
    for event in events:
        key = str(event["before_len"])
        item = by_before_len.setdefault(
            key,
            {
                "before_len": event["before_len"],
                "after_len_min": event["after_len"],
                "after_len_max": event["after_len"],
                "num_events": 0,
            },
        )
        item["after_len_min"] = min(item["after_len_min"], event["after_len"])
        item["after_len_max"] = max(item["after_len_max"], event["after_len"])
        item["num_events"] += 1

    return {
        "summary": {
            "num_events": len(events),
            "before_len_min": min(before_lens),
            "before_len_max": max(before_lens),
            "after_len_min": min(after_lens),
            "after_len_max": max(after_lens),
            "avg_compression_ratio_actual": sum(ratios) / len(ratios),
            "by_before_len": list(by_before_len.values()),
        },
        "events": events,
    }


@torch.inference_mode()
def compute_gqa_cache_stats(model, input_ids: torch.Tensor, context_len: int) -> dict | None:
    if not hasattr(model, "gpt_neox") or not hasattr(model.gpt_neox.layers[0].attention, "_gqa_kv_heads"):
        return None
    prompt_ids = input_ids[:context_len].unsqueeze(0)
    out = model(input_ids=prompt_ids, use_cache=True)
    layers = getattr(out.past_key_values, "layers", [])
    shapes = []
    for layer_idx, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            continue
        shapes.append(
            {
                "layer_idx": layer_idx,
                "key_shape": list(keys.shape),
                "value_shape": list(values.shape),
            }
        )
    if not shapes:
        return None

    query_heads = int(model.config.num_attention_heads)
    kv_heads = int(shapes[0]["key_shape"][1])
    return {
        "impl": "reduced_kv_grouped_attention",
        "num_layers": len(shapes),
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "kv_head_ratio": kv_heads / query_heads,
        "context_len": int(shapes[0]["key_shape"][2]),
        "head_dim": int(shapes[0]["key_shape"][3]),
        "sample_layer_shapes": shapes[:3],
    }


def build_eval_notes(args, device: str) -> list[str]:
    notes = [
        "ppl_metrics is chunked teacher-forced PPL and does not apply KVPress cache compression.",
        "cache_ppl_metrics uses cached autoregressive teacher forcing; KVPress compresses the prefill cache before scoring continuation tokens.",
        "Speed metrics use manual greedy prefill/decode with warmup and repeated runs; TTFT is prefill-to-first-token wall time, TPOT is decode time divided by generated tokens after the first token.",
    ]
    if "gqa" in args.mode:
        notes.append(
            f"GQA uses training-free reduced-KV grouped attention with {args.gqa_kv_heads} cached KV heads."
        )
    if args.mode == "kvpress":
        notes.append(
            f"KVPress uses method={args.kvpress_method}, compression_ratio={args.kvpress_compression_ratio}, n_sink={args.kvpress_n_sink}."
        )
    if device != "cuda" and "flash" in args.mode:
        notes.append("FlashAttention was requested but CUDA is unavailable, so attention falls back to SDPA.")
    return notes


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    model_path = (project_root / args.model_path).resolve()
    output_path = (project_root / args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    attn_impl = choose_attn_impl(args.mode, device)
    if attn_impl == "flash_attention_2" and device != "cuda":
        print("[warn] flash_attention_2 requested but CUDA unavailable; fallback to sdpa.")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype if device == "cuda" else torch.float32,
        attn_implementation=attn_impl,
    )
    model.to(device)
    model.eval()

    prefill_ctx_factory = None
    kv_cache_events = []
    if "gqa" in args.mode:
        enable_gqa(model, args.gqa_kv_heads)
    elif args.mode == "kvpress":
        press = build_press(
            method=args.kvpress_method,
            compression_ratio=args.kvpress_compression_ratio,
            n_sink=args.kvpress_n_sink,
        )
        prefill_ctx_factory = lambda m: kvpress_on_gptneox(m, press, stats=kv_cache_events)

    text = load_text(project_root, args.dataset, args.split)
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    token_ids = enc["input_ids"][0][: args.max_eval_tokens].to(device)

    if token_ids.shape[0] < 8:
        raise RuntimeError("Too few tokens for evaluation.")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    ppl_metrics = compute_ppl(model, token_ids, args.ppl_context_len)
    cache_ppl_metrics = None
    if args.enable_cached_ppl:
        cache_ppl_metrics = compute_cached_ppl(
            model,
            token_ids,
            args.cached_ppl_context_len,
            args.cached_ppl_eval_tokens,
            prefill_ctx_factory=prefill_ctx_factory,
        )
    gqa_cache_stats = (
        compute_gqa_cache_stats(model, token_ids, args.cached_ppl_context_len) if "gqa" in args.mode else None
    )

    speed_results = []
    for s in [int(x.strip()) for x in args.speed_context_lens.split(",") if x.strip()]:
        if s + 2 > token_ids.shape[0]:
            continue
        speed_results.append(
            {
                "context_len": s,
                **speed_test(
                    model,
                    token_ids[:s],
                    args.gen_new_tokens,
                    device,
                    prefill_ctx_factory=prefill_ctx_factory,
                    repeats=args.speed_repeats,
                    warmup_runs=args.speed_warmup_runs,
                ),
            }
        )

    result = {
        "mode": args.mode,
        "dataset": args.dataset,
        "split": args.split,
        "device": device,
        "dtype": args.dtype,
        "attn_implementation": attn_impl,
        "gqa_impl": "reduced_kv_grouped_attention" if "gqa" in args.mode else None,
        "gqa_num_heads": int(model.config.num_attention_heads) if "gqa" in args.mode else None,
        "gqa_kv_heads": args.gqa_kv_heads if "gqa" in args.mode else None,
        "gqa_kv_head_ratio": (
            args.gqa_kv_heads / int(model.config.num_attention_heads) if "gqa" in args.mode else None
        ),
        "kvpress_method": args.kvpress_method if args.mode == "kvpress" else None,
        "kvpress_compression_ratio": args.kvpress_compression_ratio if args.mode == "kvpress" else None,
        "kvpress_n_sink": args.kvpress_n_sink if args.mode == "kvpress" else None,
        "ppl_metrics": ppl_metrics,
        "cache_ppl_metrics": cache_ppl_metrics,
        "speed_metrics": speed_results,
        "speed_repeats": args.speed_repeats,
        "speed_warmup_runs": args.speed_warmup_runs,
        "gqa_cache_stats": gqa_cache_stats,
        "kv_cache_stats": summarize_kv_cache_stats(kv_cache_events),
        "peak_memory_mb": (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device == "cuda" else None,
        "eval_notes": build_eval_notes(args, device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
