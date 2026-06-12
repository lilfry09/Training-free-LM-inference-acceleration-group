import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import compute_cached_ppl, compute_ppl, load_text, resolve_device, resolve_dtype
from methods import choose_attn_impl


def parse_args():
    parser = argparse.ArgumentParser(description="Check cached PPL against same-slice teacher-forced PPL.")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--model_path", type=str, default="models/pythia-70m")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "pg19"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max_eval_tokens", type=int, default=4096)
    parser.add_argument("--ppl_context_len", type=int, default=512)
    parser.add_argument("--slice_context_len", type=int, default=512)
    parser.add_argument("--slice_eval_tokens", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output_json", type=str, default="outputs/cache_ppl_sanity_wikitext_test.json")
    parser.add_argument("--output_md", type=str, default="outputs/cache_ppl_sanity_wikitext_test.md")
    return parser.parse_args()


@torch.inference_mode()
def compute_same_slice_teacher_forced_ppl(
    model,
    input_ids: torch.Tensor,
    context_len: int,
    eval_tokens: int,
) -> dict:
    if context_len < 1:
        raise ValueError("slice context_len must be positive.")
    if eval_tokens < 1:
        raise ValueError("slice eval_tokens must be positive.")
    if input_ids.shape[0] <= context_len:
        raise RuntimeError("Too few tokens for same-slice PPL evaluation.")

    actual_eval_tokens = min(eval_tokens, int(input_ids.shape[0] - context_len))
    x = input_ids[: context_len + actual_eval_tokens - 1].unsqueeze(0)
    y = input_ids[1 : context_len + actual_eval_tokens].unsqueeze(0)
    logits = model(input_ids=x).logits

    start = context_len - 1
    end = start + actual_eval_tokens
    continuation_logits = logits[:, start:end, :]
    continuation_targets = y[:, start:end]
    loss = F.cross_entropy(
        continuation_logits.reshape(-1, continuation_logits.shape[-1]),
        continuation_targets.reshape(-1),
        reduction="sum",
    )
    nll_sum = float(loss.item())
    ppl = math.exp(nll_sum / max(actual_eval_tokens, 1))
    return {
        "ppl": ppl,
        "nll_sum": nll_sum,
        "tokens": actual_eval_tokens,
        "context_len": context_len,
        "requested_eval_tokens": eval_tokens,
        "target_token_start": context_len,
        "target_token_end_exclusive": context_len + actual_eval_tokens,
    }


def write_markdown(path: Path, result: dict) -> None:
    exact = result["exact_ppl_metrics"]
    same_slice = result["same_slice_teacher_forced_ppl_metrics"]
    cached = result["cache_ppl_metrics"]
    rows = [
        ("Exact teacher-forced PPL", exact),
        ("Same-slice teacher-forced PPL", same_slice),
        ("Same-slice cached PPL", cached),
    ]
    lines = [
        "# Cached PPL Sanity Check",
        "",
        f"- Dataset: `{result['dataset']}` `{result['split']}`",
        f"- Slice: {same_slice['context_len']} context tokens + {same_slice['tokens']} continuation tokens",
        "",
        "| metric | ppl | nll_sum | tokens |",
        "| --- | --- | --- | --- |",
    ]
    for label, metrics in rows:
        lines.append(
            f"| {label} | {metrics['ppl']:.6f} | {metrics['nll_sum']:.6f} | {metrics['tokens']} |"
        )
    lines.extend(
        [
            "",
            "The same-slice teacher-forced PPL uses the same continuation tokens as cached PPL.",
            "Therefore this check isolates slice choice from cache-based scoring.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    model_path = (project_root / args.model_path).resolve()
    output_json = (project_root / args.output_json).resolve()
    output_md = (project_root / args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    attn_impl = choose_attn_impl("baseline", device)

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

    text = load_text(project_root, args.dataset, args.split)
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    token_ids = enc["input_ids"][0][: args.max_eval_tokens].to(device)

    exact_ppl_metrics = compute_ppl(model, token_ids, args.ppl_context_len)
    same_slice_metrics = compute_same_slice_teacher_forced_ppl(
        model,
        token_ids,
        args.slice_context_len,
        args.slice_eval_tokens,
    )
    cache_ppl_metrics = compute_cached_ppl(
        model,
        token_ids,
        args.slice_context_len,
        args.slice_eval_tokens,
    )

    result = {
        "dataset": args.dataset,
        "split": args.split,
        "device": device,
        "dtype": args.dtype,
        "attn_implementation": attn_impl,
        "exact_ppl_metrics": exact_ppl_metrics,
        "same_slice_teacher_forced_ppl_metrics": same_slice_metrics,
        "cache_ppl_metrics": cache_ppl_metrics,
        "absolute_ppl_difference_same_slice": abs(same_slice_metrics["ppl"] - cache_ppl_metrics["ppl"]),
        "eval_notes": [
            "Exact PPL covers the full max_eval_tokens window.",
            "Same-slice teacher-forced PPL and cached PPL score the identical continuation tokens.",
            "A small same-slice difference supports using cached PPL as a cache-scoring metric, but cached PPL is not directly comparable to full-window exact PPL.",
        ],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_md, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[saved] {output_json}")
    print(f"[saved] {output_md}")


if __name__ == "__main__":
    main()
