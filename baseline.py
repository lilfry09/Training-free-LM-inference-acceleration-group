import argparse
import os
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "EleutherAI/pythia-70m"
PROMPT = (
    "Efficient language model inference reduces memory usage and latency by "
    "compressing the key-value cache while preserving generation quality."
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline generation test for Pythia-70M.")
    parser.add_argument("--model", default=MODEL_NAME, help="Hugging Face model name or local path.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Number of tokens to generate.")
    args = parser.parse_args()

    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    process = psutil.Process(os.getpid())
    rss_before_mb = process.memory_info().rss / 1024**2

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    rss_after_mb = process.memory_info().rss / 1024**2

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    new_tokens = output_ids.shape[-1] - inputs["input_ids"].shape[-1]
    tokens_per_second = new_tokens / elapsed if elapsed > 0 else float("inf")

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        memory_text = f"{peak_memory_mb:.2f} MB"
    else:
        memory_text = "N/A (CUDA unavailable; running on CPU)"

    print("\n=== Prompt ===")
    print(PROMPT)
    print("\n=== Generated Text ===")
    print(generated)
    print("\n=== Metrics ===")
    print(f"elapsed_seconds: {elapsed:.4f}")
    print(f"new_tokens: {new_tokens}")
    print(f"tokens_per_second: {tokens_per_second:.2f}")
    print(f"peak_gpu_memory: {memory_text}")
    print(f"process_rss_before_mb: {rss_before_mb:.2f}")
    print(f"process_rss_after_mb: {rss_after_mb:.2f}")
    print(f"process_rss_delta_mb: {rss_after_mb - rss_before_mb:.2f}")


if __name__ == "__main__":
    main()
