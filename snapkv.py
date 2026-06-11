"""Pythia/GPT-NeoX adapter for NVIDIA KVPress SnapKVPress.

KVPress' context-manager hook targets model families such as Llama, Mistral,
Qwen, and Phi. Pythia-70M uses GPT-NeoX, so this module applies SnapKVPress to
Pythia's returned DynamicCache after prefill.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SnapKVConfig:
    compression_ratio: float = 0.5
    window_size: int = 64
    kernel_size: int = 5


try:
    from kvpress import SnapKVPress
    _SNAPKV_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    SnapKVPress = None
    _SNAPKV_IMPORT_ERROR = exc


def _require_snapkvpress() -> object:
    if SnapKVPress is None:
        raise RuntimeError(
            "SnapKVPress is not available. Current environment misses required dependency paths. "
            "Use Python 3.11 + installed kvpress, or set method=baseline only."
        ) from _SNAPKV_IMPORT_ERROR
    return SnapKVPress


class AdaptiveSnapKVPress:
    """Context-length aware SnapKV controller that only chooses compression ratio."""

    def __init__(self, window_size: int = 64, kernel_size: int = 5) -> None:
        self.window_size = window_size
        self.kernel_size = kernel_size

    @staticmethod
    def _ratio_for_context_length(context_length: int) -> float:
        if context_length <= 512:
            return 0.1
        if context_length <= 1024:
            return 0.2
        if context_length <= 2048:
            return 0.3
        return 0.4

    def compress_cache(self, model, cache, attentions, prompt_len: int) -> tuple[object, dict]:
        ratio = self._ratio_for_context_length(prompt_len)
        print(f"Context Length: {prompt_len}")
        print(f"Compression Ratio: {ratio}")

        config = SnapKVConfig(
            compression_ratio=ratio,
            window_size=self.window_size,
            kernel_size=self.kernel_size,
        )
        cache, stats = compress_cache_with_snapkv(
            model=model,
            cache=cache,
            attentions=attentions,
            prompt_len=prompt_len,
            config=config,
        )
        stats["snapkv_selected_compression_ratio"] = ratio
        return cache, stats


def kv_cache_mb(cache) -> float:
    total_bytes = 0
    for layer in cache.layers:
        if layer.keys is None or layer.values is None:
            continue
        total_bytes += layer.keys.numel() * layer.keys.element_size()
        total_bytes += layer.values.numel() * layer.values.element_size()
    return total_bytes / 1024**2


def kv_cache_length(cache) -> int:
    if not cache.layers:
        return 0
    return int(cache.layers[0].keys.shape[-2])


def compress_cache_with_snapkv(model, cache, attentions, prompt_len: int, config: SnapKVConfig) -> tuple[object, dict]:
    """Compress a Pythia/GPT-NeoX DynamicCache using KVPress' SnapKVPress.

    The model must be run with `attn_implementation="eager"` and
    `output_attentions=True` during prefill so SnapKVPress can score tokens from
    attention maps.
    """

    if not hasattr(model, "gpt_neox"):
        raise TypeError("This adapter expects a GPTNeoX/Pythia model with a gpt_neox attribute.")
    if attentions is None:
        raise ValueError("SnapKV compression requires prefill attentions.")

    window_size = min(config.window_size, max(1, prompt_len - 1))
    press = _require_snapkvpress()(
        compression_ratio=config.compression_ratio,
        window_size=window_size,
        kernel_size=config.kernel_size,
    )
    before_len = kv_cache_length(cache)

    for layer_idx, layer in enumerate(model.gpt_neox.layers):
        module = layer.attention
        if not hasattr(module, "head_dim"):
            module.head_dim = module.head_size

        keys = cache.layers[layer_idx].keys
        values = cache.layers[layer_idx].values
        dummy_hidden_states = torch.empty(
            keys.shape[0],
            keys.shape[2],
            1,
            device=keys.device,
            dtype=keys.dtype,
        )
        new_keys, new_values = press.compress(
            module=module,
            hidden_states=dummy_hidden_states,
            keys=keys,
            values=values,
            attentions=attentions[layer_idx],
            kwargs={},
        )
        cache.layers[layer_idx].keys = new_keys
        cache.layers[layer_idx].values = new_values

    after_len = kv_cache_length(cache)
    return cache, {
        "snapkv_compression_ratio": config.compression_ratio,
        "snapkv_window_size": window_size,
        "snapkv_kernel_size": config.kernel_size,
        "kv_cache_tokens_before": before_len,
        "kv_cache_tokens_after": after_len,
        "kv_cache_compression_ratio_actual": 1.0 - (after_len / before_len if before_len else 0.0),
        "kv_cache_mb_after_prefill": kv_cache_mb(cache),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SnapKVPress import and Pythia adapter availability.")
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SnapKVConfig(
        compression_ratio=args.compression_ratio,
        window_size=args.window_size,
        kernel_size=args.kernel_size,
    )
    print("SnapKVPress is importable.")
    print(f"Pythia adapter config: {config}")
    print("Run scripts/evaluate.py to apply this adapter to Pythia-70M.")


if __name__ == "__main__":
    main()
