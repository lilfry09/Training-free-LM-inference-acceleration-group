"""
Layer-wise adaptive KV cache compression.

This module implements a training-free sliding-window compressor for HuggingFace
causal language models.  It does not patch model internals; instead it runs an
explicit autoregressive loop and compresses `past_key_values` after each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CompressionConfig:
    """Layer-wise keep ratios for shallow, middle, and deep transformer blocks."""

    shallow_ratio: float = 0.8
    middle_ratio: float = 0.5
    deep_ratio: float = 0.7
    shallow_boundary: float = 0.3
    deep_boundary: float = 0.7
    keep_initial_tokens: int = 1
    min_tokens: int = 1

    def validate(self) -> None:
        values = (
            self.shallow_ratio,
            self.middle_ratio,
            self.deep_ratio,
            self.shallow_boundary,
            self.deep_boundary,
        )
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("ratios and boundaries must be in [0, 1]")
        if self.shallow_boundary >= self.deep_boundary:
            raise ValueError("shallow_boundary must be smaller than deep_boundary")
        if self.keep_initial_tokens < 0 or self.min_tokens < 1:
            raise ValueError("keep_initial_tokens must be >= 0 and min_tokens must be >= 1")


def layer_keep_ratio(layer_idx: int, total_layers: int, config: CompressionConfig) -> float:
    """Return the keep ratio assigned to one transformer layer."""

    config.validate()
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if layer_idx < 0 or layer_idx >= total_layers:
        raise ValueError(f"layer_idx must be in [0, {total_layers})")

    shallow_end = max(1, round(config.shallow_boundary * total_layers))
    deep_start = max(shallow_end + 1, round(config.deep_boundary * total_layers))
    deep_start = min(deep_start, total_layers - 1)

    if layer_idx < shallow_end:
        return config.shallow_ratio
    if layer_idx < deep_start:
        return config.middle_ratio
    return config.deep_ratio


def build_layer_keep_ratios(total_layers: int, config: CompressionConfig) -> list[float]:
    """Build the per-layer keep-ratio schedule."""

    return [layer_keep_ratio(i, total_layers, config) for i in range(total_layers)]


def infer_num_layers(model: Any) -> int:
    """Infer the decoder layer count for common HuggingFace model families."""

    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)

    candidate_paths = (
        ("gpt_neox", "layers"),
        ("model", "layers"),
        ("transformer", "h"),
    )
    for first, second in candidate_paths:
        block = getattr(model, first, None)
        layers = getattr(block, second, None)
        if layers is not None:
            return len(layers)

    raise ValueError("Could not infer layer count from model")


def _to_legacy_cache(past_key_values: Any) -> tuple[Any, bool, Any]:
    """Convert new HF Cache objects to legacy tuples when possible."""

    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache(), True, past_key_values.__class__
    return past_key_values, False, None


def _from_legacy_cache(cache: tuple[Any, ...], was_cache_object: bool, cache_cls: Any) -> Any:
    """Restore a cache object if the model originally returned one."""

    if was_cache_object and hasattr(cache_cls, "from_legacy_cache"):
        return cache_cls.from_legacy_cache(cache)
    return cache


def _slice_tokens(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select cache tokens along the sequence dimension."""

    return tensor.index_select(dim=2, index=indices)


def _keep_indices(seq_len: int, keep_len: int, keep_initial_tokens: int, device: torch.device) -> torch.Tensor:
    """Keep a small prefix plus the most recent tokens."""

    keep_len = min(seq_len, max(1, keep_len))
    prefix_len = min(keep_initial_tokens, keep_len, seq_len)
    tail_len = keep_len - prefix_len

    pieces = []
    if prefix_len:
        pieces.append(torch.arange(prefix_len, device=device))
    if tail_len:
        tail_start = max(prefix_len, seq_len - tail_len)
        pieces.append(torch.arange(tail_start, seq_len, device=device))

    if not pieces:
        return torch.empty(0, dtype=torch.long, device=device)

    return torch.cat(pieces).unique(sorted=True)


def compress_past_key_values(
    past_key_values: Any,
    keep_ratios: Sequence[float],
    *,
    keep_initial_tokens: int = 1,
    min_tokens: int = 1,
) -> Any:
    """Compress a HuggingFace `past_key_values` object or legacy tuple.

    The compressor keeps a prefix token (useful as an anchor) and the newest
    tokens.  It preserves layer order and tensor dtype/device.
    """

    if past_key_values is None:
        return None

    legacy_cache, was_cache_object, cache_cls = _to_legacy_cache(past_key_values)
    compressed_layers = []

    for layer_idx, layer_cache in enumerate(legacy_cache):
        if len(layer_cache) < 2:
            compressed_layers.append(layer_cache)
            continue

        key, value, *rest = layer_cache
        ratio = keep_ratios[layer_idx] if layer_idx < len(keep_ratios) else keep_ratios[-1]
        if ratio <= 0:
            keep_len = min_tokens
        else:
            keep_len = max(min_tokens, ceil(key.shape[2] * ratio))

        if keep_len >= key.shape[2]:
            compressed_layers.append(layer_cache)
            continue

        indices = _keep_indices(
            seq_len=key.shape[2],
            keep_len=keep_len,
            keep_initial_tokens=keep_initial_tokens,
            device=key.device,
        )
        compressed_layers.append((_slice_tokens(key, indices), _slice_tokens(value, indices), *rest))

    return _from_legacy_cache(tuple(compressed_layers), was_cache_object, cache_cls)


def cache_token_count(past_key_values: Any) -> int:
    """Return the total number of cached token slots across layers."""

    if past_key_values is None:
        return 0

    legacy_cache, _, _ = _to_legacy_cache(past_key_values)
    return sum(layer_cache[0].shape[2] for layer_cache in legacy_cache if len(layer_cache) >= 1)


class LayerWiseKVCompressor:
    """Run generation and scoring with optional layer-wise KV compression."""

    def __init__(self, model: Any, config: CompressionConfig | None = None):
        self.model = model
        self.config = config or CompressionConfig()
        self.total_layers = infer_num_layers(model)
        self.keep_ratios = build_layer_keep_ratios(self.total_layers, self.config)

    @classmethod
    def uniform(cls, model: Any, keep_ratio: float) -> "LayerWiseKVCompressor":
        config = CompressionConfig(
            shallow_ratio=keep_ratio,
            middle_ratio=keep_ratio,
            deep_ratio=keep_ratio,
        )
        return cls(model, config)

    def describe_schedule(self) -> list[dict[str, float | int]]:
        return [
            {"layer": idx, "keep_ratio": ratio}
            for idx, ratio in enumerate(self.keep_ratios)
        ]

    def _compress(self, past_key_values: Any) -> Any:
        return compress_past_key_values(
            past_key_values,
            self.keep_ratios,
            keep_initial_tokens=self.config.keep_initial_tokens,
            min_tokens=self.config.min_tokens,
        )

    def _forward_step(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any,
        absolute_position: int,
    ) -> Any:
        if input_ids.shape[1] == 1:
            position_ids = torch.full(
                (input_ids.shape[0], 1),
                absolute_position,
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            position_ids = torch.arange(
                absolute_position,
                absolute_position + input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

        return self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 50,
        eos_token_id: int | None = None,
        compress: bool = True,
        compress_after_prefill_only: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Greedy generation with measured KV-cache retention statistics."""

        self.model.eval()
        generated = input_ids.clone()
        past_key_values = None
        retained_token_slots: list[int] = []

        for step in range(max_new_tokens):
            if past_key_values is None:
                current_input = generated
                absolute_position = 0
            else:
                current_input = generated[:, -1:]
                absolute_position = generated.shape[1] - 1

            outputs = self._forward_step(current_input, past_key_values, absolute_position)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            past_key_values = outputs.past_key_values
            should_compress = compress and (not compress_after_prefill_only or step == 0)
            if should_compress:
                past_key_values = self._compress(past_key_values)
            retained_token_slots.append(cache_token_count(past_key_values))

            if eos_token_id is not None and torch.all(next_token.eq(eos_token_id)):
                break

        stats = {
            "avg_cache_tokens_per_layer": (
                sum(retained_token_slots) / len(retained_token_slots) / self.total_layers
                if retained_token_slots
                else 0.0
            ),
            "generated_tokens": float(generated.shape[1] - input_ids.shape[1]),
        }
        return generated, stats

    @torch.no_grad()
    def perplexity(
        self,
        input_ids: torch.Tensor,
        *,
        compress: bool = True,
        max_tokens: int | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute autoregressive perplexity under the same cache policy."""

        self.model.eval()
        if input_ids.shape[1] < 2:
            raise ValueError("perplexity requires at least two tokens")

        if max_tokens is not None:
            input_ids = input_ids[:, :max_tokens]

        past_key_values = None
        losses = []
        retained_token_slots: list[int] = []

        for idx in range(input_ids.shape[1] - 1):
            current_input = input_ids[:, idx : idx + 1]
            outputs = self._forward_step(current_input, past_key_values, idx)
            target = input_ids[:, idx + 1]
            losses.append(F.cross_entropy(outputs.logits[:, -1, :], target, reduction="none"))

            past_key_values = outputs.past_key_values
            if compress:
                past_key_values = self._compress(past_key_values)
            retained_token_slots.append(cache_token_count(past_key_values))

        mean_nll = torch.cat(losses).mean()
        stats = {
            "avg_cache_tokens_per_layer": (
                sum(retained_token_slots) / len(retained_token_slots) / self.total_layers
                if retained_token_slots
                else 0.0
            ),
            "scored_tokens": float(input_ids.shape[1] - 1),
        }
        return float(torch.exp(mean_nll).item()), stats


def format_schedule(schedule: Iterable[dict[str, float | int]]) -> str:
    """Render a compact schedule string for logs and README snippets."""

    return ", ".join(f"L{item['layer']}={item['keep_ratio']:.2f}" for item in schedule)
