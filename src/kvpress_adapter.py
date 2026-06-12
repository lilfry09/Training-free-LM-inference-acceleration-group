from __future__ import annotations

from contextlib import contextmanager

from kvpress import KnormPress, StreamingLLMPress
from kvpress.utils import extract_keys_and_values


def build_press(method: str, compression_ratio: float, n_sink: int = 4):
    method = method.lower()
    if method == "knorm":
        return KnormPress(compression_ratio=compression_ratio)
    if method == "streamingllm":
        return StreamingLLMPress(compression_ratio=compression_ratio, n_sink=n_sink)
    raise ValueError(f"Unsupported kvpress method: {method}")


@contextmanager
def kvpress_on_gptneox(model, press, stats=None):
    if not hasattr(model, "gpt_neox"):
        raise ValueError("kvpress_on_gptneox expects a GPTNeoX model (e.g., Pythia).")

    hooks = []

    def _hook(module, args, kwargs, output):
        hidden_states = args[0]
        layer_past = kwargs.get("layer_past")
        if layer_past is None:
            return output

        cache_position = kwargs.get("cache_position")
        q_len = hidden_states.shape[1]
        if cache_position is not None and cache_position[-1] > q_len:
            return output

        keys, values = extract_keys_and_values(layer_past, module.layer_idx)
        before_len = int(keys.shape[2])
        keys, values = press.compress(module, hidden_states, keys, values, output[1], kwargs)
        after_len = int(keys.shape[2])
        if stats is not None:
            stats.append(
                {
                    "layer_idx": int(module.layer_idx),
                    "before_len": before_len,
                    "after_len": after_len,
                    "compression_ratio_actual": 1.0 - (after_len / before_len if before_len else 0.0),
                }
            )
        layer_past.layers[module.layer_idx].keys = keys
        layer_past.layers[module.layer_idx].values = values
        return output

    try:
        for layer in model.gpt_neox.layers:
            if not hasattr(layer.attention, "head_dim") and hasattr(layer.attention, "head_size"):
                layer.attention.head_dim = layer.attention.head_size
            hooks.append(layer.attention.register_forward_hook(_hook, with_kwargs=True))
        yield
    finally:
        for h in hooks:
            h.remove()
