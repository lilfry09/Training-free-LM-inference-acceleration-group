import torch

from layerwise_compression import (
    CompressionConfig,
    build_layer_keep_ratios,
    cache_token_count,
    compress_past_key_values,
    format_schedule,
)


def _fake_cache(num_layers: int = 6, seq_len: int = 10):
    return tuple(
        (
            torch.arange(1 * 2 * seq_len * 4, dtype=torch.float32).reshape(1, 2, seq_len, 4),
            torch.ones(1, 2, seq_len, 4) * layer_idx,
        )
        for layer_idx in range(num_layers)
    )


def test_build_layer_keep_ratios_uses_shallow_middle_deep_schedule():
    ratios = build_layer_keep_ratios(6, CompressionConfig())

    assert ratios == [0.8, 0.8, 0.5, 0.5, 0.7, 0.7]


def test_compress_past_key_values_keeps_prefix_and_recent_tokens():
    cache = _fake_cache(num_layers=1, seq_len=10)

    compressed = compress_past_key_values(
        cache,
        [0.5],
        keep_initial_tokens=1,
        min_tokens=1,
    )

    key, value = compressed[0]
    assert key.shape == (1, 2, 5, 4)
    assert value.shape == (1, 2, 5, 4)
    assert torch.equal(key[:, :, 0, :], cache[0][0][:, :, 0, :])
    assert torch.equal(key[:, :, 1:, :], cache[0][0][:, :, -4:, :])


def test_cache_token_count_sums_variable_layer_lengths():
    cache = _fake_cache(num_layers=2, seq_len=10)
    compressed = compress_past_key_values(cache, [0.5, 0.8])

    assert cache_token_count(compressed) == 13


def test_format_schedule_is_compact():
    schedule = [{"layer": 0, "keep_ratio": 0.8}, {"layer": 1, "keep_ratio": 0.5}]

    assert format_schedule(schedule) == "L0=0.80, L1=0.50"
