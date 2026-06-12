import types
from typing import Callable

import torch
import torch.nn.functional as F
from transformers.models.gpt_neox.modeling_gpt_neox import (
    FlashAttentionKwargs,
    apply_rotary_pos_emb,
)


def _kv_reduce_mean(x: torch.Tensor, kv_heads: int) -> torch.Tensor:
    batch, heads, seq_len, dim = x.shape
    if kv_heads <= 0 or kv_heads > heads:
        raise ValueError(f"kv_heads must be in [1, {heads}], got {kv_heads}")
    if heads % kv_heads != 0:
        raise ValueError(f"num_heads={heads} must be divisible by kv_heads={kv_heads}")
    group_size = heads // kv_heads
    x = x.contiguous().view(batch, kv_heads, group_size, seq_len, dim).mean(dim=2)
    return x


def _grouped_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    head_mask: torch.Tensor | None,
    output_attentions: bool | None,
):
    batch, query_heads, query_len, head_dim = query.shape
    kv_heads = key.shape[1]
    key_len = key.shape[-2]
    if query_heads % kv_heads != 0:
        raise ValueError(f"query_heads={query_heads} must be divisible by kv_heads={kv_heads}")

    if head_mask is None and not output_attentions:
        causal_mask = attention_mask[:, :, :, :key_len] if attention_mask is not None and attention_mask.ndim == 4 else None
        is_causal = query_len > 1 and causal_mask is None and getattr(module, "is_causal", True)
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_mask,
            dropout_p=0.0 if not module.training else module.attention_dropout,
            is_causal=is_causal,
            scale=module.scaling,
            enable_gqa=query_heads != kv_heads,
        )
        return attn_output.transpose(1, 2).contiguous(), None

    group_size = query_heads // kv_heads
    grouped_query = query.contiguous().view(batch, kv_heads, group_size, query_len, head_dim)
    scores = torch.einsum("bkgld,bksd->bkgls", grouped_query, key) * module.scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_len].unsqueeze(2)
        scores = scores + causal_mask

    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if head_mask is not None:
        flat_weights = weights.contiguous().view(batch, query_heads, query_len, key_len)
        flat_weights = flat_weights * head_mask
        weights = flat_weights.contiguous().view(batch, kv_heads, group_size, query_len, key_len)

    weights = F.dropout(weights, p=0.0 if not module.training else module.attention_dropout, training=module.training)
    attn_output = torch.einsum("bkgls,bksd->bkgld", weights, value)
    attn_output = attn_output.contiguous().view(batch, query_heads, query_len, head_dim).transpose(1, 2).contiguous()
    attn_weights = weights.contiguous().view(batch, query_heads, query_len, key_len) if output_attentions else None
    return attn_output, attn_weights


def _build_gqa_forward() -> Callable:
    def gqa_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        head_mask: torch.FloatTensor | None = None,
        layer_past=None,
        output_attentions: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: FlashAttentionKwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, 3 * self.head_size)

        qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        num_heads = query_states.shape[1]
        kv_heads = self._gqa_kv_heads
        key_states = _kv_reduce_mean(key_states, kv_heads)
        value_states = _kv_reduce_mean(value_states, kv_heads)

        if layer_past is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_ndims,
                "cache_position": cache_position,
            }
            key_states, value_states = layer_past.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = _grouped_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)
        return attn_output, attn_weights

    return gqa_forward


def enable_gqa(model, kv_heads: int) -> None:
    gqa_forward = _build_gqa_forward()
    for layer in model.gpt_neox.layers:
        attn = layer.attention
        heads = getattr(attn.config, "num_attention_heads", None) or attn.num_attention_heads
        if heads % kv_heads != 0:
            raise ValueError(f"num_heads={heads} must be divisible by kv_heads={kv_heads}")
        attn._gqa_kv_heads = kv_heads
        attn._gqa_impl = "reduced_kv_grouped_attention"
        attn.forward = types.MethodType(gqa_forward, attn)


def choose_attn_impl(mode: str, device: str) -> str:
    if "flash" not in mode:
        return "sdpa"
    if device != "cuda":
        return "sdpa"
    return "flash_attention_2"
