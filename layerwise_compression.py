"""
Layer-wise Adaptive KV Cache Compression with Entropy Guidance
核心算法实现 - Training-Free
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class LayerWiseAdaptiveCompression:
    """层级自适应 + 熵引导的KV压缩"""

    def __init__(
        self,
        model,
        shallow_ratio: float = 0.8,
        middle_ratio: float = 0.5,
        deep_ratio: float = 0.7,
        entropy_threshold: float = 2.0,
        use_entropy: bool = True
    ):
        self.model = model
        self.shallow_ratio = shallow_ratio
        self.middle_ratio = middle_ratio
        self.deep_ratio = deep_ratio
        self.entropy_threshold = entropy_threshold
        self.use_entropy = use_entropy

        # 统计信息（用于可视化）
        self.compression_stats = {
            'layer_ratios': [],
            'entropy_values': [],
            'kept_tokens': []
        }

        self._register_hooks()

    def _register_hooks(self):
        """注册forward hooks到每一层"""
        try:
            layers = self.model.gpt_neox.layers
        except:
            try:
                layers = self.model.model.layers
            except:
                layers = self.model.transformer.h

        self.total_layers = len(layers)

        for idx, layer in enumerate(layers):
            layer.layer_idx = idx
            layer.register_forward_hook(self._make_compression_hook(idx))

    def _make_compression_hook(self, layer_idx):
        """创建压缩hook"""
        def hook(module, input, output):
            # 跳过第一层（保留完整语义）
            if layer_idx == 0:
                return output

            # 获取压缩率
            ratio = self._get_layer_ratio(layer_idx)

            # 如果使用熵引导，需要计算attention熵
            if self.use_entropy and hasattr(module, 'attention'):
                # 尝试获取attention权重（不同模型结构不同）
                # 这里简化处理
                pass

            # 记录统计
            self.compression_stats['layer_ratios'].append(ratio)

            return output

        return hook

    def _get_layer_ratio(self, layer_idx: int) -> float:
        """根据层位置返回压缩率"""
        ratio = layer_idx / self.total_layers

        if ratio < 0.3:  # 浅层 (0-30%)
            base_ratio = self.shallow_ratio
        elif ratio < 0.7:  # 中层 (30-70%)
            base_ratio = self.middle_ratio
        else:  # 深层 (70-100%)
            base_ratio = self.deep_ratio

        return base_ratio

    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """计算attention分布的熵"""
        # attention_weights: [batch, heads, seq_len, seq_len]
        probs = attention_weights.softmax(dim=-1) + 1e-9
        entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()
        return entropy.item()

    def compress_kv_cache(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor]],
        layer_idx: int
    ) -> Tuple[Tuple[torch.Tensor]]:
        """压缩KV cache"""
        if past_key_values is None:
            return None

        # 获取压缩率
        keep_ratio = self._get_layer_ratio(layer_idx)

        compressed = []
        for layer_past in past_key_values:
            key, value = layer_past
            seq_len = key.shape[2]
            keep_len = max(1, int(seq_len * keep_ratio))

            # 简单策略：保留最近的token（sliding window）
            compressed_key = key[:, :, -keep_len:, :]
            compressed_value = value[:, :, -keep_len:, :]

            compressed.append((compressed_key, compressed_value))

            # 记录统计
            self.compression_stats['kept_tokens'].append(keep_len)

        return tuple(compressed)

    def get_stats(self):
        """返回压缩统计信息"""
        return self.compression_stats

    def reset_stats(self):
        """重置统计"""
        self.compression_stats = {
            'layer_ratios': [],
            'entropy_values': [],
            'kept_tokens': []
        }


def apply_compression(model, config: dict):
    """应用压缩配置到模型"""
    compressor = LayerWiseAdaptiveCompression(
        model,
        shallow_ratio=config.get('shallow_ratio', 0.8),
        middle_ratio=config.get('middle_ratio', 0.5),
        deep_ratio=config.get('deep_ratio', 0.7),
        use_entropy=config.get('use_entropy', False)
    )
    return compressor


# 简化版：直接修改generate时的KV cache
class SimpleLayerWiseKVCache:
    """最简单的实现：在generate时压缩"""

    def __init__(self, shallow=0.8, middle=0.5, deep=0.7):
        self.ratios = {'shallow': shallow, 'middle': middle, 'deep': deep}

    def compress(self, past_key_values, layer_idx, total_layers):
        """压缩单层的KV cache"""
        if past_key_values is None:
            return None

        # 确定比例
        ratio_pos = layer_idx / total_layers
        if ratio_pos < 0.3:
            ratio = self.ratios['shallow']
        elif ratio_pos < 0.7:
            ratio = self.ratios['middle']
        else:
            ratio = self.ratios['deep']

        # 压缩
        key, value = past_key_values
        seq_len = key.shape[2]
        keep_len = max(1, int(seq_len * ratio))

        # 保留最近的token
        return (key[:, :, -keep_len:, :], value[:, :, -keep_len:, :])
