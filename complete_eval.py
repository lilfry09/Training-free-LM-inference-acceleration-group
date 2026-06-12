"""
完整评估：包含PPL、TTFT、TPOT的可复现实验
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import json
import numpy as np


class LayerwiseKVCompressor:
    """Layer-wise KV压缩器"""

    def __init__(self, model, ratios):
        self.model = model
        self.ratios = ratios
        self.original_forward = {}

    def __enter__(self):
        try:
            layers = self.model.gpt_neox.layers
        except:
            layers = self.model.model.layers

        for idx, layer in enumerate(layers):
            self.original_forward[idx] = layer.forward
            layer.forward = self._make_compressed_forward(layer, idx)
        return self

    def __exit__(self, *args):
        try:
            layers = self.model.gpt_neox.layers
        except:
            layers = self.model.model.layers

        for idx, layer in enumerate(layers):
            if idx in self.original_forward:
                layer.forward = self.original_forward[idx]

    def _make_compressed_forward(self, layer, layer_idx):
        original_forward = self.original_forward[layer_idx]
        ratio = self.ratios[layer_idx]

        def compressed_forward(hidden_states, attention_mask=None, position_ids=None,
                              head_mask=None, use_cache=False, layer_past=None,
                              output_attentions=False, **kwargs):
            if layer_past is not None and len(layer_past) == 2:
                key, value = layer_past
                seq_len = key.shape[2]
                keep_len = max(1, int(seq_len * ratio))
                layer_past = (key[:, :, -keep_len:, :], value[:, :, -keep_len:, :])

            return original_forward(hidden_states, attention_mask=attention_mask,
                                  position_ids=position_ids, head_mask=head_mask,
                                  use_cache=use_cache, layer_past=layer_past,
                                  output_attentions=output_attentions, **kwargs)
        return compressed_forward


def evaluate_ppl(model, tokenizer, dataset_name="wikitext", split="test", max_samples=100):
    """评估困惑度"""
    print(f"  Loading {dataset_name}...")

    try:
        dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
    except:
        # 如果失败，用简单文本
        return None

    texts = []
    for item in dataset:
        if item['text'].strip():
            texts.append(item['text'])
        if len(texts) >= max_samples:
            break

    if not texts:
        return None

    # 合并文本
    text = " ".join(texts[:10])  # 只用前10个sample
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    encodings = {k: v.to(model.device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings, labels=encodings['input_ids'])
        loss = outputs.loss.item()

    ppl = torch.exp(torch.tensor(loss)).item()
    return ppl


def measure_ttft_tpot(model, tokenizer, prompt, max_new_tokens=50):
    """测量TTFT和TPOT"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 预热
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # 测量总时间
    torch.cuda.synchronize() if model.device.type == "cuda" else None
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=False, use_cache=True)

    torch.cuda.synchronize() if model.device.type == "cuda" else None
    total_time = time.time() - start_time

    # TTFT估算：生成1个token的时间
    torch.cuda.synchronize() if model.device.type == "cuda" else None
    start_time = time.time()

    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)

    torch.cuda.synchronize() if model.device.type == "cuda" else None
    ttft = time.time() - start_time

    # TPOT：剩余token的平均时间
    tpot = (total_time - ttft) / (max_new_tokens - 1) if max_new_tokens > 1 else 0

    return ttft, tpot, total_time


def run_complete_evaluation():
    """完整评估"""

    print("="*70)
    print("Complete Evaluation: PPL + TTFT + TPOT + Speedup")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # 加载模型
    print("\n[1/4] Loading model...")
    model_name = "EleutherAI/pythia-70m"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    print(f"✓ Pythia-70M loaded ({num_layers} layers)")

    # 测试prompt
    test_prompt = "The quick brown fox jumps over the lazy dog. " * 20

    results = []

    # ==========================================
    # 实验1: Dense
    # ==========================================
    print("\n[2/4] Experiment 1: Dense (Baseline)")

    # PPL
    print("  Measuring PPL...")
    ppl_dense = evaluate_ppl(model, tokenizer)

    # TTFT & TPOT
    print("  Measuring TTFT & TPOT...")
    ttft_dense, tpot_dense, time_dense = measure_ttft_tpot(model, tokenizer, test_prompt)

    ppl_str = f"{ppl_dense:.2f}" if ppl_dense is not None else "N/A"
    print(f"  PPL: {ppl_str}")
    print(f"  TTFT: {ttft_dense*1000:.2f}ms")
    print(f"  TPOT: {tpot_dense*1000:.2f}ms")
    print(f"  Total: {time_dense:.3f}s")

    results.append({
        "method": "Dense",
        "ppl": round(ppl_dense, 2) if ppl_dense else None,
        "ttft_ms": round(ttft_dense * 1000, 2),
        "tpot_ms": round(tpot_dense * 1000, 2),
        "time": round(time_dense, 3),
        "speedup": "1.00x"
    })

    # ==========================================
    # 实验2: Uniform-50%
    # ==========================================
    print("\n[3/4] Experiment 2: Uniform-50%")

    uniform_ratios = [0.5] * num_layers

    # PPL
    print("  Measuring PPL...")
    with LayerwiseKVCompressor(model, uniform_ratios):
        ppl_uniform = evaluate_ppl(model, tokenizer)

    # TTFT & TPOT
    print("  Measuring TTFT & TPOT...")
    with LayerwiseKVCompressor(model, uniform_ratios):
        ttft_uniform, tpot_uniform, time_uniform = measure_ttft_tpot(model, tokenizer, test_prompt)

    speedup_uniform = time_dense / time_uniform

    ppl_str = f"{ppl_uniform:.2f}" if ppl_uniform is not None else "N/A"
    print(f"  PPL: {ppl_str}")
    print(f"  TTFT: {ttft_uniform*1000:.2f}ms")
    print(f"  TPOT: {tpot_uniform*1000:.2f}ms")
    print(f"  Total: {time_uniform:.3f}s")
    print(f"  Speedup: {speedup_uniform:.2f}x")

    results.append({
        "method": "Uniform-50%",
        "ppl": round(ppl_uniform, 2) if ppl_uniform else None,
        "ttft_ms": round(ttft_uniform * 1000, 2),
        "tpot_ms": round(tpot_uniform * 1000, 2),
        "time": round(time_uniform, 3),
        "speedup": f"{speedup_uniform:.2f}x"
    })

    # ==========================================
    # 实验3: LayerWise
    # ==========================================
    print("\n[4/4] Experiment 3: LayerWise (Ours)")

    layerwise_ratios = []
    for i in range(num_layers):
        pos = i / num_layers
        if pos < 0.3:
            ratio = 0.8
        elif pos < 0.7:
            ratio = 0.5
        else:
            ratio = 0.7
        layerwise_ratios.append(ratio)

    print(f"  Ratios: {layerwise_ratios}")

    # PPL
    print("  Measuring PPL...")
    with LayerwiseKVCompressor(model, layerwise_ratios):
        ppl_layerwise = evaluate_ppl(model, tokenizer)

    # TTFT & TPOT
    print("  Measuring TTFT & TPOT...")
    with LayerwiseKVCompressor(model, layerwise_ratios):
        ttft_layerwise, tpot_layerwise, time_layerwise = measure_ttft_tpot(model, tokenizer, test_prompt)

    speedup_layerwise = time_dense / time_layerwise

    ppl_str = f"{ppl_layerwise:.2f}" if ppl_layerwise is not None else "N/A"
    print(f"  PPL: {ppl_str}")
    print(f"  TTFT: {ttft_layerwise*1000:.2f}ms")
    print(f"  TPOT: {tpot_layerwise*1000:.2f}ms")
    print(f"  Total: {time_layerwise:.3f}s")
    print(f"  Speedup: {speedup_layerwise:.2f}x")

    results.append({
        "method": "LayerWise",
        "ppl": round(ppl_layerwise, 2) if ppl_layerwise else None,
        "ttft_ms": round(ttft_layerwise * 1000, 2),
        "tpot_ms": round(tpot_layerwise * 1000, 2),
        "time": round(time_layerwise, 3),
        "speedup": f"{speedup_layerwise:.2f}x"
    })

    # 保存结果
    with open("results_reproducible.json", "w") as f:
        json.dump(results, f, indent=2)

    # 打印表格
    print("\n" + "="*70)
    print("COMPLETE RESULTS")
    print("="*70)
    print(f"{'Method':<15} {'PPL':<8} {'TTFT(ms)':<10} {'TPOT(ms)':<10} {'Time(s)':<10} {'Speedup':<10}")
    print("-"*70)
    for r in results:
        ppl_str = f"{r['ppl']:.2f}" if r['ppl'] else "N/A"
        print(f"{r['method']:<15} {ppl_str:<8} {r['ttft_ms']:<10} {r['tpot_ms']:<10} {r['time']:<10} {r['speedup']:<10}")

    print("\n" + "="*70)
    print("Markdown Table")
    print("="*70)
    print("| Method | PPL | TTFT(ms) | TPOT(ms) | Time(s) | Speedup |")
    print("|--------|-----|----------|----------|---------|---------|")
    for r in results:
        ppl_str = f"{r['ppl']:.2f}" if r['ppl'] else "N/A"
        print(f"| {r['method']} | {ppl_str} | {r['ttft_ms']} | {r['tpot_ms']} | {r['time']} | {r['speedup']} |")

    print("\n✅ Complete evaluation finished!")
    return results


if __name__ == "__main__":
    run_complete_evaluation()
