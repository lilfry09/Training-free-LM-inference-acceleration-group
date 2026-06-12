# SnapKV Course Project

This project evaluates no-training KV cache compression for
`EleutherAI/pythia-70m` using NVIDIA KVPress SnapKV. It compares:

- `baseline`: full KV cache
- `snapkv`: fixed-ratio SnapKV compression
- `adaptive_snapkv`: context-length aware compression ratio selection

The experiments in this repository were run on CPU.

## Environment

Recommended environment:

- Python 3.11
- PyTorch CPU build
- `transformers`
- `datasets`
- `kvpress`
- `pandas`
- `psutil`
- `matplotlib`

Install dependencies:

```bash
pip install -r requirements.txt
```

If using conda:

```bash
conda activate snapkv_py311
```

## How to Run

Validate the SnapKV adapter:

```bash
python scripts/snapkv.py
```

Run the baseline demo:

```bash
python scripts/baseline.py
```

Run baseline and SnapKV evaluation:

```bash
python scripts/evaluate.py --datasets wikitext pg19 --methods baseline snapkv --snapkv-compression-ratio 0.3 --output results/metrics.csv
```

Run baseline, fixed SnapKV, and Adaptive SnapKV:

```bash
python scripts/evaluate.py --datasets wikitext pg19 --methods baseline snapkv adaptive_snapkv --snapkv-compression-ratio 0.3 --output results/metrics_adaptive.csv
```

Run a compression-ratio sweep:

```bash
python scripts/run_compression_sweep.py
```

Generate plots from saved CSV files:

```bash
python scripts/plot_results.py --inputs results/metrics_cr010_full.csv results/metrics_cr020_full.csv results/metrics_cr030_full.csv results/metrics_cr040_full.csv results/metrics_cr050_full.csv
```

Run the 8192-token context comparison used in the final summary:

```bash
python scripts/evaluate.py --methods baseline snapkv adaptive_snapkv --datasets pg19 --snapkv-compression-ratio 0.3 --ppl-prefill-tokens 8192 --max-ppl-tokens 8193 --generation-prompt-tokens 8192 --output results/metrics_8192_raw.csv
```

## Method

KVPress SnapKV does not directly hook into Pythia/GPT-NeoX in this setup, so
`scripts/snapkv.py` provides a small adapter:

1. run prefill with `use_cache=True`, `output_attentions=True`, and
   `attn_implementation="eager"`;
2. read Pythia's returned `DynamicCache` and attention maps;
3. call `SnapKVPress.compress()` layer by layer on cached keys and values;
4. continue perplexity evaluation or greedy generation from the compressed cache.

`adaptive_snapkv` does not change the SnapKV algorithm. It only selects the
compression ratio from the prefill context length:

| Context length | Compression ratio |
| ---: | ---: |
| `<= 512` | 0.1 |
| `<= 1024` | 0.2 |
| `<= 2048` | 0.3 |
| `> 2048` | 0.4 |

## Saved Results

Important result files:

- `results/metrics_baseline_snapkv_sweep_adaptive.csv`
- `results/metrics_cr010_full.csv` through `results/metrics_cr050_full.csv`
- `results/metrics_8192_context_summary.csv`

Important figures:

- `figures/compression_vs_memory.png`
- `figures/compression_vs_ppl.png`
- `figures/method_vs_kv_cache_memory.png`
- `figures/method_vs_tokens_per_second.png`
- `figures/method_vs_pg19_ppl.png`
- `figures/method_vs_wikitext_ppl.png`

## Short Report

### 128-token context sweep

The sweep compares baseline, fixed SnapKV compression from 10% to 50%, and
Adaptive SnapKV on WikiText and PG-19 samples.

| Dataset | Method | PPL | Tokens/s | KV Cache Memory MB |
| --- | --- | ---: | ---: | ---: |
| wikitext | baseline | 45.043467 | 91.243386 | 3.000000 |
| wikitext | snapkv(10%) | 47.191342 | 104.250216 | 2.695312 |
| wikitext | snapkv(20%) | 55.700569 | 82.386797 | 2.390625 |
| wikitext | snapkv(30%) | 63.827642 | 87.880002 | 2.085938 |
| wikitext | snapkv(40%) | 83.240339 | 87.130309 | 1.781250 |
| wikitext | snapkv(50%) | 122.399065 | 83.777244 | 1.500000 |
| wikitext | adaptive_snapkv | 47.191342 | 69.043727 | 2.695312 |
| pg19 | baseline | 51.417376 | 94.975626 | 3.000000 |
| pg19 | snapkv(10%) | 54.352853 | 95.509829 | 2.695312 |
| pg19 | snapkv(20%) | 66.011594 | 93.744406 | 2.390625 |
| pg19 | snapkv(30%) | 74.960806 | 99.256291 | 2.085938 |
| pg19 | snapkv(40%) | 99.510977 | 94.508146 | 1.781250 |
| pg19 | snapkv(50%) | 157.185674 | 85.928647 | 1.500000 |
| pg19 | adaptive_snapkv | 54.352853 | 97.718219 | 2.695312 |

At 128 context length, SnapKV reduces KV cache memory from 3.0 MB to 2.70,
2.39, 2.09, 1.78, and 1.50 MB as the compression ratio increases from 10% to
50%. The tradeoff is higher PPL at larger compression ratios. CPU throughput is
noisy, but PG-19 shows small speed gains for SnapKV at 10% and 30%.

### 8192-token context comparison

The long-context experiment compares baseline, SnapKV at 30%, and Adaptive
SnapKV on one PG-19 sample.

| Method | PPL | Tokens/s | KV Cache Memory MB | Compression Ratio |
| --- | ---: | ---: | ---: | ---: |
| baseline | 71.177773 | 3.384341 | 192.000000 | 0.000000 |
| snapkv30 | 71.177773 | 3.025041 | 134.390625 | 0.300049 |
| adaptive | 71.177773 | 3.941588 | 115.195312 | 0.400024 |

In this 8192-token context test, Adaptive SnapKV keeps the same measured PPL as
baseline while reducing KV cache memory from 192.0 MB to 115.2 MB, about a 40%
reduction. It also improves throughput from 3.38 tokens/s to 3.94 tokens/s,
about a 16.5% speedup over baseline. Fixed SnapKV at 30% also reduces memory to
134.4 MB, but it is slower than baseline in this CPU run.

## Figures

![Compression vs memory](figures/compression_vs_memory.png)
![Compression vs PPL](figures/compression_vs_ppl.png)
![Method vs KV Cache Memory](figures/method_vs_kv_cache_memory.png)
![Method vs Tokens/s](figures/method_vs_tokens_per_second.png)
![Method vs WikiText PPL](figures/method_vs_wikitext_ppl.png)
![Method vs PG-19 PPL](figures/method_vs_pg19_ppl.png)

## Project Layout

```text
data/
figures/
results/
scripts/
  baseline.py
  evaluate.py
  plot_results.py
  run_adaptive_comparison.py
  run_compression_sweep.py
  snapkv.py
README.md
requirements.txt
```
