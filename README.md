# Training-Free LM Inference Acceleration Group Project

This repository is the group-part submission for training-free language-model
inference acceleration.  It integrates and evaluates several methods on
Pythia-70M without changing model weights:

- Baseline SDPA inference
- Training-free reduced-KV GQA conversion
- KVPress-Knorm cache compression
- KVPress-StreamingLLM cache compression
- Additional layer-wise sink-plus-recent KV compression prototype

The report is written with the provided NeurIPS 2025 template:
`paper.tex`, `paper.pdf`, and `neurips_2025.sty`.

## Quick Start

Run commands from this repository:

```powershell
cd D:\SJTUlearning\2026Spring\NLP\FinalProject\Training-free-LM-inference-acceleration-group
pip install -r requirements.txt
```

Prepare local assets if they are not already available:

```powershell
.\scripts\prepare_assets.ps1
```

This downloads `EleutherAI/pythia-70m`, WikiText-103, and one PG-19 sample
into local `models/` and `datasets/` directories.  These large assets are not
committed to Git.

## Reproduce Group Experiments

Run the full matrix:

```powershell
.\scripts\run_matrix.ps1
```

This evaluates:

- `baseline`
- `gqa` with 2 cached KV heads
- `kvpress knorm` with compression ratio 0.5
- `kvpress streamingllm` with compression ratio 0.5 and 4 sink tokens

on both WikiText and PG-19.  Outputs are written to:

- `outputs/comparison_quality.md`
- `outputs/comparison_speed.md`
- `outputs/comparison_quality.csv`
- `outputs/comparison_speed.csv`

Extra diagnostics:

```powershell
.\scripts\run_ablations.ps1
.\scripts\run_cache_ppl_sanity.ps1 -Dataset wikitext -Split test
.\scripts\run_qualitative_examples.ps1
```

## Layer-wise Compression Prototype

The additional layer-wise implementation is kept as a compact reproducible
prototype:

```powershell
python eval_quick.py --dataset pg19 --output_json results_pg19.json
python eval_quick.py --dataset wikitext --output_json results_wikitext.json
```

Default paths:

- Model: `..\finalproj\models\pythia-70m` or local `models\pythia-70m`
- PG-19 sample: `..\finalproj\datasets\pg19_samples\test_1.txt`
- WikiText split: `..\finalproj\datasets\wikitext-103-raw-v1\test`

## Main Results

Quality summary from `outputs/comparison_quality.md`:

| Dataset | Method | Chunked PPL | Cached PPL | KV events | Avg. removed |
|---|---|---:|---:|---:|---:|
| WikiText | Baseline | 63.72 | 13.09 | 0 | -- |
| WikiText | GQA-2KV | 2258.01 | 3659.00 | 0 | -- |
| WikiText | KVPress-Knorm | 63.72 | 52.44 | 78 | 0.50 |
| WikiText | KVPress-StreamingLLM | 63.72 | 61.30 | 78 | 0.50 |
| PG-19 | Baseline | 33.57 | 29.60 | 0 | -- |
| PG-19 | GQA-2KV | 816.70 | 776.88 | 0 | -- |
| PG-19 | KVPress-Knorm | 33.57 | 31.76 | 78 | 0.50 |
| PG-19 | KVPress-StreamingLLM | 33.57 | 29.49 | 78 | 0.50 |

Selected throughput observations from `outputs/comparison_speed.md`:

| Dataset | Context | Baseline tok/s | Best compressed tok/s | Best method |
|---|---:|---:|---:|---|
| WikiText | 128 | 50.87 | 68.83 | StreamingLLM |
| WikiText | 512 | 34.95 | 52.54 | StreamingLLM |
| WikiText | 1024 | 51.24 | 43.89 | Knorm |
| PG-19 | 128 | 74.31 | 66.41 | StreamingLLM |
| PG-19 | 512 | 60.64 | 58.85 | StreamingLLM |
| PG-19 | 1024 | 47.32 | 51.44 | StreamingLLM |

Layer-wise quick-test results:

| Dataset | Method | PPL | Tokens/s | Speedup | Cache reduction |
|---|---|---:|---:|---:|---:|
| PG-19 | Dense | 37.50 | 19.49 | 1.00x | 0.0% |
| PG-19 | LayerWise-80/50/70 | 39.25 | 15.19 | 0.78x | 28.4% |
| WikiText | Dense | 80.36 | 16.89 | 1.00x | 0.0% |
| WikiText | LayerWise-80/50/70 | 82.30 | 36.41 | 2.16x | 28.4% |

The conservative conclusion is that KV cache compression can improve speed in
some CPU settings, but gains depend on dataset, context length, compression
overhead, and quality tolerance.  Naive training-free GQA is an honest negative
result: it reduces cache size but severely damages PPL.

## Tests

```powershell
python -m py_compile eval_quick.py layerwise_compression.py test_layerwise_compression.py
python -m pytest -q
```

Current local status: `4 passed`.

## Group Contribution

Current submitted workload statement:

- Fu Ruoyu: method integration, reduced-KV GQA implementation, KVPress
  evaluation adapter, layer-wise compression prototype, experiment scripts,
  result analysis, README, and NeurIPS report.  Workload: 100%.
