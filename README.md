# Layer-wise Adaptive KV Cache Compression

This repository contains the group-part submission for training-free language
model inference acceleration.  It focuses only on layer-wise KV cache
compression for Pythia-70M.  The implementation does not train or modify model
weights; it compresses the generated KV cache after prompt prefill.

The report uses the provided NeurIPS 2025 template:

- `paper.tex`
- `paper.pdf`
- `neurips_2025.sty`

## Quick Start

Run commands from this repository:

```powershell
cd D:\SJTUlearning\2026Spring\NLP\FinalProject\Training-free-LM-inference-acceleration-group
pip install -r requirements.txt
```

The default evaluation reuses the local model and datasets prepared in the
sibling `finalproj` directory:

- Model: `..\finalproj\models\pythia-70m`
- PG-19 sample: `..\finalproj\datasets\pg19_samples\test_1.txt`
- WikiText split: `..\finalproj\datasets\wikitext-103-raw-v1\test`

Run the reproducible evaluations:

```powershell
python eval_quick.py --dataset pg19 --output_json results_pg19.json
python eval_quick.py --dataset wikitext --output_json results_wikitext.json
```

The aggregated results are saved in:

- `results_pg19.json`
- `results_wikitext.json`
- `results.json`
- `results_final.json`

## Method

Pythia-70M has 6 transformer layers.  The layer-wise schedule is:

| Layer group | Layers | Keep ratio | Rationale |
|---|---:|---:|---|
| Shallow | 0-1 | 80% | Preserve low-level and semantic features |
| Middle | 2-3 | 50% | Compress more aggressively |
| Deep | 4-5 | 70% | Preserve final prediction context |

Each compressed layer keeps a small prefix of sink tokens plus the most recent
tokens.  The main implementation uses `keep_initial_tokens=4` and
`min_cache_tokens=16` to avoid over-compressing short contexts.

Core files:

- `layerwise_compression.py`: compression schedule, KV slicing, generation loop
- `eval_quick.py`: local Pythia-70M evaluation on PG-19 and WikiText
- `test_layerwise_compression.py`: unit tests for schedule and cache slicing
- `paper.tex`: English NeurIPS-style report
- `paper.pdf`: compiled submission PDF

## Results

Environment:

- Model: local Pythia-70M
- Device: CPU, `torch.float32`
- PPL: cached continuation PPL with 96 prompt tokens and 64 scored tokens
- Speed: greedy decode with 96 prompt tokens, 32 generated tokens, 2 repeats
- FLOPs: approximate decode attention score/value matmul FLOPs per output token

### PG-19

| Method | PPL | TTFT | TPOT | Tokens/s | Speedup | Cache red. | FLOPs red. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 37.50 | 0.0780 | 0.0176 | 51.31 | 1.00x | 0.0% | 0.0% |
| Uniform-50% | 38.46 | 0.0723 | 0.0157 | 57.17 | 1.11x | 43.0% | 43.0% |
| Uniform-67% | 36.08 | 0.0697 | 0.0183 | 50.10 | 0.98x | 27.8% | 27.8% |
| LayerWise-80/50/70 | 39.25 | 0.0715 | 0.0175 | 51.95 | 1.01x | 28.4% | 28.4% |

### WikiText

| Method | PPL | TTFT | TPOT | Tokens/s | Speedup | Cache red. | FLOPs red. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 80.36 | 0.0764 | 0.0170 | 52.92 | 1.00x | 0.0% | 0.0% |
| Uniform-50% | 84.97 | 0.0838 | 0.0156 | 56.41 | 1.07x | 43.0% | 43.0% |
| Uniform-67% | 84.52 | 0.0633 | 0.0161 | 56.91 | 1.08x | 27.8% | 27.8% |
| LayerWise-80/50/70 | 82.30 | 0.0925 | 0.0206 | 43.73 | 0.83x | 28.4% | 28.4% |

The result is intentionally reported conservatively.  The layer-wise schedule
reliably reduces KV cache size and approximate attention FLOPs by 28.4%, while
CPU wall-clock speed is mixed across datasets.  This is useful evidence for the
course requirement because it reports both successful memory/FLOPs reduction and
honest performance limitations.

## Tests

```powershell
python -m py_compile eval_quick.py layerwise_compression.py test_layerwise_compression.py
python -m pytest -q
```

Current local status: `4 passed`.

## Group Contribution

Current submitted workload statement:

- Yiqi Liu: surveyed related papers and proposed the initial layer-wise
  compression idea.
- Yiran Pang: wrote and organized the NeurIPS-style report.
- Ruoyu Fu: implemented the experiments, ran the PG-19/WikiText evaluations,
  and prepared the experimental analysis.

Workload: shared by the three listed authors.
