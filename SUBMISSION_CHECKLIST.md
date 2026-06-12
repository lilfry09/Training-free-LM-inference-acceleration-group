# Submission Checklist

## Assignment Mapping

- [x] Group part focuses on integrating a series of training-free inference
  acceleration methods.
- [x] Baseline model is Pythia-70M.
- [x] No model training or weight updates are used.
- [x] Evaluation includes WikiText and PG-19.
- [x] Metrics include PPL, TTFT, TPOT, throughput, and approximate KV cache
  memory.
- [x] Report is in English and uses the provided NeurIPS 2025 template.
- [x] Report includes abstract, introduction, methods, and experiments.
- [x] Body is kept within the 4-page course limit.
- [x] Code link is included in the paper.
- [x] Group division of labor/workload is included in the paper and README.

## Implemented Components

- [x] `src/evaluate.py`: baseline, GQA, and KVPress evaluation.
- [x] `src/methods.py`: reduced-KV grouped attention implementation.
- [x] `src/kvpress_adapter.py`: KVPress hook adapter for GPT-NeoX/Pythia.
- [x] `scripts/run_matrix.ps1`: full group experiment matrix.
- [x] `scripts/run_ablations.ps1`: KVPress ratio and GQA sanity diagnostics.
- [x] `layerwise_compression.py`: additional layer-wise KV compression prototype.
- [x] `eval_quick.py`: quick PG-19/WikiText evaluation for the layer-wise
  prototype.

## Verification Commands

```powershell
python -m py_compile eval_quick.py layerwise_compression.py test_layerwise_compression.py
python -m pytest -q
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

Verified locally:

- `pytest`: 4 passed
- `eval_quick.py --dataset pg19`: wrote `results_pg19.json`
- `eval_quick.py --dataset wikitext`: wrote `results_wikitext.json`

## Key Result Files

- `outputs/comparison_quality.md`
- `outputs/comparison_speed.md`
- `outputs/comparison_memory.md`
- `outputs/comparison_kvpress_ablation.md`
- `outputs/comparison_gqa_sanity.md`
- `results_pg19.json`
- `results_wikitext.json`
- `paper.pdf`
