# Submission Checklist

## Assignment Mapping

- [x] Group part keeps only the layer-wise KV compression project.
- [x] Baseline model is Pythia-70M.
- [x] No training or weight updates are used.
- [x] Evaluation includes PG-19 and WikiText.
- [x] Metrics include PPL, TTFT, TPOT, throughput, cache reduction, and
  approximate decode attention FLOPs.
- [x] Report is in English and uses the provided NeurIPS 2025 template.
- [x] Report includes abstract, introduction, method, and experiments.
- [x] Body is kept within the 4-page course limit.
- [x] Code link is included in the paper.
- [x] Group division of labor/workload is included in the paper and README.
- [x] Personal-part methods are not included in the group submission.

## Implemented Components

- [x] `layerwise_compression.py`: training-free layer-wise KV cache compression.
- [x] `eval_quick.py`: PG-19/WikiText evaluation script.
- [x] `test_layerwise_compression.py`: unit tests for schedule and slicing.
- [x] `results_pg19.json`: latest PG-19 result.
- [x] `results_wikitext.json`: latest WikiText result.
- [x] `paper.tex` and `paper.pdf`: NeurIPS-style report.

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
- `paper.pdf`: compiled with 4 pages in NeurIPS submission style

## Key Result Files

- `results_pg19.json`
- `results_wikitext.json`
- `results.json`
- `results_final.json`
- `paper.pdf`
