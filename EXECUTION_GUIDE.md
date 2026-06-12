# Execution Guide

This guide documents the commands used for the group-part layer-wise KV cache
compression submission. Run all commands from this repository:

```powershell
cd D:\SJTUlearning\2026Spring\NLP\FinalProject\Training-free-LM-inference-acceleration-group
pip install -r requirements.txt
```

## Local Assets

The default paths reuse the prepared assets in the sibling `finalproj`
directory:

- Model: `..\finalproj\models\pythia-70m`
- PG-19 sample: `..\finalproj\datasets\pg19_samples\test_1.txt`
- WikiText split: `..\finalproj\datasets\wikitext-103-raw-v1\test`

The evaluation script loads the model with `local_files_only=True`, so it uses
the local Pythia-70M checkpoint instead of downloading a new model.

## Tests

```powershell
python -m py_compile eval_quick.py layerwise_compression.py test_layerwise_compression.py
python -m pytest -q
```

Expected result:

```text
4 passed
```

## Reproduce Results

```powershell
python eval_quick.py --dataset pg19 --output_json results_pg19.json
python eval_quick.py --dataset wikitext --output_json results_wikitext.json
```

The script evaluates Dense, Uniform-50%, Uniform-67%, and
LayerWise-80/50/70 using:

- PPL: cached continuation perplexity with 96 prompt tokens and 64 scored tokens
- Speed: greedy decoding with 32 output tokens and 2 timing repeats
- Device: CPU, `torch.float32`
- Cache/FLOPs: average retained KV tokens per layer and approximate
  decode-attention FLOPs per generated token

## Current Results

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

## Compile Paper

```powershell
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

The final PDF uses the provided NeurIPS 2025 style file and is 3 pages, within
the course limit of 4 body pages.

## Interpretation

The implementation performs real KV cache slicing after prompt prefill. The
LayerWise-80/50/70 schedule consistently reduces KV cache size and approximate
attention FLOPs by 28.4%, while CPU wall-clock speed is mixed. The report keeps
this negative timing case as part of the analysis rather than overstating the
method.
