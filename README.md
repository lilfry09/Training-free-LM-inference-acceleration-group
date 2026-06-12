# Layer-wise Adaptive KV Cache Compression

Training-free KV cache compression for causal language model inference.  The
main implementation uses a prefill-then-compress policy: run the prompt once
with a full KV cache, compress each layer according to its position, then decode
with the compressed cache.

## Quick Start

This project reuses the local Pythia-70M model prepared in the sibling
`finalproj` directory:

```powershell
cd D:\SJTUlearning\2026Spring\NLP\FinalProject\layerwise-kv-compression
pip install -r requirements.txt
python eval_quick.py
```

Default paths:

- Model: `..\finalproj\models\pythia-70m`
- Text sample: `..\finalproj\datasets\pg19_samples\test_1.txt`
- Output: `results.json`

You can override them:

```powershell
python eval_quick.py --model_path ..\finalproj\models\pythia-70m --text_path ..\finalproj\datasets\pg19_samples\test_1.txt
```

## Method

Pythia-70M has 6 transformer layers.  The layer-wise schedule is:

| Layer group | Layers | Keep ratio | Motivation |
|---|---:|---:|---|
| Shallow | 0-1 | 80% | Preserve lower-level and semantic features |
| Middle | 2-3 | 50% | Compress more aggressively |
| Deep | 4-5 | 70% | Preserve final prediction context |

The compressor keeps a small prefix (`keep_initial_tokens=4`) plus recent
tokens, with `min_cache_tokens=16` to avoid over-compressing short contexts.

Core files:

- `layerwise_compression.py`: reusable compression utilities and generation loop
- `eval_quick.py`: real local-model evaluation script
- `test_layerwise_compression.py`: unit tests for ratio scheduling and KV slicing
- `paper.tex`: report draft using the bundled NeurIPS 2025 submission template
- `neurips_2025.sty`, `lineno.sty`, `natbib.sty`: local template/style files needed for compilation
- `paper.pdf`: compiled 4-page NeurIPS submission-style report with line numbers

## Latest Results

Command:

```powershell
python eval_quick.py
```

Environment:

- Model: local `Pythia-70M`
- Device: CPU, `torch.float32`
- Dataset sample: PG-19 `test_1.txt`
- PPL: cached continuation PPL with 96 prompt tokens and 64 scored tokens
- Speed: greedy decode, 96 prompt tokens, 32 generated tokens, 2 repeats

| Method | PPL | Time (s) | Tokens/s | Speedup | Cache reduction |
|---|---:|---:|---:|---:|---:|
| Dense | 37.5041 | 0.6647 | 48.1439 | 1.0000x | 0.0% |
| Uniform-50% | 38.4568 | 1.8157 | 17.6239 | 0.3661x | 43.0% |
| Uniform-67% | 36.0841 | 0.6518 | 49.0958 | 1.0198x | 27.8% |
| LayerWise-80/50/70 | 39.2461 | 0.5826 | 54.9280 | 1.1410x | 28.4% |

The result should be read conservatively because it is a small CPU quick test.
It nevertheless demonstrates that the implementation performs real KV cache
compression, not a theoretical speed estimate.

## Tests

```powershell
python -m pytest -q
```

Current status: `4 passed`.

## Notes for Submission

- Use `eval_quick.py` and `results.json` as the primary reproducible evidence.
- The older `quick_test.py`, `final_test.py`, and `real_compression_*.py` files
  are exploratory prototypes; they are not the main reported implementation.
- For a cleaner final hand-in, cite the local model path and the exact command
  used to produce `results.json`.
