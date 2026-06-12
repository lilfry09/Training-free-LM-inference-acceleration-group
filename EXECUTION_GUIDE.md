# Execution Guide

## 1. Environment

Run from this directory:

```powershell
cd D:\SJTUlearning\2026Spring\NLP\FinalProject\layerwise-kv-compression
pip install -r requirements.txt
```

The default evaluation reuses the local model and data in:

- `D:\SJTUlearning\2026Spring\NLP\FinalProject\finalproj\models\pythia-70m`
- `D:\SJTUlearning\2026Spring\NLP\FinalProject\finalproj\datasets\pg19_samples\test_1.txt`

## 2. Run Tests

```powershell
python -m pytest -q
```

Expected result:

```text
4 passed
```

## 3. Run the Main Evaluation

```powershell
python eval_quick.py
```

This writes `results.json` and prints a Markdown table.

Current reproduced result on CPU:

| Method | PPL | Time (s) | Tokens/s | Speedup | Cache reduction |
|---|---:|---:|---:|---:|---:|
| Dense | 37.5041 | 0.6647 | 48.1439 | 1.0000x | 0.0% |
| Uniform-50% | 38.4568 | 1.8157 | 17.6239 | 0.3661x | 43.0% |
| Uniform-67% | 36.0841 | 0.6518 | 49.0958 | 1.0198x | 27.8% |
| LayerWise-80/50/70 | 39.2461 | 0.5826 | 54.9280 | 1.1410x | 28.4% |

## 4. Optional Overrides

```powershell
python eval_quick.py `
  --model_path ..\finalproj\models\pythia-70m `
  --text_path ..\finalproj\datasets\pg19_samples\test_1.txt `
  --prompt_tokens 96 `
  --ppl_tokens 160 `
  --max_new_tokens 32 `
  --repeats 2
```

## 5. Interpretation

- The implementation performs real KV cache slicing after prompt prefill.
- The reported PPL is cached continuation PPL, not a theoretical estimate.
- CPU timing is noisy, so the speedup should be treated as preliminary.
- The older `quick_test.py`, `final_test.py`, and `real_compression_*.py` scripts are exploratory prototypes; use `eval_quick.py` for submission results.
