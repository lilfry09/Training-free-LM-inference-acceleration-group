# Submission Checklist

## Completed

- [x] Real layer-wise KV cache compressor in `layerwise_compression.py`
- [x] Main reproducible evaluation script in `eval_quick.py`
- [x] Local model loading from `..\finalproj\models\pythia-70m`
- [x] PG-19 sample evaluation from `..\finalproj\datasets\pg19_samples\test_1.txt`
- [x] Unit tests in `test_layerwise_compression.py`
- [x] Latest metrics saved to `results.json`
- [x] README updated with real command and real results
- [x] `paper.tex` updated to match the implementation

## Verification Commands

```powershell
python -m pytest -q
python eval_quick.py
```

Verified locally:

- `pytest`: 4 passed
- `eval_quick.py`: completed and wrote `results.json`

## Latest Main Result

| Method | PPL | Time (s) | Speedup | Cache reduction |
|---|---:|---:|---:|---:|
| Dense | 37.5041 | 0.6647 | 1.0000x | 0.0% |
| Uniform-50% | 38.4568 | 1.8157 | 0.3661x | 43.0% |
| Uniform-67% | 36.0841 | 0.6518 | 1.0198x | 27.8% |
| LayerWise-80/50/70 | 39.2461 | 0.5826 | 1.1410x | 28.4% |

## Before Submission

- [ ] Replace `Your Name`, `Your University`, and email in `paper.tex`
- [ ] Add the final GitHub repository link in the report if required
- [ ] Compile `paper.tex` to PDF
- [ ] Push the repository to GitHub

## Suggested Submission Statement

This project implements a training-free layer-wise KV cache compression method
for Pythia-70M.  The submitted results are produced by `python eval_quick.py`
using a local Pythia-70M checkpoint and a PG-19 sample.  The implementation
compresses the KV cache after prompt prefill and evaluates cached continuation
PPL plus greedy decoding speed.
