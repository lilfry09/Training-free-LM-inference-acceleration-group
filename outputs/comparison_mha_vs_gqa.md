# MHA (Baseline) vs GQA Comparison

- Generated at: 2026-04-13 14:41:44
- Fixed config: MaxEvalTokens=4096, PplContextLen=512, SpeedContextLens=128/512/1024, GenNewTokens=64, GqaKvHeads=2

## Validation

- [PASS] 4 JSON files exist and are non-empty.
- [PASS] Each JSON includes ppl_metrics and speed_metrics.
- [PASS] All speed_metrics include context_len=128,512,1024.
- [PASS] Baseline vs GQA config fields match within each dataset (except mode/gqa_kv_heads).

## wikitext (test)

- PPL: baseline=63.718854, gqa=2258.012514

| mode | context_len | ttft_sec | tpot_sec | throughput_tok_per_sec |
|---|---:|---:|---:|---:|
| baseline | 128 | 0.059151 | 0.020355 | 47.707292 |
| baseline | 512 | 0.161837 | 0.023560 | 38.878947 |
| baseline | 1024 | 0.229979 | 0.025304 | 35.084992 |
| gqa | 128 | 0.049740 | 0.020841 | 46.965411 |
| gqa | 512 | 0.118627 | 0.021867 | 42.773584 |
| gqa | 1024 | 0.286920 | 0.028960 | 30.312044 |

## pg19 (test)

- PPL: baseline=33.573871, gqa=816.692256

| mode | context_len | ttft_sec | tpot_sec | throughput_tok_per_sec |
|---|---:|---:|---:|---:|
| baseline | 128 | 0.062515 | 0.020958 | 46.281422 |
| baseline | 512 | 0.144792 | 0.024466 | 37.955717 |
| baseline | 1024 | 0.249112 | 0.023258 | 37.332034 |
| gqa | 128 | 0.052614 | 0.022195 | 44.110026 |
| gqa | 512 | 0.237982 | 0.024199 | 36.312065 |
| gqa | 1024 | 0.317271 | 0.023282 | 35.873818 |