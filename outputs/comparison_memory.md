# Approximate KV Cache Memory

| method | kv_heads | cache_tokens | approx_kv_cache_mb | reduction_vs_baseline |
| --- | --- | --- | --- | --- |
| baseline / gqa_8kv | 8 | 512 | 12.000000 | 0.0% |
| gqa_4kv | 4 | 512 | 6.000000 | 50.0% |
| gqa_2kv | 2 | 512 | 3.000000 | 75.0% |
| streamingllm_r0.25 | 8 | 384 | 9.000000 | 25.0% |
| streamingllm_r0.5 | 8 | 256 | 6.000000 | 50.0% |
| streamingllm_r0.75 | 8 | 128 | 3.000000 | 75.0% |
