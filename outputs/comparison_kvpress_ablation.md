# KVPress Ratio Ablation

| dataset | method | compression_ratio | cached_ppl | throughput_tok_per_sec | throughput_tok_per_sec_std | ttft_sec | ttft_sec_std | tpot_sec | tpot_sec_std | cache_tokens | approx_kv_cache_mb | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wikitext | kvpress_streamingllm | 0.250000 | 22.918877 | 20.878022 | 7.667238 | 1.426161 | 0.975633 | 0.030742 | 0.007485 | 384 | 9.000000 | ablation_streamingllm_r0.25_wikitext_test.json |
| wikitext | kvpress_streamingllm | 0.500000 | 61.299741 | 59.618778 | 1.226254 | 0.160756 | 0.009170 | 0.014493 | 0.000219 | 256 | 6.000000 | ablation_streamingllm_r0.5_wikitext_test.json |
| wikitext | kvpress_streamingllm | 0.750000 | 62.607378 | 13.934054 | 1.231992 | 1.433173 | 0.543009 | 0.050523 | 0.014821 | 128 | 3.000000 | ablation_streamingllm_r0.75_wikitext_test.json |
