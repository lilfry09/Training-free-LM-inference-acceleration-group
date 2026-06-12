# Quality Comparison

| dataset | split | method | chunked_ppl | chunked_tokens | cache_ppl | cache_tokens | cache_context_len | kv_events | kv_avg_compression | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pg19 | test | baseline | 33.573871 | 4095 | 29.601993 | 256 | 512 | 0 |  | metrics_baseline_pg19_test.json |
| pg19 | test | gqa_reduced_kv2 | 816.695421 | 4095 | 776.882581 | 256 | 512 | 0 |  | metrics_gqa_pg19_test.json |
| pg19 | test | kvpress_knorm_r0.5 | 33.573871 | 4095 | 31.757461 | 256 | 512 | 78 | 0.500000 | metrics_kvpress_knorm_r0.5_pg19_test.json |
| pg19 | test | kvpress_streamingllm_r0.5_sink4 | 33.573871 | 4095 | 29.486057 | 256 | 512 | 78 | 0.500000 | metrics_kvpress_streamingllm_r0.5_pg19_test.json |
| wikitext | test | baseline | 63.718854 | 4095 | 13.088739 | 256 | 512 | 0 |  | metrics_baseline_wikitext_test.json |
| wikitext | test | gqa_reduced_kv2 | 2258.011303 | 4095 | 3659.004914 | 256 | 512 | 0 |  | metrics_gqa_wikitext_test.json |
| wikitext | test | kvpress_knorm_r0.5 | 63.718854 | 4095 | 52.436601 | 256 | 512 | 78 | 0.500000 | metrics_kvpress_knorm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_streamingllm_r0.5_sink4 | 63.718854 | 4095 | 61.299741 | 256 | 512 | 78 | 0.500000 | metrics_kvpress_streamingllm_r0.5_wikitext_test.json |
