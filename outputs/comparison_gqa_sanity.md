# GQA KV-Head Sanity

| dataset | kv_heads | kv_head_ratio | chunked_ppl | cached_ppl | throughput_tok_per_sec | throughput_tok_per_sec_std | cache_tokens | approx_kv_cache_mb | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wikitext | 8 | 1.000000 | 63.718854 | 13.088739 | 55.259370 | 3.362321 | 512 | 12.000000 | gqa_sanity_kv8_wikitext_test.json |
| wikitext | 4 | 0.500000 | 1240.553384 | 1976.445839 | 30.419304 | 3.952899 | 512 | 6.000000 | gqa_sanity_kv4_wikitext_test.json |
| wikitext | 2 | 0.250000 | 2258.011303 | 3659.004914 | 31.046966 | 10.075282 | 512 | 3.000000 | gqa_sanity_kv2_wikitext_test.json |
