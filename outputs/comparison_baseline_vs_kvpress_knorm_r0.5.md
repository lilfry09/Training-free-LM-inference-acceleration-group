# Baseline(MHA) vs KVPress(Knorm, ratio=0.5)

## wikitext
- PPL: baseline=63.718854, kvpress=63.718854

| mode | context_len | TTFT(s) | TPOT(s) | Throughput(tok/s) |
|---|---:|---:|---:|---:|
| baseline | 128 | 0.059151 | 0.020355 | 47.707292 |
| baseline | 512 | 0.161837 | 0.023560 | 38.878947 |
| baseline | 1024 | 0.229979 | 0.025304 | 35.084992 |
| kvpress_knorm_r0.5 | 128 | 0.061270 | 0.023057 | 42.275346 |
| kvpress_knorm_r0.5 | 512 | 0.139783 | 0.023453 | 39.572212 |
| kvpress_knorm_r0.5 | 1024 | 0.233122 | 0.022224 | 39.185541 |

## pg19
- PPL: baseline=33.573871, kvpress=33.573871

| mode | context_len | TTFT(s) | TPOT(s) | Throughput(tok/s) |
|---|---:|---:|---:|---:|
| baseline | 128 | 0.062515 | 0.020958 | 46.281422 |
| baseline | 512 | 0.144792 | 0.024466 | 37.955717 |
| baseline | 1024 | 0.249112 | 0.023258 | 37.332034 |
| kvpress_knorm_r0.5 | 128 | 0.061335 | 0.032245 | 30.581062 |
| kvpress_knorm_r0.5 | 512 | 0.135512 | 0.022009 | 42.048028 |
| kvpress_knorm_r0.5 | 1024 | 0.233486 | 0.023112 | 37.880653 |
