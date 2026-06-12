# Speed Comparison

| dataset | split | method | context_len | ttft_sec | ttft_sec_std | tpot_sec | tpot_sec_std | e2e_sec | e2e_sec_std | throughput_tok_per_sec | throughput_tok_per_sec_std | generated_tokens | speed_repeats | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pg19 | test | baseline | 128 | 0.082925 | 0.003414 | 0.012354 | 0.000020 | 0.861248 | 0.004092 | 74.311894 | 0.352903 | 64 | 3 | metrics_baseline_pg19_test.json |
| pg19 | test | baseline | 512 | 0.235807 | 0.017307 | 0.013029 | 0.000603 | 1.056648 | 0.045447 | 60.644775 | 2.646282 | 64 | 3 | metrics_baseline_pg19_test.json |
| pg19 | test | baseline | 1024 | 0.421688 | 0.004956 | 0.014775 | 0.000178 | 1.352508 | 0.012450 | 47.322151 | 0.433397 | 64 | 3 | metrics_baseline_pg19_test.json |
| pg19 | test | gqa_reduced_kv2 | 128 | 0.137634 | 0.023270 | 0.017007 | 0.001682 | 1.209065 | 0.122787 | 53.279574 | 5.112459 | 64 | 3 | metrics_gqa_pg19_test.json |
| pg19 | test | gqa_reduced_kv2 | 512 | 0.774618 | 0.027887 | 0.039647 | 0.003928 | 3.272372 | 0.228553 | 19.619097 | 1.319073 | 64 | 3 | metrics_gqa_pg19_test.json |
| pg19 | test | gqa_reduced_kv2 | 1024 | 1.764500 | 0.014657 | 0.027033 | 0.001541 | 3.467604 | 0.083702 | 18.463804 | 0.451023 | 64 | 3 | metrics_gqa_pg19_test.json |
| pg19 | test | kvpress_knorm_r0.5 | 128 | 0.127634 | 0.026467 | 0.013632 | 0.000741 | 0.986469 | 0.066699 | 65.073239 | 4.337145 | 64 | 3 | metrics_kvpress_knorm_r0.5_pg19_test.json |
| pg19 | test | kvpress_knorm_r0.5 | 512 | 0.319038 | 0.023100 | 0.014704 | 0.001663 | 1.245416 | 0.127772 | 51.733489 | 5.046518 | 64 | 3 | metrics_kvpress_knorm_r0.5_pg19_test.json |
| pg19 | test | kvpress_knorm_r0.5 | 1024 | 0.619141 | 0.025672 | 0.015150 | 0.000440 | 1.573619 | 0.043371 | 40.691159 | 1.120762 | 64 | 3 | metrics_kvpress_knorm_r0.5_pg19_test.json |
| pg19 | test | kvpress_streamingllm_r0.5_sink4 | 128 | 0.107203 | 0.005270 | 0.013619 | 0.000730 | 0.965213 | 0.046865 | 66.408657 | 3.153667 | 64 | 3 | metrics_kvpress_streamingllm_r0.5_pg19_test.json |
| pg19 | test | kvpress_streamingllm_r0.5_sink4 | 512 | 0.279409 | 0.037219 | 0.012880 | 0.000901 | 1.090861 | 0.076386 | 58.854160 | 3.961108 | 64 | 3 | metrics_kvpress_streamingllm_r0.5_pg19_test.json |
| pg19 | test | kvpress_streamingllm_r0.5_sink4 | 1024 | 0.424589 | 0.014508 | 0.013013 | 0.000165 | 1.244398 | 0.019824 | 51.439103 | 0.812591 | 64 | 3 | metrics_kvpress_streamingllm_r0.5_pg19_test.json |
| wikitext | test | baseline | 128 | 0.118843 | 0.083761 | 0.019425 | 0.005767 | 1.342602 | 0.443368 | 50.871728 | 14.642830 | 64 | 3 | metrics_baseline_wikitext_test.json |
| wikitext | test | baseline | 512 | 0.568919 | 0.090142 | 0.020438 | 0.005078 | 1.856502 | 0.265616 | 34.950060 | 5.014091 | 64 | 3 | metrics_baseline_wikitext_test.json |
| wikitext | test | baseline | 1024 | 0.324587 | 0.037571 | 0.014732 | 0.000723 | 1.252690 | 0.080932 | 51.237877 | 3.432263 | 64 | 3 | metrics_baseline_wikitext_test.json |
| wikitext | test | gqa_reduced_kv2 | 128 | 0.178517 | 0.060352 | 0.021883 | 0.001846 | 1.557179 | 0.157572 | 41.377571 | 4.121378 | 64 | 3 | metrics_gqa_wikitext_test.json |
| wikitext | test | gqa_reduced_kv2 | 512 | 0.448013 | 0.307548 | 0.028127 | 0.003976 | 2.219997 | 0.506512 | 29.787900 | 6.341068 | 64 | 3 | metrics_gqa_wikitext_test.json |
| wikitext | test | gqa_reduced_kv2 | 1024 | 1.084065 | 0.358545 | 0.023911 | 0.004059 | 2.590451 | 0.432723 | 25.184492 | 4.314732 | 64 | 3 | metrics_gqa_wikitext_test.json |
| wikitext | test | kvpress_knorm_r0.5 | 128 | 0.116102 | 0.013648 | 0.013368 | 0.000215 | 0.958291 | 0.026827 | 66.819931 | 1.840942 | 64 | 3 | metrics_kvpress_knorm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_knorm_r0.5 | 512 | 0.333139 | 0.011058 | 0.014129 | 0.000617 | 1.223286 | 0.028170 | 52.336439 | 1.195861 | 64 | 3 | metrics_kvpress_knorm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_knorm_r0.5 | 1024 | 0.522897 | 0.054612 | 0.014896 | 0.002152 | 1.461330 | 0.083783 | 43.888860 | 2.437195 | 64 | 3 | metrics_kvpress_knorm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_streamingllm_r0.5_sink4 | 128 | 0.118794 | 0.018960 | 0.012885 | 0.000212 | 0.930558 | 0.032236 | 68.830224 | 2.351665 | 64 | 3 | metrics_kvpress_streamingllm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_streamingllm_r0.5_sink4 | 512 | 0.306238 | 0.013704 | 0.014591 | 0.001951 | 1.225458 | 0.119733 | 52.544604 | 4.903195 | 64 | 3 | metrics_kvpress_streamingllm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_streamingllm_r0.5_sink4 | 1024 | 0.614020 | 0.031587 | 0.014315 | 0.000898 | 1.515866 | 0.065658 | 42.274168 | 1.872426 | 64 | 3 | metrics_kvpress_streamingllm_r0.5_wikitext_test.json |
