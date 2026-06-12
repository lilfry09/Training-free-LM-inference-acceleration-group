param(
  [int]$SpeedRepeats = 3,
  [int]$SpeedWarmupRuns = 1
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outDir = Join-Path $projectRoot "outputs"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$ratios = @(0.25, 0.5, 0.75)
foreach ($ratio in $ratios) {
  Write-Host "Running KVPress StreamingLLM ratio=$ratio"
  python (Join-Path $projectRoot "src/evaluate.py") `
    --project_root $projectRoot `
    --mode kvpress `
    --dataset wikitext `
    --split test `
    --max_eval_tokens 4096 `
    --ppl_context_len 512 `
    --cached_ppl_context_len 512 `
    --cached_ppl_eval_tokens 256 `
    --speed_context_lens "512" `
    --gen_new_tokens 64 `
    --speed_repeats $SpeedRepeats `
    --speed_warmup_runs $SpeedWarmupRuns `
    --kvpress_method streamingllm `
    --kvpress_compression_ratio $ratio `
    --kvpress_n_sink 4 `
    --output_json (Join-Path $outDir ("ablation_streamingllm_r{0}_wikitext_test.json" -f $ratio))
}

$kvHeadsList = @(8, 4, 2)
foreach ($kvHeads in $kvHeadsList) {
  Write-Host "Running GQA sanity kv_heads=$kvHeads"
  python (Join-Path $projectRoot "src/evaluate.py") `
    --project_root $projectRoot `
    --mode gqa `
    --dataset wikitext `
    --split test `
    --max_eval_tokens 4096 `
    --ppl_context_len 512 `
    --cached_ppl_context_len 512 `
    --cached_ppl_eval_tokens 256 `
    --speed_context_lens "512" `
    --gen_new_tokens 64 `
    --speed_repeats $SpeedRepeats `
    --speed_warmup_runs $SpeedWarmupRuns `
    --gqa_kv_heads $kvHeads `
    --output_json (Join-Path $outDir ("gqa_sanity_kv{0}_wikitext_test.json" -f $kvHeads))
}

Push-Location $projectRoot
try {
  python "src/summarize_ablations.py"
} finally {
  Pop-Location
}
Write-Host "Ablation run complete."
