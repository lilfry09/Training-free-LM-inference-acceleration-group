param(
  [ValidateSet("baseline", "flash", "gqa", "gqa_flash", "kvpress")]
  [string]$Mode = "baseline",
  [ValidateSet("wikitext", "pg19")]
  [string]$Dataset = "wikitext",
  [ValidateSet("train", "validation", "test")]
  [string]$Split = "test",
  [int]$MaxEvalTokens = 4096,
  [int]$PplContextLen = 512,
  [int]$CachedPplContextLen = 512,
  [int]$CachedPplEvalTokens = 256,
  [bool]$EnableCachedPpl = $true,
  [string]$SpeedContextLens = "128,512,1024",
  [int]$GenNewTokens = 64,
  [int]$SpeedRepeats = 3,
  [int]$SpeedWarmupRuns = 1,
  [int]$GqaKvHeads = 2,
  [ValidateSet("knorm", "streamingllm")]
  [string]$KvpressMethod = "knorm",
  [double]$KvpressCompressionRatio = 0.5,
  [int]$KvpressNSink = 4
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outDir = Join-Path $projectRoot "outputs"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
if ($Mode -eq "kvpress") {
  $outJson = Join-Path $outDir ("metrics_{0}_{1}_r{2}_{3}_{4}.json" -f $Mode, $KvpressMethod, $KvpressCompressionRatio, $Dataset, $Split)
} else {
  $outJson = Join-Path $outDir ("metrics_{0}_{1}_{2}.json" -f $Mode, $Dataset, $Split)
}

$pythonArgs = @(
  (Join-Path $projectRoot "src/evaluate.py"),
  "--project_root", $projectRoot,
  "--mode", $Mode,
  "--dataset", $Dataset,
  "--split", $Split,
  "--max_eval_tokens", $MaxEvalTokens,
  "--ppl_context_len", $PplContextLen,
  "--cached_ppl_context_len", $CachedPplContextLen,
  "--cached_ppl_eval_tokens", $CachedPplEvalTokens,
  "--speed_context_lens", $SpeedContextLens,
  "--gen_new_tokens", $GenNewTokens,
  "--speed_repeats", $SpeedRepeats,
  "--speed_warmup_runs", $SpeedWarmupRuns,
  "--gqa_kv_heads", $GqaKvHeads,
  "--kvpress_method", $KvpressMethod,
  "--kvpress_compression_ratio", $KvpressCompressionRatio,
  "--kvpress_n_sink", $KvpressNSink,
  "--output_json", $outJson
)

if ($EnableCachedPpl) {
  $pythonArgs += "--enable_cached_ppl"
} else {
  $pythonArgs += "--no-enable_cached_ppl"
}

python @pythonArgs

Write-Host "Saved metrics to $outJson"
