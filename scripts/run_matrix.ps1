param(
  [int]$SpeedRepeats = 3,
  [int]$SpeedWarmupRuns = 1
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$datasets = @("wikitext", "pg19")
$splits = @("test")

foreach ($dataset in $datasets) {
  foreach ($split in $splits) {
    Write-Host "Running mode=baseline dataset=$dataset split=$split"
    & (Join-Path $PSScriptRoot "run_eval.ps1") `
      -Mode baseline `
      -Dataset $dataset `
      -Split $split `
      -MaxEvalTokens 4096 `
      -PplContextLen 512 `
      -CachedPplContextLen 512 `
      -CachedPplEvalTokens 256 `
      -SpeedContextLens "128,512,1024" `
      -GenNewTokens 64 `
      -SpeedRepeats $SpeedRepeats `
      -SpeedWarmupRuns $SpeedWarmupRuns

    Write-Host "Running mode=gqa dataset=$dataset split=$split"
    & (Join-Path $PSScriptRoot "run_eval.ps1") `
      -Mode gqa `
      -Dataset $dataset `
      -Split $split `
      -MaxEvalTokens 4096 `
      -PplContextLen 512 `
      -CachedPplContextLen 512 `
      -CachedPplEvalTokens 256 `
      -SpeedContextLens "128,512,1024" `
      -GenNewTokens 64 `
      -SpeedRepeats $SpeedRepeats `
      -SpeedWarmupRuns $SpeedWarmupRuns `
      -GqaKvHeads 2

    Write-Host "Running mode=kvpress method=knorm dataset=$dataset split=$split"
    & (Join-Path $PSScriptRoot "run_eval.ps1") `
      -Mode kvpress `
      -Dataset $dataset `
      -Split $split `
      -MaxEvalTokens 4096 `
      -PplContextLen 512 `
      -CachedPplContextLen 512 `
      -CachedPplEvalTokens 256 `
      -SpeedContextLens "128,512,1024" `
      -GenNewTokens 64 `
      -SpeedRepeats $SpeedRepeats `
      -SpeedWarmupRuns $SpeedWarmupRuns `
      -KvpressMethod knorm `
      -KvpressCompressionRatio 0.5 `
      -KvpressNSink 4

    Write-Host "Running mode=kvpress method=streamingllm dataset=$dataset split=$split"
    & (Join-Path $PSScriptRoot "run_eval.ps1") `
      -Mode kvpress `
      -Dataset $dataset `
      -Split $split `
      -MaxEvalTokens 4096 `
      -PplContextLen 512 `
      -CachedPplContextLen 512 `
      -CachedPplEvalTokens 256 `
      -SpeedContextLens "128,512,1024" `
      -GenNewTokens 64 `
      -SpeedRepeats $SpeedRepeats `
      -SpeedWarmupRuns $SpeedWarmupRuns `
      -KvpressMethod streamingllm `
      -KvpressCompressionRatio 0.5 `
      -KvpressNSink 4
  }
}

python (Join-Path $projectRoot "src/summarize_results.py") --outputs_dir (Join-Path $projectRoot "outputs")
Write-Host "Matrix run complete."
