param(
  [ValidateSet("wikitext", "pg19")]
  [string]$Dataset = "wikitext",
  [ValidateSet("train", "validation", "test")]
  [string]$Split = "test",
  [int]$MaxEvalTokens = 4096,
  [int]$PplContextLen = 512,
  [int]$SliceContextLen = 512,
  [int]$SliceEvalTokens = 256
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outDir = Join-Path $projectRoot "outputs"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$outJson = Join-Path $outDir ("cache_ppl_sanity_{0}_{1}.json" -f $Dataset, $Split)
$outMd = Join-Path $outDir ("cache_ppl_sanity_{0}_{1}.md" -f $Dataset, $Split)

python (Join-Path $projectRoot "src/cache_ppl_sanity.py") `
  --project_root $projectRoot `
  --dataset $Dataset `
  --split $Split `
  --max_eval_tokens $MaxEvalTokens `
  --ppl_context_len $PplContextLen `
  --slice_context_len $SliceContextLen `
  --slice_eval_tokens $SliceEvalTokens `
  --output_json $outJson `
  --output_md $outMd

Write-Host "Saved sanity check to $outJson and $outMd"
