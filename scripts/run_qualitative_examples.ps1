param(
  [int]$NewTokens = 48
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

python (Join-Path $projectRoot "src/generate_qualitative_examples.py") `
  --project_root $projectRoot `
  --new_tokens $NewTokens `
  --output_json "outputs/qualitative_examples.json" `
  --output_md "outputs/qualitative_examples.md"
