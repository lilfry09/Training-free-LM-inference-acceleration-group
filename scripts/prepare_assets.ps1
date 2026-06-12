param(
  [switch]$SkipModel,
  [switch]$SkipWikiText,
  [switch]$SkipPG19
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$pythonArgs = @(
  (Join-Path $projectRoot "src/prepare_assets.py"),
  "--project_root", $projectRoot
)

if ($SkipModel) {
  $pythonArgs += "--skip_model"
}
if ($SkipWikiText) {
  $pythonArgs += "--skip_wikitext"
}
if ($SkipPG19) {
  $pythonArgs += "--skip_pg19"
}

python @pythonArgs
