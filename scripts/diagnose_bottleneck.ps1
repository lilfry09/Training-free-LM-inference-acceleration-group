# Diagnose where the bottleneck is

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=== Bottleneck Diagnosis ===" -ForegroundColor Cyan

# Test 1: Does cache size matter?
Write-Host "`n[Test 1] Compare different context lengths (baseline)" -ForegroundColor Yellow
python "$ProjectRoot\src\evaluate.py" `
    --mode baseline `
    --dataset wikitext `
    --split test `
    --speed_context_lens "64,128,256,512,1024,2048" `
    --speed_repeats 1 `
    --enable_cached_ppl False `
    --output_json "$ProjectRoot\outputs\diag_baseline_ctx.json"

# Test 2: Does compression overhead dominate?
Write-Host "`n[Test 2] Minimal compression (ratio=0.1)" -ForegroundColor Yellow
python "$ProjectRoot\src\evaluate.py" `
    --mode kvpress `
    --kvpress_method streamingllm `
    --kvpress_compression_ratio 0.1 `
    --dataset wikitext `
    --split test `
    --speed_context_lens "512,1024" `
    --speed_repeats 1 `
    --enable_cached_ppl False `
    --output_json "$ProjectRoot\outputs\diag_kvpress_r0.1.json"

# Test 3: Profile per-layer timing
Write-Host "`n[Test 3] Per-layer timing analysis" -ForegroundColor Yellow
Write-Host "You need to manually add timing hooks in methods.py" -ForegroundColor Red
Write-Host "See the profile_layers.py script we'll create next" -ForegroundColor Red

Write-Host "`n=== Diagnosis complete. Check outputs/diag_*.json ===" -ForegroundColor Green