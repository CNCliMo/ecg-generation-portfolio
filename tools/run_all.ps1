# One-click reproduction pipeline (order matters).
#
#   powershell -File tools\run_all.ps1                 # everything missing
#   powershell -File tools\run_all.ps1 -Stage data     # only data/preprocess
#   powershell -File tools\run_all.ps1 -Stage train    # only training
#   powershell -File tools\run_all.ps1 -Stage eval     # only evaluation/figures
param(
    [ValidateSet("all", "data", "classifier", "train", "eval", "app")]
    [string]$Stage = "all",
    [int]$DiffusionEpochs = 0,
    [int]$GanEpochs = 0
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$e = { param($p) & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "tools\run.ps1") $p @args }
$py = { param($script, [string[]]$a = @()) & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "tools\run.ps1") $script @a }

function Run-Script([string]$rel, [string[]]$a = @()) {
    Write-Host "`n=== $rel $($a -join ' ') ===" -ForegroundColor Cyan
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "tools\run.ps1") $rel @a
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $rel (exit $LASTEXITCODE)" }
}

if ($Stage -in "all", "data") {
    Run-Script "preprocess\download_data.py"
    Run-Script "preprocess\extract_ptbxl.py"
    Run-Script "preprocess\make_ptbxl_cache.py"
    Run-Script "preprocess\recompute_hr.py"
    Run-Script "preprocess\make_mitbih_cache.py"
    Run-Script "preprocess\extract_reference_params.py"
}

if ($Stage -in "all", "classifier") {
    Run-Script "experiments\train_classifier.py"
}

if ($Stage -in "all", "train") {
    $epB = if ($DiffusionEpochs -gt 0) { @("--epochs", "$DiffusionEpochs") } else { @() }
    Run-Script "experiments\train_diffusion.py" (@("--name", "b_uncond", "--cond-mode", "none") + $epB)
    Run-Script "experiments\train_diffusion.py" (@("--name", "b_cond", "--cond-mode", "label") + $epB)
    Run-Script "experiments\train_diffusion.py" (@("--name", "c_full", "--cond-mode", "both") + $epB)
    $ep = if ($DiffusionEpochs -gt 0) { @("--epochs", "$DiffusionEpochs") } else { @() }
    Run-Script "experiments\run_ablations.py" $ep
    $epG = if ($GanEpochs -gt 0) { @("--epochs", "$GanEpochs") } else { @() }
    Run-Script "experiments\run_route_d.py" $epG
}

if ($Stage -in "all", "eval") {
    Run-Script "experiments\generate_a.py"
    Run-Script "experiments\generate_diffusion.py" @("--ckpt", "models\checkpoints\b_uncond.pt", "--out", "B_uncond")
    Run-Script "experiments\generate_diffusion.py" @("--ckpt", "models\checkpoints\b_cond.pt", "--out", "B_cond")
    Run-Script "experiments\generate_diffusion.py" @("--ckpt", "models\checkpoints\c_full.pt", "--out", "C_full")
    Run-Script "experiments\eval_morphology.py"
    Run-Script "experiments\eval_regularity.py"
    Run-Script "experiments\eval_functional.py"
    Run-Script "experiments\build_metrics_csv.py"
    Run-Script "experiments\make_figures.py"
    Run-Script "experiments\summarize_results.py"
}

if ($Stage -in "all", "app") {
    Run-Script "tools\test_app.py"
}

Write-Host "`nDONE (stage: $Stage)" -ForegroundColor Green
