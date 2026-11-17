# W2 experiments chain (idempotent): E2 lambda scan + E6 TSTR scaling + P-B correlation.
#
#   powershell -File tools\run_w2.ps1
#   powershell -File tools\run_w2.ps1 -SkipTraining    # after c_aux05_v2 training is done
#
# Performs:
#   1) generate c_aux05_v2 (lambda=0.5) 500 cond + 1000 uncond
#   2) generate 8000-sample conditional sets for TSTR scaling (c_full_v2/c_mask_v2/B_cond)
#   3) full evaluation of all routes (morphology/regularity/functional/consistency/downstream)
#   4) TSTR scaling experiment (E6) and metric-utility correlation (P-B)
#   5) rebuild metrics.csv, figures, summary
$ErrorActionPreference = "Continue"
param([switch]$SkipTraining)
$root = Split-Path $PSScriptRoot -Parent
$log = Join-Path $root "results\logs\run_w2.log"
$py = Join-Path $root ".venv_iso\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$env:PYTHONPATH = "$root"
$env:TEMP = "$root\.tmp"
$env:TMP = "$root\.tmp"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force -Path (Join-Path $root ".tmp") | Out-Null
Set-Location $root

function Log([string]$m) {
    $line = "[$(Get-Date -Format 'MM-dd HH:mm:ss')] $m"
    Add-Content -Path $log -Value $line -Encoding UTF8
    Write-Host $line
}
function Run-Py([string]$script, [string[]]$pyArgs = @()) {
    Log "RUN $script $($pyArgs -join ' ')"
    & $py (Join-Path $root $script) @pyArgs 2>&1 | ForEach-Object { Add-Content -Path $log -Value ("    " + $_) -Encoding UTF8 }
    Log "  exit=$LASTEXITCODE"
    return $LASTEXITCODE
}

# ------------------------------------------------------------------ training
$ckpt = Join-Path $root "models\checkpoints\c_aux05_v2.pt"
if (-not $SkipTraining) {
    if (Test-Path $ckpt) { Log "skip training c_aux05_v2 (checkpoint exists)" }
    else {
        Run-Py "experiments\train_diffusion.py" @("--name", "c_aux05_v2", "--cond-mode", "both",
            "--cond-style", "nle", "--aux-intervals", "--aux-weight", "0.5",
            "--epochs", "40", "--base", "48", "--ema-decay", "0.999")
    }
}

# ---------------------------------------------------------------- generation
$gen = Join-Path $root "results\generated"
if (-not (Test-Path (Join-Path $gen "c_aux05_v2_cond.npz"))) {
    Run-Py "experiments\generate_diffusion.py" @("--ckpt", "models\checkpoints\c_aux05_v2.pt", "--out", "c_aux05_v2")
} else { Log "skip generation c_aux05_v2" }

$gen8k = Join-Path $root "results\generated_8k"
New-Item -ItemType Directory -Force -Path $gen8k | Out-Null
foreach ($name in @("c_full_v2", "c_mask_v2", "B_cond")) {
    $big = Join-Path $gen8k "${name}_8k_cond.npz"
    $tmp = Join-Path $gen "${name}_8k_cond.npz"
    if (Test-Path $big) { Log "skip 8k generation $name"; continue }
    if (Test-Path $tmp) { Move-Item $tmp $big -Force; Log "moved existing 8k set $name"; continue }
    Run-Py "experiments\generate_diffusion.py" @("--ckpt", "models\checkpoints\$name.pt",
        "--out", "${name}_8k", "--n-cond", "8000", "--only", "cond")
    if (Test-Path $tmp) { Move-Item $tmp $big -Force }
}

# ---------------------------------------------------------------- evaluation
Run-Py "experiments\eval_morphology.py"
Run-Py "experiments\eval_regularity.py"
Run-Py "experiments\eval_functional.py"
Run-Py "experiments\eval_consistency.py"
Run-Py "experiments\eval_downstream.py"
Run-Py "experiments\eval_tstr_scaling.py"
Run-Py "experiments\eval_metric_utility.py"
Run-Py "experiments\build_metrics_csv.py"
Run-Py "experiments\make_figures.py"
Run-Py "experiments\summarize_results.py"
Run-Py "tools\final_check.py"
Log "W2 DONE"
