# Run the v2 improvement pipeline (literature-driven), idempotent + resumable.
#
# ⚠ Time-boxed run: dorm power is cut at 23:00, so this v2 recipe uses a reduced
#   budget (base=48, 40 epochs) instead of the original plan (base=64, 120 epochs).
#   Training order prioritises the main method first, so a tight deadline still
#   leaves a complete, evaluable result set.
#
#   powershell -File tools\run_v2.ps1            # train + generate + evaluate
#   powershell -File tools\run_v2.ps1 -SkipTraining
#
# v2 models (base=48, 40 epochs, EMA 0.999):
#   c_full_v2 : nle label + HR conditioning + interval auxiliary loss  (main method)
#   c_mask_v2 : v1 mask label embedding + interval auxiliary loss      (ablation: nle off)
#   c_nle_v2  : nle label + HR conditioning, no interval auxiliary loss (ablation: aux off)
#   b_cond_v2 : improved route B (nle label embedding)
# v1 results are never overwritten: new routes are appended as *_v2.
$ErrorActionPreference = "Continue"
param([switch]$SkipTraining)
$root = Split-Path $PSScriptRoot -Parent
$log = Join-Path $root "results\logs\run_v2.log"
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
function CkptEpoch([string]$name) {
    $ckpt = Join-Path $root "models\checkpoints\$name.pt"
    if (-not (Test-Path $ckpt)) { return 0 }
    $script = "import torch; c=torch.load(r'$ckpt',map_location='cpu',weights_only=False); print(int(c['meta'].get('epoch',0)))"
    $ep = & $py -c $script 2>$null
    return [int]($ep | Select-Object -Last 1)
}

$epochs = 40
$base = 48
$deadlineTraining = Get-Date "2026-11-06 22:10:00"
$deadlineEvalStart = Get-Date "2026-11-06 22:20:00"

# --------------------------------------------------------------- aux targets
if (-not ((Test-Path (Join-Path $root "preprocess\cache\interval_targets_train.npz")) -and
          (Test-Path (Join-Path $root "preprocess\cache\interval_targets_val.npz")))) {
    Log "WARN interval targets missing; running extraction now"
    Run-Py "preprocess\extract_interval_targets.py"
}

# ---------------------------------------------------------------- training --
# priority order matters: the main method is trained first
$models = @(
    @{ name = "c_full_v2"; args = @("--cond-mode", "both", "--cond-style", "nle", "--aux-intervals", "--aux-weight", "0.1") },
    @{ name = "c_mask_v2"; args = @("--cond-mode", "both", "--cond-style", "mask", "--aux-intervals", "--aux-weight", "0.1") },
    @{ name = "c_nle_v2"; args = @("--cond-mode", "both", "--cond-style", "nle") },
    @{ name = "b_cond_v2"; args = @("--cond-mode", "label", "--cond-style", "nle") }
)
if (-not $SkipTraining) {
    foreach ($m in $models) {
        if ((Get-Date) -gt $deadlineTraining) { Log "deadline reached; skipping remaining training"; break }
        $done = CkptEpoch $m.name
        if ($done -ge $epochs) { Log "skip training $($m.name) (done $done/$epochs)"; continue }
        $resume = @()
        if ($done -gt 0) { $resume = @("--resume") }
        $pyArgs = @("--name", $m.name, "--epochs", "$epochs", "--base", "$base",
                    "--ema-decay", "0.999") + $m.args + $resume
        Run-Py "experiments\train_diffusion.py" $pyArgs
    }
}

# ------------------------------------------------------------- generation ---
foreach ($m in $models) {
    $ck = Join-Path $root "models\checkpoints\$($m.name).pt"
    if (-not (Test-Path $ck)) { Log "skip generation $($m.name) (no checkpoint)"; continue }
    $out = Join-Path $root "results\generated\$($m.name)_cond.npz"
    if (Test-Path $out) { Log "skip generation $($m.name)"; continue }
    Run-Py "experiments\generate_diffusion.py" @("--ckpt", "models\checkpoints\$($m.name).pt", "--out", $m.name)
}

# -------------------------------------------------------------- evaluation --
if ((Get-Date) -lt $deadlineEvalStart) {
    Run-Py "experiments\eval_morphology.py"
    Run-Py "experiments\eval_regularity.py"
    Run-Py "experiments\eval_functional.py"
} else {
    Log "eval start deadline reached; running lightweight evaluation only"
    Run-Py "experiments\eval_morphology.py"
    Run-Py "experiments\eval_functional.py"
}
Run-Py "experiments\eval_downstream.py"
Run-Py "experiments\eval_consistency.py"
Run-Py "experiments\build_metrics_csv.py"
Run-Py "experiments\make_figures.py"
Run-Py "experiments\summarize_results.py"
Run-Py "tools\final_check.py"
Log "V2 DONE"
