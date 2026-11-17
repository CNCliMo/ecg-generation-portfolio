# Run every evaluation / aggregation / figure script in the correct order.
#   powershell -File tools\run_eval_all.ps1
# Assumes generated signal sets already exist in results/generated/.
$root = Split-Path $PSScriptRoot -Parent
function Run-Script([string]$rel, [string[]]$a = @()) {
    Write-Host "`n=== $rel $($a -join ' ') ===" -ForegroundColor Cyan
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "tools\run.ps1") $rel @a
    if ($LASTEXITCODE -ne 0) { Write-Warning "exit code $LASTEXITCODE from $rel (see results/logs)" }
}
Run-Script "experiments\eval_morphology.py"
Run-Script "experiments\eval_regularity.py"
Run-Script "experiments\eval_functional.py"
Run-Script "experiments\build_metrics_csv.py"
Run-Script "experiments\make_figures.py"
Run-Script "experiments\summarize_results.py"
Write-Host "`nEvaluation done: results\metrics.csv / results\summary.md / results\figures\*.png" -ForegroundColor Green
