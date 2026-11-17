# Project python wrapper.
#
# Runtime preference:
#   1) ECG\.venv_iso  -- FULLY ISOLATED venv (no --system-site-packages): torch 2.5.1+cu121,
#      numpy/scipy/pandas/sklearn/matplotlib/pyarrow, wfdb/neurokit2/streamlit/ecgen-eval all
#      physically inside the venv. Nothing is inherited from any system conda environment.
#   2) ECG\.venv      -- legacy venv using the system environment (fallback)
#   3) conda 环境 python -- last resort; may be missing the project packages
#
# Usage (from anywhere):
#   powershell -File tools\run.ps1 experiments\train_classifier.py --smoke
#   powershell -File tools\run.ps1 tools\env_check.py
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter()][string]$Module = "",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Args,
    [Parameter()][string]$FallbackPython = "python"
)
$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent

$isoPy = Join-Path $root ".venv_iso\Scripts\python.exe"
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $isoPy) {
    $env:PYTHONPATH = "$root"
    $python = $isoPy
} elseif (Test-Path $venvPy) {
    $env:PYTHONPATH = "$root"
    $python = $venvPy
} else {
    Write-Warning "No project venv found (.venv_iso / .venv). Using $FallbackPython; project packages may be missing."
    $env:PYTHONPATH = "$root"
    $python = $FallbackPython
}
$env:TEMP = "$root\.tmp"
$env:TMP = "$root\.tmp"
$env:PYTHONUNBUFFERED = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
New-Item -ItemType Directory -Force -Path "$root\.tmp" | Out-Null
Set-Location $root

if ($Module) {
    & $python -m $Module @Args
} else {
    & $python @Args
}
exit $LASTEXITCODE
