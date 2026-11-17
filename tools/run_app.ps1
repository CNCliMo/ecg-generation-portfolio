# Launch the Streamlit app with the project environment (isolated venv first).
#
#   powershell -File tools\run_app.ps1                 # default port 8501
#   powershell -File tools\run_app.ps1 -Port 8600      # custom port
[CmdletBinding()]
param([int]$Port = 8501)
$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
$env:PYTHONPATH = "$root"
$isoPy = Join-Path $root ".venv_iso\Scripts\python.exe"
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $isoPy) {
    $python = $isoPy
} elseif (Test-Path $venvPy) {
    $python = $venvPy
} else {
    Write-Warning "No project venv found; using system python (project packages may be missing)."
    $python = "python"
}
$env:TEMP = "$root\.tmp"
$env:TMP = "$root\.tmp"
Set-Location $root
& $python -m streamlit run (Join-Path $root "app\streamlit_app.py") --server.port $Port --server.address 127.0.0.1
exit $LASTEXITCODE
