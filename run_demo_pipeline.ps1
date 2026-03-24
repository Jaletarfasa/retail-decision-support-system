param(
    [switch]$UsePytestEnv
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$env:RETAIL_RUNTIME_MODE = "demo"
if ($UsePytestEnv) {
    $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
}

Write-Host "Running retail decision support demo pipeline..."
Write-Host "Project root: $projectRoot"
Write-Host "Runtime mode: demo"

python -m src.orchestrator

Write-Host ""
Write-Host "Demo run complete."
Write-Host "Next steps:"
Write-Host "  1. streamlit run app/streamlit_app.py"
Write-Host "  2. Select the newest run from artifacts/runs/"
