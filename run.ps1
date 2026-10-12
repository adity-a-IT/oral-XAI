Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  oral-XAI :: High-Performance Diagnostic Suite" -ForegroundColor Green
Write-Host "  Running in Dedicated Python Virtual Environment (.venv)" -ForegroundColor Yellow
Write-Host "  [Streamlit decoupled - Standalone Web Engine]" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[INFO] Creating Python virtual environment in .venv..." -ForegroundColor Cyan
    python -m venv .venv
    Write-Host "[INFO] Installing required dependencies..." -ForegroundColor Cyan
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

Write-Host "[INFO] Starting oral-XAI standalone server on http://localhost:5000 ..." -ForegroundColor Green
Start-Process "http://localhost:5000"
.\.venv\Scripts\python.exe server.py
