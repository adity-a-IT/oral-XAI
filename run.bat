@echo off
title oral-XAI :: Cyber-Medical Diagnostics Server
echo ========================================================
echo   oral-XAI :: High-Performance Diagnostic Suite
echo   Running in Dedicated Python Virtual Environment (.venv)
echo   [Streamlit decoupled - Standalone Web Engine]
echo ========================================================
echo.

cd /d "%~dp0"

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [INFO] Creating Python virtual environment in .venv...
    python -m venv .venv
    echo [INFO] Installing required dependencies...
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
)

echo [INFO] Starting oral-XAI standalone server on http://localhost:5000 ...
start http://localhost:5000
.\.venv\Scripts\python.exe server.py
pause
