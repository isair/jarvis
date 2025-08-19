@echo off
setlocal EnableDelayedExpansion

:: Get script directory and repo root
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
cd /d "%REPO_ROOT%"

:: Create virtual environment if it doesn't exist
if not exist .venv (
    python -m venv .venv
    if errorlevel 1 (
        echo Error creating virtual environment
        exit /b 1
    )
)

:: Activate virtual environment
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error activating virtual environment
    exit /b 1
)

:: Install requirements
pip install -r requirements.txt
if errorlevel 1 (
    echo Error installing requirements
    exit /b 1
)

:: Set environment variables
set "PYTHONPATH=%REPO_ROOT%\src"
if not defined JARVIS_VOICE_DEBUG set "JARVIS_VOICE_DEBUG=0"

:: Run Jarvis daemon
python -m jarvis.daemon
