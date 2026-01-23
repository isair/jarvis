@echo off
REM Run script for the Jarvis Desktop App on Windows
REM Uses the project's mamba environment

echo Testing Jarvis Desktop App locally...
echo.

REM Navigate to project root
cd /d "%~dp0.."

REM Set up paths
set "PROJECT_ROOT=%cd%"
set "MAMBA_ENV=%PROJECT_ROOT%\.mamba_env"
set "PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%"

REM Check if mamba environment exists
if not exist "%MAMBA_ENV%\python.exe" (
    echo ERROR: Mamba environment not found at %MAMBA_ENV%
    echo Please run the setup script first.
    pause
    exit /b 1
)

REM Check Python version in mamba env
echo Checking Python version...
"%MAMBA_ENV%\python.exe" --version
echo.

REM Install/update dependencies from requirements.txt
echo Installing dependencies...
"%MAMBA_ENV%\python.exe" -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some dependencies may have failed to install
)
echo.

REM Generate icons
echo Generating icons...
"%MAMBA_ENV%\python.exe" src\jarvis\desktop_assets\generate_icons.py
echo.

REM Run the desktop app
echo Starting desktop app...
echo    Click the system tray icon to open menu
echo    Select 'Start Listening' from menu to begin
echo    Or press Ctrl+C to quit
echo.

"%MAMBA_ENV%\python.exe" -m jarvis.desktop_app

