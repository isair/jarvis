@echo off
REM Run script for the Jarvis Desktop App on Windows
REM Uses the project's mamba environment
REM Usage: run_desktop_app.bat [--voice-debug]

REM Parse arguments
set "VOICE_DEBUG=0"
:parse_args
if "%~1"=="" goto done_args
if "%~1"=="--voice-debug" (
    set "VOICE_DEBUG=1"
    shift
    goto parse_args
)
shift
goto parse_args
:done_args

echo Testing Jarvis Desktop App locally...
if "%VOICE_DEBUG%"=="1" (
    echo    Voice debug: ENABLED
)
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
"%MAMBA_ENV%\python.exe" src\desktop_app\desktop_assets\generate_icons.py
echo.

REM Run the desktop app
echo Starting desktop app...
echo    Click the system tray icon to open menu
echo    Select 'Start Listening' from menu to begin
echo    Or press Ctrl+C to quit
echo.

REM Set voice debug environment variable if requested
if "%VOICE_DEBUG%"=="1" (
    set "JARVIS_VOICE_DEBUG=1"
)

"%MAMBA_ENV%\python.exe" -m desktop_app

