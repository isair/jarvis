@echo off
REM Test script for the Jarvis Desktop App on Windows

echo 🔧 Testing Jarvis Desktop App locally...
echo.

REM Check Python version
echo 📋 Checking Python version...
python --version
echo.

REM Install desktop dependencies if needed
echo 📦 Installing desktop app dependencies...
pip install PyQt6 psutil pillow
echo.

REM Generate icons
echo 🎨 Generating icons...
cd /d "%~dp0.."
python src\jarvis\desktop_assets\generate_icons.py
echo.

REM Run the desktop app
echo 🚀 Starting desktop app...
echo    Click the system tray icon to open menu
echo    Select 'Start Listening' from menu to begin
echo    Or press Ctrl+C to quit
echo.

REM Set PYTHONPATH to include src directory
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%"

python -m jarvis.desktop_app

