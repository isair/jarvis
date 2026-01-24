@echo off
REM Test script to build and run the bundled Windows app locally

echo.
echo 🔨 Building Jarvis Desktop App with PyInstaller...
echo.

REM Get to project root
cd /d "%~dp0\.."

REM Set up paths
set "PROJECT_ROOT=%cd%"
set "MAMBA_ENV=%PROJECT_ROOT%\.mamba_env"
set "PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%"

REM Check if mamba environment exists
if not exist "%MAMBA_ENV%\python.exe" (
    echo ❌ ERROR: Mamba environment not found at %MAMBA_ENV%
    echo    Please run the setup script first.
    pause
    exit /b 1
)

REM Clean previous builds
echo 🧹 Cleaning previous builds...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist
echo.

REM Build with PyInstaller
echo 📦 Building app bundle...
"%MAMBA_ENV%\python.exe" -m PyInstaller jarvis_desktop.spec
echo.

REM Check if build succeeded
if exist "dist\Jarvis.exe" (
    echo ✅ Build successful!
    echo.
    echo 📍 App location: %cd%\dist\Jarvis.exe
    echo.

    REM Show file info
    echo 📂 File info:
    dir dist\Jarvis.exe
    echo.

    REM Run the app
    echo 🚀 Launching app...
    echo    Press Ctrl+C in this window to stop the app
    echo.

    dist\Jarvis.exe

    echo.
    echo 👋 App exited.
) else (
    echo ❌ Build failed! Check the output above for errors.
    exit /b 1
)
