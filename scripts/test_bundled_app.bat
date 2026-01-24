@echo off
REM Test script to build and run the bundled Windows app locally

echo.
echo === Building Jarvis Desktop App with PyInstaller ===
echo.

REM Get to project root
cd /d "%~dp0\.."

REM Clean previous builds
echo Cleaning previous builds...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist
echo.

REM Build with PyInstaller
echo Building app bundle...
python -m PyInstaller jarvis_desktop.spec
echo.

REM Check if build succeeded
if exist "dist\Jarvis.exe" (
    echo Build successful!
    echo.
    echo App location: %cd%\dist\Jarvis.exe
    echo.

    REM Show file info
    echo File info:
    dir dist\Jarvis.exe
    echo.

    REM Run the app
    echo Launching app...
    echo Press Ctrl+C in this window to stop the app
    echo.

    dist\Jarvis.exe

    echo.
    echo App exited.
) else (
    echo Build failed! Check the output above for errors.
    exit /b 1
)
