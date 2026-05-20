@echo off
REM Build dist\Jarvis\Jarvis.exe using .venv (see build_exe.ps1)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_exe.ps1"
exit /b %ERRORLEVEL%
