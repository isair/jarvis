# Start cafe orchestrator + dashboard + Jarvis shell (UTF-8 safe on Windows)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$env:JARVIS_ROOT = $Root
$env:PYTHONPATH = "$Root\src"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Test-PortUp([int]$Port) {
    try {
        $null = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -TimeoutSec 2 -UseBasicParsing
        return $true
    } catch { return $false }
}

if (-not (Test-PortUp 8787)) {
    Write-Host "  Starting cafe orchestrator on :8787..."
    Start-Process powershell -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $Root "scripts\run_cafe_orchestrator.ps1")
    ) -WorkingDirectory $Root
    Start-Sleep -Seconds 3
}

if (-not (Test-PortUp 5050)) {
    Write-Host "  Starting memory viewer on :5050..."
    Start-Process powershell -ArgumentList @(
        "-NoProfile", "-Command",
        "`$env:JARVIS_ROOT='$Root'; `$env:PYTHONPATH='$Root\src'; `$env:PYTHONIOENCODING='utf-8'; & '$Root\.venv\Scripts\python.exe' -X utf8 -m desktop_app.memory_viewer 5050"
    ) -WorkingDirectory $Root
    Start-Sleep -Seconds 4
}

Write-Host "  Launching Jarvis shell..."
& (Join-Path $Root "scripts\run_jarvis_shell.ps1")
