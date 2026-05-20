# Launch Jarvis Tauri shell (dev mode)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$env:JARVIS_ROOT = $Root
$env:Path = "$env:USERPROFILE\.cargo\bin;" + $env:Path

$vcvars = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (Test-Path $vcvars) {
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path "env:$($matches[1])" -Value $matches[2]
        }
    }
}

Set-Location (Join-Path $Root "apps\jarvis_shell")
if (-not (Test-Path "node_modules")) { npm install }
& "$Root\.venv\Scripts\python.exe" scripts\gen_icons.py
npm run dev
