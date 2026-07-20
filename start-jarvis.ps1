<#
    Jarvis launcher — hardened, venv-based.

    Differences from the upstream scripts/run_desktop_app.bat:
      * Uses .venv (Python 3.12) instead of .mamba_env.
      * Does NOT run `pip install` on every launch. Upstream re-resolves the whole
        dependency tree from PyPI each time you start the app, which re-exposes the
        supply chain on every run and is slow. Dependencies are pinned and already
        installed; use -Update to deliberately refresh them.
      * Starts the local Ollama server if it is not already running.

    Usage:
      .\start-jarvis.ps1                # desktop app (system tray GUI)
      .\start-jarvis.ps1 -Daemon        # headless voice daemon (CLI)
      .\start-jarvis.ps1 -VoiceDebug    # verbose voice pipeline logging
      .\start-jarvis.ps1 -Update        # git pull + refresh dependencies
#>
Param(
    [switch]$Daemon,
    [switch]$VoiceDebug,
    [switch]$Update
)

$ErrorActionPreference = 'Stop'

$REPO_ROOT  = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_PY    = Join-Path $REPO_ROOT '.venv\Scripts\python.exe'
$OLLAMA_EXE = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'

function Write-Info($msg) { Write-Host "[jarvis] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[jarvis] $msg" -ForegroundColor Yellow }

if (-not (Test-Path $VENV_PY)) {
    Write-Warn "Virtual environment not found at $VENV_PY"
    Write-Warn "Recreate it with:  py -3.12 -m venv .venv"
    exit 1
}

Set-Location $REPO_ROOT
$env:PYTHONPATH = Join-Path $REPO_ROOT 'src'
$env:JARVIS_VOICE_DEBUG = if ($VoiceDebug) { '1' } else { '0' }

if ($Update) {
    Write-Warn 'Updating from source. Review the diff before trusting it:'
    Write-Warn '  git -C "' + $REPO_ROOT + '" log --oneline HEAD..origin/main'
    git -C $REPO_ROOT pull
    & $VENV_PY -m pip install --prefer-binary -r (Join-Path $REPO_ROOT 'requirements.txt')
}

# Ollama must be reachable on loopback before Jarvis starts.
if (-not (Get-Process ollama -ErrorAction SilentlyContinue)) {
    if (Test-Path $OLLAMA_EXE) {
        Write-Info 'Starting Ollama server...'
        Start-Process -FilePath $OLLAMA_EXE -ArgumentList 'serve' -WindowStyle Hidden
        Start-Sleep -Seconds 3
    } else {
        Write-Warn "Ollama not found at $OLLAMA_EXE — Jarvis will fail to reach the LLM."
    }
}

try {
    $ver = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 10
    Write-Info "Ollama $($ver.version) ready on 127.0.0.1:11434"
} catch {
    Write-Warn 'Ollama API is not responding on 127.0.0.1:11434.'
}

if ($Daemon) {
    Write-Info 'Starting voice daemon (Ctrl+C to quit)...'
    & $VENV_PY -m jarvis.daemon
} else {
    Write-Info 'Starting desktop app — look for the tray icon, then "Start Listening".'
    & $VENV_PY -m desktop_app
}

exit $LASTEXITCODE
