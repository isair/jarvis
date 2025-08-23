Param()

$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[jarvis] $msg" }

# Repo root
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO_ROOT = Resolve-Path (Join-Path $SCRIPT_DIR '..')
Set-Location $REPO_ROOT

# Helper to set env vars for the current process
$env:PYTHONPATH = Join-Path $REPO_ROOT 'src'
if (-not $env:JARVIS_VOICE_DEBUG) { $env:JARVIS_VOICE_DEBUG = '0' }

# Prefer micromamba if available
$micromamba = Get-Command micromamba -ErrorAction SilentlyContinue
if ($micromamba) {
  $envPrefix = Join-Path $REPO_ROOT '.mamba_env'
  Write-Info "Using Micromamba environment at '$envPrefix'"

  if (-not (Test-Path $envPrefix)) {
    Write-Info 'Creating environment (python 3.12)...'
    micromamba create -y -p $envPrefix python=3.12 -c conda-forge
  }

  Write-Info 'Installing PyAV (FFmpeg bindings) from conda-forge...'
  micromamba install -y -p $envPrefix -c conda-forge av

  Write-Info 'Installing Python requirements with pip...'
  micromamba run -p $envPrefix pip install -r requirements.txt

  Write-Info 'Starting daemon...'
  micromamba run -p $envPrefix python -m jarvis.daemon
  exit $LASTEXITCODE
}

# Fallback: venv + pip
$venvPath = Join-Path $REPO_ROOT '.venv'
$venvPython = Join-Path $venvPath 'Scripts/python.exe'

if (-not (Test-Path $venvPython)) {
  Write-Info 'Creating virtual environment (.venv)...'
  python -m venv $venvPath
}

Write-Info 'Installing Python requirements with pip...'
& $venvPython -m pip install -r requirements.txt

Write-Info 'Starting daemon...'
& $venvPython -m jarvis.daemon


