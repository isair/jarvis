Param([switch]$Rebuild, [switch]$NoDesktop)

$ErrorActionPreference = 'Stop'

function Info($m) { Write-Host "[jarvis] $m" }

# Repo root (folder one level above `scripts/`).
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO_ROOT  = (Resolve-Path (Join-Path $SCRIPT_DIR '..')).Path
Set-Location $REPO_ROOT

$BUNDLED_EXE   = Join-Path $REPO_ROOT 'dist\Jarvis\Jarvis.exe'
$NATIVE_BUILD  = Join-Path $REPO_ROOT 'build\native_audio_engine'
$NATIVE_DEBUG  = Join-Path $NATIVE_BUILD 'Debug\jarvis_audio_engine.dll'
$NATIVE_REL    = Join-Path $NATIVE_BUILD 'Release\jarvis_audio_engine.dll'

# --- resolve the best python ------------------------------------------------
function Resolve-Python {
    $c = @(
        (Join-Path $REPO_ROOT '.venv-openvino-npu\Scripts\python.exe'),  # deps installed here
        (Join-Path $REPO_ROOT '.mamba_env\python.exe'),
        (Join-Path $REPO_ROOT '.venv\Scripts\python.exe')
    )
    foreach ($p in $c) {
        if (Test-Path -LiteralPath $p) { return (Resolve-Path -LiteralPath $p).Path }
    }
    foreach ($cand in @(
            $env:MAMBA_PYTHON, 'C:\Users\lukes.COREI9\Miniconda3\python.exe',
            'C:\Python313\python.exe', 'C:\Python314\python.exe'
        )) {
        if ($cand -and (Test-Path -LiteralPath $cand)) { return (Resolve-Path -LiteralPath $cand).Path }
    }
    $c2 = Get-Command python -ErrorAction SilentlyContinue
    if ($c2) { return $c2.Source }
    throw 'No python interpreter found. Install Python 3.11+ or run `conda activate` first.'
}
$PY = Resolve-Python
$PYDIR = Split-Path -Parent $PY

# 3.14 ships with a different layout than 3.12; both are fine for these
# imports — keep the same directory as the venv root so `import jarvis` works.
$env:PYTHONPATH          = (Join-Path $REPO_ROOT 'src')
$env:JARVIS_PROJECT_ROOT = $REPO_ROOT

# --- 1. native engine: cmake + MSVC (skip if cached, rebuild via -Rebuild) -
if ($Rebuild -or -not (Test-Path -LiteralPath $NATIVE_DEBUG -PathType Leaf)) {
    if (-not (Test-Path -LiteralPath $NATIVE_REL -PathType Leaf)) {
        Info 'Building native audio engine (CMake)'
        if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
            throw 'cmake not on PATH. Install cmake (>=3.23) + MSVC Build Tools.'
        }
        & cmake -S "$REPO_ROOT\native\audio_engine" -B "$NATIVE_BUILD" "-DPython3_EXECUTABLE=$PY"
        if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }
        & cmake --build "$NATIVE_BUILD" --parallel 8
        if ($LASTEXITCODE -ne 0) { throw "cmake build failed ($LASTEXITCODE)" }
    }
}

$DLL = if (Test-Path -LiteralPath $NATIVE_DEBUG -PathType Leaf) { $NATIVE_DEBUG }
       elseif (Test-Path -LiteralPath $NATIVE_REL -PathType Leaf) { $NATIVE_REL }
else { throw "jarvis_audio_engine.dll missing under $NATIVE_BUILD" }
Info "native DLL = $DLL"

# --- 2. PyInstaller bundle of the desktop app (Jarvis.exe) ------------------
if ($Rebuild -or -not (Test-Path -LiteralPath $BUNDLED_EXE -PathType Leaf)) {
    Info 'Running PyInstaller (jarvis_desktop.spec)'
    & $PY -m PyInstaller --noconfirm --clean "$REPO_ROOT\jarvis_desktop.spec"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed ($LASTEXITCODE); check jarvis_desktop.spec + installed deps."
    }
}
$BUNDLED_DLL = Join-Path $REPO_ROOT 'dist\Jarvis\_internal\jarvis_audio_engine.dll'
if (-not (Test-Path -LiteralPath $BUNDLED_DLL -PathType Leaf)) {
    Info 'Bundling DLL into the onedir dist'
    Copy-Item -LiteralPath $DLL -Destination $BUNDLED_DLL -Force
}
Info "desktop EXE = $BUNDLED_EXE"

# --- 3. run ------------------------------------------------------------------
Info ('python   = ' + $PY)
Info ('1) Run the packaged app: ' + $BUNDLED_EXE)
Info ('2) Or the raw interpreter: "' + $PY + '" -m jarvis.main (PYTHONPATH=src)')
Info ('3) Sidecars optional: ' + (Join-Path $NATIVE_BUILD 'Debug\jarvis_audio_engine_sidecar.exe'))

if ($NoDesktop -and $PY) {
    & $PY '-m' 'jarvis.main' @args
    exit $LASTEXITCODE
}
& $BUNDLED_EXE @args
exit $LASTEXITCODE
