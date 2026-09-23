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

$VM_BUILD   = Join-Path $REPO_ROOT 'build\virtual_mic'
$VM_REL     = Join-Path $VM_BUILD 'Release'
$VM_PKG     = Join-Path $REPO_ROOT 'native\virtual_mic\package'
$VM_DRV_OUT = Join-Path $REPO_ROOT 'native\virtual_mic\driver\Toustova.5E9AEB52\x64\Release'

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

# --- 1b. virtual-mic stack: broker + installer (cmake) -----------------------
$BROKER_EXE = Join-Path $VM_REL 'ToustovacAudioBroker.exe'
$INSTALL_EXE = Join-Path $VM_REL 'ToustovacAudioInstall.exe'
if ($Rebuild -or -not (Test-Path -LiteralPath $BROKER_EXE -PathType Leaf) `
    -or -not (Test-Path -LiteralPath $INSTALL_EXE -PathType Leaf)) {
    Info 'Building virtual-mic user-mode binaries (CMake)'
    & cmake -S "$REPO_ROOT\native\virtual_mic" -B "$VM_BUILD" "-DPython3_EXECUTABLE=$PY"
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }
    & cmake --build "$VM_BUILD" --config Release --parallel 8
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed ($LASTEXITCODE)" }
}
foreach ($f in @($BROKER_EXE, $INSTALL_EXE)) {
    if (-not (Test-Path -LiteralPath $f -PathType Leaf)) {
        throw "virtual-mic binary missing: $f"
    }
    Info ('virtual-mic binary = ' + $f)
}

# --- 1c. virtual-mic WDK driver package (msbuild) ----------------------------
# The WDK installs km headers either flat (Include\km) or per SDK version
# (Include\<ver>\km); probe both before declaring the kit missing.
$KITS_INC  = 'C:\Program Files (x86)\Windows Kits\10\Include'
$KM_INC    = $null
if (Test-Path -LiteralPath (Join-Path $KITS_INC 'km')) {
    $KM_INC = Join-Path $KITS_INC 'km'
} elseif (Test-Path -LiteralPath $KITS_INC) {
    # Pick the highest version dir that actually contains a 'km' subfolder.
    # (A plain 'Sort-Object Name | Select -Last 1' lands on the 'wdf' dir,
    # which has no 'km', and would falsely report the headers as missing.)
    foreach ($d in (Get-ChildItem $KITS_INC -Directory | Sort-Object Name)) {
        $cand = Join-Path $d.FullName 'km'
        if (Test-Path -LiteralPath $cand) { $KM_INC = $cand }
    }
}
$DRV_SYS  = Join-Path $VM_DRV_OUT 'ToustovacVirtualMic.sys'
$PKG_INF  = Join-Path $VM_PKG 'ToustovacVirtualMic.inf'
$DRV_DONE = (Test-Path -LiteralPath $PKG_INF -PathType Leaf) -and `
            (Test-Path -LiteralPath $DRV_SYS -PathType Leaf)
if ($Rebuild -or -not $DRV_DONE) {
    $hasKm = $false
    if ($KM_INC -and (Test-Path -LiteralPath $KM_INC)) { $hasKm = $true }
    if (-not $hasKm) {
        Info 'WDK km headers missing (Include\km): driver package not built here;'
        Info 'driver sources under native\virtual_mic\driver build with WDK 28000.2526.'
    } else {
        Info 'Building virtual-mic driver package (msbuild)'
        $MSBUILD = Get-Command MSBuild.exe -ErrorAction SilentlyContinue
        $msb = if ($MSBUILD) { $MSBUILD.Source } else {
            'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe'
        }
        & $msb "$REPO_ROOT\native\virtual_mic\driver\ToustovacVirtualMic.vcxproj" `
            /m /p:Configuration=Release /p:Platform=x64 /nologo
        if ($LASTEXITCODE -ne 0) { throw "driver msbuild failed ($LASTEXITCODE)" }
    }
}
if (-not (Test-Path -LiteralPath $PKG_INF -PathType Leaf)) {
    throw "virtual-mic package INF missing: $PKG_INF"
}
# Copy the staged package + broker/install exes into the dist so the bundle
# (and the idempotent bootstrapper) share one folder.
$DIST     = Join-Path $REPO_ROOT 'dist\Jarvis'
$DIST_INT = Join-Path $DIST '_internal'
if (Test-Path -LiteralPath $DIST -PathType Container) {
    Info 'Staging virtual-mic artifacts into dist\Jarvis'
    foreach ($src in @(
            (Join-Path $VM_PKG 'ToustovacVirtualMic.inf'),
            (Join-Path $VM_PKG 'ToustovacVirtualMic.sys'),
            (Join-Path $VM_PKG 'ToustovacVirtualMic.cat'),
            $BROKER_EXE, $INSTALL_EXE)) {
        if (Test-Path -LiteralPath $src -PathType Leaf) {
            Copy-Item -LiteralPath $src -Destination $DIST -Force
            if (Test-Path -LiteralPath $DIST_INT -PathType Container) {
                Copy-Item -LiteralPath $src -Destination $DIST_INT -Force
            }
        }
    }
}

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
