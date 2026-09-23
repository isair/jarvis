# Optimized build for the existing env: kill jarvis processes, reuse cached
# native artifacts, PyInstaller onedir, no tests.
Set-Location $PSScriptRoot/..
$env:PYTHONPATH = "$PWD\src"

# Free the mapped .pyd/.dll files so Remove-Item can delete dist\.
Get-Process -Name Jarvis, python, python313 -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -ne $PID } |
    ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
Start-Sleep -Milliseconds 700

# Only the PyInstaller artifacts are stale between runs. The native build
# trees (build\native_audio_engine, build\virtual_mic) are cached inputs the
# spec stages into the bundle, so they survive the clean.
if (Test-Path build\jarvis_desktop) { Remove-Item -LiteralPath build\jarvis_desktop -Recurse -Force }
if (Test-Path dist) { Remove-Item -LiteralPath dist -Recurse -Force }

# The native engine DLL must exist before PyInstaller runs: jarvis_desktop.spec
# stages build\native_audio_engine\{Debug,Release}\jarvis_audio_engine.dll and
# the daemon fails closed (AUDIO_DSP_ERROR) when the bundle lacks it.
# Rebuild only when the cached DLL is gone; otherwise skip cmake entirely.
$aeDir = "$PWD\build\native_audio_engine"
$aeDll = @( @(
    (Join-Path $aeDir 'Debug\jarvis_audio_engine.dll'),
    (Join-Path $aeDir 'Release\jarvis_audio_engine.dll')
) | Where-Object { Test-Path $_ } )

if ($aeDll) {
    Write-Host "Native audio engine cached: $($aeDll[0]) - skipping cmake"
} else {
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        Write-Host "cmake not on PATH - install cmake (>=3.23) + MSVC Build Tools"
        exit 1
    }
    if (Test-Path (Join-Path $aeDir 'CMakeCache.txt')) {
        # Configured already (e.g. after an interrupted run): incremental build.
        & cmake --build $aeDir --parallel 8
    } else {
        & cmake -S "$PWD\native\audio_engine" -B $aeDir
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & cmake --build $aeDir --parallel 8
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $aeDll = @( @(
        (Join-Path $aeDir 'Debug\jarvis_audio_engine.dll'),
        (Join-Path $aeDir 'Release\jarvis_audio_engine.dll')
    ) | Where-Object { Test-Path $_ } )
    if (-not $aeDll) {
        Write-Host "Native audio engine DLL missing after cmake - bundle would ship without AEC3"
        exit 1
    }
    Write-Host "Native audio engine built: $($aeDll[0])"
}

# No test run here: pytest is 1 (skip) or 5 (no tests), both non-zero by design.
& "$PSScriptRoot\..\.venv-openvino-npu\Scripts\python.exe" -W ignore -m PyInstaller --noconfirm jarvis_desktop.spec
$pyi = $LASTEXITCODE
if ($pyi -ne 0) { exit $pyi }

& .\dist\Jarvis\Jarvis.exe --smoke-test
exit $LASTEXITCODE
