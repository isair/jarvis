[CmdletBinding()]
param(
    [string]$OpenVinoRoot = "",
    [string]$Python = "",
    [int]$Port = 8010
)

$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$requirements = Join-Path $PSScriptRoot "requirements-openvino-npu.txt"
$venvRoot = Join-Path $repositoryRoot ".venv-openvino-npu"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
# setupvars.ps1 can replace PATH with the OpenVINO runtime paths. Capture the
# interpreter before dot-sourcing it so an otherwise valid system Python is
# not hidden by the native runtime environment. The Python launcher (py.exe)
# is not itself an interpreter, so resolve it to the selected 64-bit Python
# executable while the original PATH is still available.
$pythonCandidatesBeforeSetup = @("python", "py")
$pythonCandidatesBeforeSetup = $pythonCandidatesBeforeSetup | ForEach-Object {
    Get-Command $_ -ErrorAction SilentlyContinue
} | Where-Object { $_ } | ForEach-Object {
    if ($_.Path) { $_.Path } else { $_.Source }
} | Where-Object { $_ } | Select-Object -Unique
$pythonCandidatesBeforeSetup += @(
    (Get-ChildItem -Path "C:\Python*\python.exe", "$env:LOCALAPPDATA\Programs\Python\Python*\python.exe", "C:\Program Files\Python*\python.exe" -File -ErrorAction SilentlyContinue).FullName
) | Where-Object { $_ } | Select-Object -Unique
$pythonFromLauncherBeforeSetup = @()
$pythonLauncherBeforeSetup = Get-Command "py" -ErrorAction SilentlyContinue
if ($pythonLauncherBeforeSetup) {
    $launcherPath = if ($pythonLauncherBeforeSetup.Path) { $pythonLauncherBeforeSetup.Path } else { $pythonLauncherBeforeSetup.Source }
    $launcherPython = & $launcherPath -3 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $launcherPython) {
        $pythonFromLauncherBeforeSetup = @($launcherPython | Where-Object {
            Test-Path -LiteralPath $_ -PathType Leaf
        })
    }
}

if (-not $OpenVinoRoot) {
    $OpenVinoRoot = if ($env:OPENVINO_ROOT) { $env:OPENVINO_ROOT } else { "C:\Intel\openvino_2026.2.1\openvino_toolkit_windows_2026.2.1.21919.ede283a88e3_x86_64" }
}

function Resolve-PythonExecutable([string]$RequestedPython) {
    if ($RequestedPython) {
        if (Test-Path -LiteralPath $RequestedPython -PathType Leaf) { return (Resolve-Path -LiteralPath $RequestedPython).Path }
        $command = Get-Command $RequestedPython -ErrorAction SilentlyContinue
        if ($command) { return $command.Source }
        throw "The requested Python executable was not found: $RequestedPython"
    }

    if (Test-Path -LiteralPath $venvPython -PathType Leaf) { return $venvPython }

    $commands = @(
        $pythonFromLauncherBeforeSetup
        $pythonCandidatesBeforeSetup
        @("python", "py") | ForEach-Object {
            Get-Command $_ -ErrorAction SilentlyContinue
        } | Where-Object { $_ } | ForEach-Object {
            if ($_.Path) { $_.Path } else { $_.Source }
        } | Where-Object { $_ } | Select-Object -Unique
    ) | Where-Object { $_ } | Select-Object -Unique
    foreach ($candidate in $commands) {
        if ((Split-Path -Leaf $candidate) -eq "py.exe") { continue }
        return $candidate
    }
    throw "Python was not found. Install Python 3.10-3.14 (64-bit), or pass -Python with the executable path."
}

function Test-NativeDependencies([string]$PythonExecutable) {
    & $PythonExecutable -c "import numpy, openvino, transformers" 2>$null
    return $LASTEXITCODE -eq 0
}

$PythonExecutable = Resolve-PythonExecutable $Python
$pythonVersion = & $PythonExecutable -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $pythonVersion) {
    throw "Unable to determine the selected Python version: $PythonExecutable"
}

$setup = Join-Path $OpenVinoRoot "setupvars.ps1"
if (-not (Test-Path -LiteralPath $setup)) {
    throw "OpenVINO setupvars.ps1 was not found: $setup. Set OPENVINO_ROOT or pass -OpenVinoRoot."
}

# Ensure Python is in PATH before setupvars to avoid the "Python not found" warning
$pythonDir = Split-Path -Parent $PythonExecutable
if ($env:PATH -notlike "*$pythonDir*") {
    $env:PATH = "$pythonDir;$env:PATH"
}

# setupvars must be dot-sourced so its PATH/PYTHONPATH changes remain in this process.
# Suppress the setupvars output that was cluttering the logs
$setupOutput = . $setup -python_version $pythonVersion 2>&1 | Out-String
if ($setupOutput -notmatch "Warning") {
    # Only show setupvars output if there were no warnings (i.e., it was successful)
    Write-Host "[setupvars] OpenVINO environment initialized" -ForegroundColor DarkGray
}

$defaultVenvNeedsBootstrap = -not $Python -and (
    -not (Test-Path -LiteralPath $venvPython -PathType Leaf) -or
    -not (Test-NativeDependencies $venvPython)
)
$selectedPythonNeedsBootstrap = $Python -and -not (Test-NativeDependencies $PythonExecutable)
if ($defaultVenvNeedsBootstrap -or $selectedPythonNeedsBootstrap) {
    if ($Python) {
        throw "Selected Python does not have the OpenVINO package: $PythonExecutable. Install the pinned dependencies or omit -Python to use the repository venv."
    }

    if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
        throw "OpenVINO dependency manifest was not found: $requirements"
    }
    Write-Host "Creating repository-local OpenVINO environment: $venvRoot" -ForegroundColor Cyan
    & $PythonExecutable -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the OpenVINO virtual environment: $venvRoot" }

    Write-Host "Installing pinned OpenVINO NPU dependencies." -ForegroundColor Cyan
    & $venvPython -m pip install --disable-pip-version-check --upgrade -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "Failed to install OpenVINO dependencies. Check network access and rerun the launcher." }
    $PythonExecutable = $venvPython
}

$openVinoProbe = & $PythonExecutable -c "import numpy, openvino, transformers" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Selected Python does not have the native OpenVINO dependencies after environment setup: $PythonExecutable."
}

$env:OPENVINO_NPU_PORT = $Port
$env:OPENVINO_VERSION = "2026.2.1"
$env:NPU_DRIVER_VERSION = "32.0.100.4778"
# Suppress the transformers warning about PyTorch/TensorFlow not being installed
$env:TRANSFORMERS_NO_ADVISORY_WARNINGS = "1"
& $PythonExecutable (Join-Path $PSScriptRoot "native_server\app.py")
if ($LASTEXITCODE -ne 0) {
    throw "Native OpenVINO NPU server exited with code $LASTEXITCODE."
}
