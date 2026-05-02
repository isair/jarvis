<#
.SYNOPSIS
    Download and install CUDA libraries for GPU-accelerated speech recognition.

.DESCRIPTION
    Downloads NVIDIA cuBLAS and cuDNN libraries from PyPI wheel packages
    and extracts the DLLs into the target directory. Wheels are just ZIP
    files, so no Python is needed.

    The script is intended to be safe to re-run: a stale marker file from
    a previous half-successful install does not cause us to skip work.
    Every run probes for the expected DLLs first, downloads what's
    missing, verifies SHA256 against the digest PyPI returns, verifies
    that every expected DLL ended up on disk, and only then writes the
    marker. Output is also written to a transcript log so failures from
    Inno Setup's hidden invocation are recoverable.

    Invoked by the Inno Setup installer when the user opts into GPU
    acceleration, by the tray-menu recovery action, or manually:
        powershell -ExecutionPolicy Bypass -File install_cuda.ps1 `
            -TargetDir "C:\Program Files\Jarvis\cuda"

.PARAMETER TargetDir
    Directory to extract CUDA DLLs into (e.g. {app}\cuda).

.PARAMETER LogPath
    Optional path for the transcript log. Defaults to {TargetDir}\install.log.

.PARAMETER PyPIIndexUrl
    Base URL for the PyPI JSON API. Override for testing only.

.PARAMETER SkipGpuCheck
    Skip the local nvcuda.dll check. Used by tests; never set in production.
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$TargetDir,

    [string]$LogPath,

    [string]$PyPIIndexUrl = "https://pypi.org/pypi",

    [switch]$SkipGpuCheck
)

$ErrorActionPreference = "Stop"
# Suppress the progress bar before any Invoke-WebRequest call. With the
# default 'Continue' preference, PowerShell repaints the progress UI on
# every byte, which slows large downloads by 5–10x; the 643 MB cuDNN
# wheel goes from ~3 minutes to half an hour on common connections.
$ProgressPreference = "SilentlyContinue"

# ---------------------------------------------------------------------------
# Package manifest
# ---------------------------------------------------------------------------
# Pinned versions known to work with CTranslate2 4.x (CUDA 12, cuDNN 9).
# `ExpectedDlls` is the list we verify on disk after extraction; if any are
# missing or suspiciously small the install fails loudly instead of leaving
# a stale marker behind.
$packages = @(
    @{
        Name         = "nvidia-cublas-cu12"
        Version      = "12.9.1.4"
        Wheel        = "nvidia_cublas_cu12-12.9.1.4-py3-none-win_amd64.whl"
        Prefix       = "nvidia/cublas/bin/"
        ExpectedDlls = @(
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "nvblas64_12.dll"
        )
    },
    @{
        Name         = "nvidia-cudnn-cu12"
        Version      = "9.20.0.48"
        Wheel        = "nvidia_cudnn_cu12-9.20.0.48-py3-none-win_amd64.whl"
        Prefix       = "nvidia/cudnn/bin/"
        ExpectedDlls = @(
            "cudnn64_9.dll",
            "cudnn_adv64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll",
            "cudnn_graph64_9.dll",
            "cudnn_heuristic64_9.dll",
            "cudnn_ops64_9.dll"
        )
    }
)

# Minimum reasonable size for a CUDA DLL. The smallest real cuDNN file is
# ~260 KB (`cudnn64_9.dll`); anything below this is almost certainly a
# truncated download or an AV stub. Catch this case explicitly so we don't
# write a marker for a corrupt install.
$MIN_DLL_BYTES = 4096

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Get-AllExpectedDlls {
    $names = New-Object System.Collections.Generic.List[string]
    foreach ($pkg in $packages) {
        foreach ($dll in $pkg.ExpectedDlls) {
            $names.Add($dll) | Out-Null
        }
    }
    return ,$names.ToArray()
}

function Test-InstalledDlls {
    param([string]$Dir)

    $missing = New-Object System.Collections.Generic.List[string]
    foreach ($name in (Get-AllExpectedDlls)) {
        $path = Join-Path $Dir $name
        if (-not (Test-Path $path)) {
            $missing.Add($name) | Out-Null
            continue
        }
        $size = (Get-Item $path).Length
        if ($size -lt $MIN_DLL_BYTES) {
            $missing.Add("$name (truncated: $size bytes)") | Out-Null
        }
    }
    return ,$missing.ToArray()
}

function Get-WheelInfo {
    param([string]$PackageName, [string]$Version, [string]$WheelFilename)

    $url = "$PyPIIndexUrl/$PackageName/$Version/json"
    $resp = Invoke-RestMethod -Uri $url -UseBasicParsing -TimeoutSec 60
    foreach ($file in $resp.urls) {
        if ($file.filename -eq $WheelFilename) {
            $sha256 = $null
            if ($file.digests -and $file.digests.sha256) {
                $sha256 = $file.digests.sha256
            }
            return @{ Url = $file.url; Sha256 = $sha256 }
        }
    }
    throw "Wheel $WheelFilename not found on PyPI for $PackageName==$Version"
}

function Test-FileSha256 {
    param([string]$Path, [string]$Expected)

    if ([string]::IsNullOrEmpty($Expected)) {
        # PyPI always returns digests for hosted wheels; if it didn't, fail
        # loudly rather than silently skip the integrity check.
        throw "PyPI did not return a SHA256 digest for $Path"
    }
    $actual = (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $Expected.ToLower()) {
        throw "SHA256 mismatch for $Path (expected $Expected, got $actual)"
    }
}

# ---------------------------------------------------------------------------
# Begin install
# ---------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

if (-not $LogPath) {
    $LogPath = Join-Path $TargetDir "install.log"
}

# Ensure log directory exists, then start a transcript so every line — Write-Host,
# Write-Error, exceptions — lands in the file. The Inno Setup invocation runs
# hidden, so without this a failure is invisible to the user.
$logDir = Split-Path -Parent $LogPath
if ($logDir) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
try {
    Start-Transcript -Path $LogPath -Force | Out-Null
    $transcriptStarted = $true
} catch {
    $transcriptStarted = $false
}

$marker = Join-Path $TargetDir ".cuda_installed"

try {
    # --- Pre-flight: NVIDIA GPU driver detection ---
    if (-not $SkipGpuCheck) {
        $nvcudaPaths = @(
            (Join-Path $env:SystemRoot "System32\nvcuda.dll"),
            (Join-Path $env:windir "System32\nvcuda.dll")
        )
        $gpuFound = $false
        foreach ($p in $nvcudaPaths) {
            if (Test-Path $p) { $gpuFound = $true; break }
        }
        if (-not $gpuFound) {
            Write-Host "No NVIDIA GPU detected, skipping CUDA installation."
            return  # exit 0; no GPU is not a failure
        }
    }

    # --- Idempotence: skip only if every expected DLL is actually on disk ---
    $missing = Test-InstalledDlls -Dir $TargetDir
    if ((Test-Path $marker) -and $missing.Length -eq 0) {
        Write-Host "CUDA libraries already installed and verified."
        return
    }

    if (Test-Path $marker) {
        Write-Host "Stale marker found but DLLs missing/truncated; reinstalling..."
        Write-Host "  Missing: $($missing -join ', ')"
        # Remove the marker up-front so a crash mid-install can't leave a
        # falsely-green state.
        Remove-Item -Force $marker -ErrorAction SilentlyContinue
    }

    Write-Host "Downloading CUDA libraries for GPU acceleration..."
    Write-Host "Target: $TargetDir"
    Write-Host "Log:    $LogPath"

    foreach ($pkg in $packages) {
        Write-Host ""
        Write-Host "Downloading $($pkg.Name) $($pkg.Version)..."

        $info = Get-WheelInfo `
            -PackageName $pkg.Name `
            -Version $pkg.Version `
            -WheelFilename $pkg.Wheel

        $tmpFile = [System.IO.Path]::GetTempFileName() + ".whl"

        try {
            # Use Invoke-WebRequest: it's slower than WebClient on some
            # systems but it raises on truncation rather than silently
            # writing a partial file, which is the documented WebClient
            # failure mode that motivated this rewrite.
            Invoke-WebRequest -Uri $info.Url -OutFile $tmpFile -UseBasicParsing -TimeoutSec 600
            Write-Host "  Download complete."

            Test-FileSha256 -Path $tmpFile -Expected $info.Sha256
            Write-Host "  SHA256 verified."

            Write-Host "  Extracting DLLs..."
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            $zip = [System.IO.Compression.ZipFile]::OpenRead($tmpFile)
            try {
                foreach ($entry in $zip.Entries) {
                    if ($entry.FullName.StartsWith($pkg.Prefix) -and $entry.FullName.EndsWith(".dll")) {
                        $destPath = Join-Path $TargetDir $entry.Name
                        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
                        Write-Host "    $($entry.Name)"
                    }
                }
            } finally {
                $zip.Dispose()
            }
        } finally {
            if (Test-Path $tmpFile) {
                Remove-Item $tmpFile -Force -ErrorAction SilentlyContinue
            }
        }
    }

    # --- Post-extract verification ---
    $missingAfter = Test-InstalledDlls -Dir $TargetDir
    if ($missingAfter.Length -gt 0) {
        throw "Verification failed after extract; missing/truncated: $($missingAfter -join ', ')"
    }

    # --- Marker is the LAST thing written ---
    $markerContent = $packages | ForEach-Object { "$($_.Name)==$($_.Version)" }
    $markerContent | Out-File -FilePath $marker -Encoding utf8

    Write-Host ""
    Write-Host "CUDA libraries installed successfully!"

} catch {
    Write-Host ""
    Write-Host "CUDA installation FAILED: $_"
    Write-Host "See transcript at $LogPath"
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
    exit 1
} finally {
    if ($transcriptStarted) {
        try { Stop-Transcript | Out-Null } catch { }
    }
}
