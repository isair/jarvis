<#
.SYNOPSIS
    Download and install CUDA libraries for GPU-accelerated speech recognition.

.DESCRIPTION
    Downloads NVIDIA cuBLAS and cuDNN libraries from PyPI wheel packages
    and extracts the DLLs into the target directory. Wheels are just ZIP
    files, so no Python is needed.

    Invoked by the Inno Setup installer when the user opts into GPU
    acceleration, or can be run manually:
        powershell -ExecutionPolicy Bypass -File install_cuda.ps1 -TargetDir "C:\Program Files\Jarvis\cuda"

.PARAMETER TargetDir
    Directory to extract CUDA DLLs into (e.g. {app}\cuda).
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$TargetDir
)

$ErrorActionPreference = "Stop"

# Pinned versions known to work with CTranslate2 4.x (CUDA 12, cuDNN 9)
$packages = @(
    @{
        Name    = "nvidia-cublas-cu12"
        Version = "12.9.1.4"
        Wheel   = "nvidia_cublas_cu12-12.9.1.4-py3-none-win_amd64.whl"
        Prefix  = "nvidia/cublas/bin/"
    },
    @{
        Name    = "nvidia-cudnn-cu12"
        Version = "9.20.0.48"
        Wheel   = "nvidia_cudnn_cu12-9.20.0.48-py3-none-win_amd64.whl"
        Prefix  = "nvidia/cudnn/bin/"
    }
)

function Get-WheelUrl {
    param([string]$PackageName, [string]$Version, [string]$WheelFilename)

    $url = "https://pypi.org/pypi/$PackageName/$Version/json"
    $resp = Invoke-RestMethod -Uri $url -UseBasicParsing
    foreach ($file in $resp.urls) {
        if ($file.filename -eq $WheelFilename) {
            return $file.url
        }
    }
    throw "Wheel $WheelFilename not found on PyPI for $PackageName==$Version"
}

# Check for NVIDIA GPU via the CUDA driver DLL
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
    exit 0
}

# Check if already installed
$marker = Join-Path $TargetDir ".cuda_installed"
if (Test-Path $marker) {
    Write-Host "CUDA libraries already installed."
    exit 0
}

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

Write-Host "Downloading CUDA libraries for GPU acceleration..."
Write-Host "Target: $TargetDir"

foreach ($pkg in $packages) {
    Write-Host ""
    Write-Host "Downloading $($pkg.Name) $($pkg.Version)..."

    $url = Get-WheelUrl -PackageName $pkg.Name -Version $pkg.Version -WheelFilename $pkg.Wheel

    $tmpFile = [System.IO.Path]::GetTempFileName() + ".whl"

    try {
        # Download with progress
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($url, $tmpFile)
        Write-Host "  Download complete."

        # Extract DLLs (wheels are ZIP files)
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
            Remove-Item $tmpFile -Force
        }
    }
}

# Write marker file
$markerContent = $packages | ForEach-Object { "$($_.Name)==$($_.Version)" }
$markerContent | Out-File -FilePath $marker -Encoding utf8

Write-Host ""
Write-Host "CUDA libraries installed successfully!"
