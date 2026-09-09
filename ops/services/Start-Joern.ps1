<#
.SYNOPSIS
    Verify (and optionally install) Joern — the BLACKGLASS reachability plane.

.DESCRIPTION
    Joern is the deterministic reachability backend for BLACKGLASS (spec §10/§11).
    Without it the fabric falls back to UnavailableCodeGraphService, and per §35
    that means reachability is reported as *unknown* — never as "not reachable".
    A vulnerable dependency that is present but unproven-reachable stays a
    HYPOTHESIS, so having Joern installed is what lets a chain reach
    STATIC_CONFIRMED at all.

    This script does NOT start a server, because nothing in this repository talks
    to one. JoernCodeGraphAdapter invokes the binary per query with an argument
    array and a timeout (see program\JoernCodeGraphAdapter.ts). An earlier
    revision of this script launched `joern --server` on a TCP port; that port was
    a fiction twice over — the fabric never connected to it, and the number picked
    was already contested on a normal dev box (8080 is SearXNG here; BeeLlama is
    pinned to 8888). A green
    light on a socket nobody dials is worse than no light at all.

    So "starting" Joern means: confirm a JDK, confirm the binary runs, and record
    where it is. When Joern is not installed at all, this script downloads the
    latest release from GitHub and extracts it into %LOCALAPPDATA%\Kelvin-Clyne\
    blackglass\joern-cli, then persists the resolved binary into the
    ZOO_BLACKGLASS_JOERN_BIN environment variable at User scope. The BLACKGLASS
    backend resolver runs once per activation, so a freshly installed Joern
    attaches on the next window reload.

.PARAMETER JoernHome
    Directory containing joern / joern.bat. Defaults to $env:JOERN_HOME, then to
    whatever `joern` resolves to on PATH.

.PARAMETER Persist
    Record the resolved binary in ZOO_BLACKGLASS_JOERN_BIN at User scope, so the
    extension uses this exact Joern rather than re-guessing from PATH. Also clears
    ZOO_BLACKGLASS_DISABLE_JOERN if a previous Stop set it. Defaults to true when
    a fresh install is performed (the Settings button does not pass -Persist, but
    pinning after an install is the only sane default).

.PARAMETER InstallDir
    Where to extract the Joern distribution when downloading. Defaults to
    %LOCALAPPDATA%\Kelvin-Clyne\blackglass\joern-cli.

.EXAMPLE
    .\scripts\services\Start-Joern.ps1
    .\scripts\services\Start-Joern.ps1 -JoernHome 'C:\tools\joern' -Persist
#>
[CmdletBinding()]
param(
    [string]$JoernHome = $env:JOERN_HOME,
    [switch]$Persist,
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "Kelvin-Clyne\blackglass\joern-cli"),
    [int]$ProbeTimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"

# ── Require a JDK first ──────────────────────────────────────────────────────
# Checked before the launcher, because a missing `java` produces a stack trace
# from inside the shim, which reads as "Joern is broken" rather than "Joern has
# no JVM to run on".
$java = Get-Command "java" -ErrorAction SilentlyContinue
if (-not $java) {
    throw "Joern requires a JDK (17+) on PATH; 'java' was not found. Install a JDK, then re-run this script."
}
Write-Host "java: $($java.Source)" -ForegroundColor DarkGray

# ── Locate the binary ────────────────────────────────────────────────────────
# Prefer an explicit JOERN_HOME. A PATH hit is accepted but reported, because a
# shim resolved from PATH may point at a different Joern version than the one
# whose analyzer version gets stamped into every projection (§11.2).
$binary = $null
if ($env:ZOO_BLACKGLASS_JOERN_BIN -and (Test-Path -LiteralPath $env:ZOO_BLACKGLASS_JOERN_BIN)) {
    $binary = $env:ZOO_BLACKGLASS_JOERN_BIN
    Write-Host "Using ZOO_BLACKGLASS_JOERN_BIN: $binary" -ForegroundColor DarkGray
}
if (-not $binary -and $JoernHome -and (Test-Path -LiteralPath $JoernHome)) {
    foreach ($candidate in @("joern.bat", "joern.cmd", "joern")) {
        $probe = Join-Path $JoernHome $candidate
        if (Test-Path -LiteralPath $probe) { $binary = $probe; break }
    }
}
if (-not $binary) {
    $onPath = Get-Command "joern" -ErrorAction SilentlyContinue
    if ($onPath) {
        $binary = $onPath.Source
        Write-Host "Using joern from PATH: $binary" -ForegroundColor DarkGray
    }
}
if (-not $binary) {
    # ── Auto-install from GitHub releases ────────────────────────────────
    # Joern publishes platform-specific zips. On Windows the asset is
    # joern-cli-windows-x86_64.zip (or arm64). The zip contains a top-level
    # joern-cli/ directory with joern.bat inside.
    Write-Host ""
    Write-Host "Joern was not found on this machine." -ForegroundColor Yellow
    Write-Host "Downloading the latest release from GitHub..." -ForegroundColor Cyan
    Write-Host ""

    # Determine platform asset name.
    $arch = if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq [System.Runtime.InteropServices.Architecture]::Arm64) { "arm64" } else { "x86_64" }
    $assetName = "joern-cli-windows-$arch.zip"

    # Query the GitHub Releases API for the latest release.
    $apiUrl = "https://api.github.com/repos/joernio/joern/releases/latest"
    Write-Host "Fetching latest release metadata from $apiUrl ..." -ForegroundColor DarkGray
    try {
        $headers = @{ "Accept" = "application/vnd.github+json"; "User-Agent" = "Kelvin-Clyne-Joern-Installer/1.0" }
        $release = Invoke-RestMethod -Uri $apiUrl -Headers $headers -TimeoutSec 30
    } catch {
        throw "Failed to query GitHub Releases API: $_`n`nManual install: https://docs.joern.io/installation"
    }
    $asset = $release.assets | Where-Object { $_.name -eq $assetName } | Select-Object -First 1
    if (-not $asset) {
        $available = ($release.assets | ForEach-Object { $_.name }) -join ", "
        throw "Asset '$assetName' not found in release $($release.tag_name). Available: $available`n`nManual install: https://docs.joern.io/installation"
    }
    $downloadUrl = $asset.browser_download_url
    $tagName = $release.tag_name
    Write-Host "  Release : $tagName" -ForegroundColor DarkGray
    Write-Host "  Asset   : $assetName" -ForegroundColor DarkGray
    Write-Host "  Size    : $([math]::Round($asset.size / 1MB, 1)) MB" -ForegroundColor DarkGray

    # Prepare install directory.
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $zipPath = Join-Path $InstallDir "$assetName"

    # Download.
    Write-Host ""
    Write-Host "Downloading $downloadUrl ..." -ForegroundColor Cyan
    Write-Host "  (Joern is ~1.7 GB; this may take a few minutes)" -ForegroundColor DarkGray
    try {
        # Use BITS for resume support and progress, with a web-request fallback.
        $useBits = $true
        try { Import-Module BitsTransfer -ErrorAction Stop } catch { $useBits = $false }
        if ($useBits) {
            Start-BitsTransfer -Source $downloadUrl -Destination $zipPath -Description "Downloading Joern $tagName"
        } else {
            $ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest progress bar is catastrophically slow
            Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -UseBasicParsing -TimeoutSec 600
        }
    } catch {
        Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
        throw "Download failed: $_`n`nManual install: https://docs.joern.io/installation"
    }
    if (-not (Test-Path -LiteralPath $zipPath)) {
        throw "Download completed but zip file not found at $zipPath"
    }
    Write-Host "  Download complete." -ForegroundColor Green

    # Extract. The zip contains a top-level joern-cli/ directory.
    # Remove any previous extraction first to avoid stale files.
    $extractTarget = $InstallDir
    $joernCliDir = Join-Path $extractTarget "joern-cli"
    if (Test-Path -LiteralPath $joernCliDir) {
        Write-Host "Removing previous installation at $joernCliDir ..." -ForegroundColor DarkGray
        Remove-Item -LiteralPath $joernCliDir -Recurse -Force
    }
    Write-Host "Extracting to $extractTarget ..." -ForegroundColor Cyan
    try {
        Expand-Archive -LiteralPath $zipPath -DestinationPath $extractTarget -Force
    } catch {
        throw "Extraction failed: $_`n`nThe downloaded zip may be corrupt. Delete $zipPath and retry."
    }
    # Clean up the zip to reclaim disk space.
    Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue

    # Locate the binary inside the extracted distribution.
    $binary = $null
    foreach ($candidate in @("joern.bat", "joern.cmd", "joern")) {
        $probe = Join-Path $joernCliDir $candidate
        if (Test-Path -LiteralPath $probe) { $binary = $probe; break }
    }
    if (-not $binary) {
        throw "Extraction succeeded but no joern binary found in $joernCliDir.`nContents: $(Get-ChildItem -LiteralPath $joernCliDir -Name | Select-Object -First 20)"
    }
    Write-Host "  Installed: $binary" -ForegroundColor Green
    Write-Host ""

    # A fresh install always persists — the Settings button calls this script
    # without -Persist, and requiring the user to know about an env var flag
    # after a successful automated install is hostile.
    $Persist = [switch]$true
}

# ── Prove it runs ────────────────────────────────────────────────────────────
# The same probe the extension performs (probeJoern -> `joern --version`), so the
# Integrations panel and this script cannot disagree about whether the plane is
# usable. A cold JVM plus Joern's Scala REPL warm-up is genuinely slow, hence the
# generous default timeout.
Write-Host "Probing $binary --version (this can take a minute on a cold JVM)..." -ForegroundColor Cyan
$stdoutFile = [System.IO.Path]::GetTempFileName()
$stderrFile = [System.IO.Path]::GetTempFileName()
try {
    $proc = Start-Process -FilePath $binary -ArgumentList @("--version") -NoNewWindow -PassThru `
        -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
    if (-not $proc.WaitForExit($ProbeTimeoutSeconds * 1000)) {
        try { $proc.Kill($true) } catch { }
        throw "joern --version did not finish within $ProbeTimeoutSeconds seconds. The install may be incomplete."
    }
    $version = @(
        (Get-Content -LiteralPath $stdoutFile -ErrorAction SilentlyContinue)
        (Get-Content -LiteralPath $stderrFile -ErrorAction SilentlyContinue)
    ) | Where-Object { $_ -and $_.Trim() } | Select-Object -First 1
    if ($proc.ExitCode -ne 0) {
        throw "joern --version exited $($proc.ExitCode). Output: $version"
    }
} finally {
    Remove-Item -LiteralPath $stdoutFile, $stderrFile -Force -ErrorAction SilentlyContinue
}

Write-Host "Joern is usable: $($version.Trim())" -ForegroundColor Green

# The CPG working directory. Created here rather than lazily so a permissions
# problem surfaces now, in a window the user is already looking at, instead of
# mid-query inside the extension host.
$workDir = Join-Path $env:LOCALAPPDATA "Kelvin-Clyne\blackglass\joern"
New-Item -ItemType Directory -Force -Path $workDir | Out-Null
Write-Host "  CPG workdir: $workDir" -ForegroundColor DarkGray

if ($Persist) {
    [Environment]::SetEnvironmentVariable("ZOO_BLACKGLASS_JOERN_BIN", $binary, "User")
    [Environment]::SetEnvironmentVariable("ZOO_BLACKGLASS_DISABLE_JOERN", $null, "User")
    Write-Host "Pinned ZOO_BLACKGLASS_JOERN_BIN=$binary (User scope)." -ForegroundColor Green
}

Write-Host ""
Write-Host "BLACKGLASS resolves its backends once per activation." -ForegroundColor Yellow
Write-Host "Reload the VS Code window for the reachability plane to attach." -ForegroundColor Yellow
