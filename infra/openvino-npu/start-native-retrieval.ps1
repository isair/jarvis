[CmdletBinding()]
param(
    [string]$OpenVinoRoot = "",
    [string]$Python = "",
    [int]$Port = 8010,
    [switch]$NewWindow,
    [switch]$UseCurrentUserProfile
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$defaultOpenVinoRoot = "C:\Intel\openvino_2026.2.1\openvino_toolkit_windows_2026.2.1.21919.ede283a88e3_x86_64"
if (-not $OpenVinoRoot) {
    $OpenVinoRoot = if ($env:OPENVINO_ROOT) { $env:OPENVINO_ROOT } else { $defaultOpenVinoRoot }
}
$setup = Join-Path $OpenVinoRoot "setupvars.ps1"
if (-not (Test-Path -LiteralPath $setup)) { throw "OpenVINO setupvars.ps1 was not found: $setup" }

$profileArguments = if ($UseCurrentUserProfile) { @() } else { @("-NoProfile") }
$arguments = @("-NoLogo") + $profileArguments + @("-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "start-native-server.ps1"), "-OpenVinoRoot", $OpenVinoRoot, "-Python", $Python, "-Port", $Port)
if ($NewWindow) {
    Start-Process pwsh -ArgumentList $arguments -WorkingDirectory $root
    Write-Host "Started native Kelvin retrieval service in a new window on port $Port"
    exit 0
}
& pwsh @arguments
