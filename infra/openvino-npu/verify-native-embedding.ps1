[CmdletBinding()]
param(
    [string]$OpenVinoRoot = "C:\Intel\openvino_2026.2.1\openvino_toolkit_windows_2026.2.1.21919.ede283a88e3_x86_64",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$setup = Join-Path $OpenVinoRoot "setupvars.ps1"
if (-not (Test-Path -LiteralPath $setup)) { throw "OpenVINO setupvars.ps1 was not found: $setup" }
& $setup
& $Python (Join-Path $PSScriptRoot "native_embedding_runtime.py")
if ($LASTEXITCODE -ne 0) { throw "Native NPU embedding acceptance failed." }
