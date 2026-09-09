<#
.SYNOPSIS
    Start the native OpenVINO NPU embeddings/reranker server as a detached daemon.

.DESCRIPTION
    Delegates to infra/openvino-npu/start-native-retrieval.ps1 in a detached
    PowerShell window. Runs on the native Windows OpenVINO runtime for NPU
    access (not Docker). Reuses a healthy existing instance.

.EXAMPLE
    .\scripts\services\Start-OpenVinoNpuRetrieval.ps1
    .\scripts\services\Start-OpenVinoNpuRetrieval.ps1 -Port 8010
#>
[CmdletBinding()]
param(
    [string]$OpenVinoRoot = $(if ([string]::IsNullOrWhiteSpace($env:OPENVINO_ROOT)) { 'C:\Intel\openvino_2026.2.1\openvino_toolkit_windows_2026.2.1.21919.ede283a88e3_x86_64' } else { $env:OPENVINO_ROOT }),
    [string]$Python = $env:OPENVINO_PYTHON,
    [int]$Port = 8010,
    [switch]$UseCurrentUserProfile
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $PSCommandPath
. (Join-Path $scriptRoot "lib\LocalService.Common.ps1")

$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptRoot)
$launcher = Join-Path $repoRoot "infra\openvino-npu\start-native-retrieval.ps1"

$healthUrl = "http://127.0.0.1:$Port/health"
if (-not (Assert-ServiceNotRunning "OpenVINO NPU retrieval" $healthUrl $Port)) { return }

Write-Host "Starting OpenVINO NPU embeddings/reranker: $launcher (port $Port)" -ForegroundColor Cyan
Start-DetachedService -ScriptPath $launcher -WorkingDirectory $repoRoot `
    -UseCurrentUserProfile:$UseCurrentUserProfile -Parameters @{
        OpenVinoRoot = $OpenVinoRoot
        Python       = $Python
        Port         = $Port
    }
Write-Host "NPU retrieval start requested; health endpoint: $healthUrl" -ForegroundColor Yellow
