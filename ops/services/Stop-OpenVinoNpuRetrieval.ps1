<#
.SYNOPSIS
    Stop the native OpenVINO NPU embeddings/reranker server.

.EXAMPLE
    .\scripts\services\Stop-OpenVinoNpuRetrieval.ps1
#>
[CmdletBinding()]
param(
    [int]$Port = 8010
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $PSCommandPath
. (Join-Path $scriptRoot "lib\LocalService.Common.ps1")

Write-Host "Stopping OpenVINO NPU retrieval (port $Port)..." -ForegroundColor Cyan
$stopped = Stop-ServiceProcesses -Patterns @('start-native-retrieval\.ps1', 'native_retrieval', 'npu.*server\.py') -DisplayName "OpenVINO NPU retrieval"
$portReleased = Stop-PortOwner -LocalPort $Port
if ($stopped -and $portReleased) {
    Write-Host "OpenVINO NPU retrieval stopped." -ForegroundColor Green
} else {
    throw "OpenVINO NPU retrieval did not fully stop; manual intervention required."
}
