<#
.SYNOPSIS
    Stop the canonical Qwen3-VL Intel Xe (GPU.0) vision lane.

.DESCRIPTION
    Stops the gateway on 8102 first, then the backend on 20000. Both stops are
    scoped to the tracked PID and the port owner, so the CPU lane on 8103/20002
    is left alone.

.EXAMPLE
    .\scripts\services\Stop-Qwen3VlXe.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
. (Join-Path (Split-Path -Parent $PSCommandPath) "lib\LocalService.Common.ps1")

Stop-VisionLane `
    -LaneName "Qwen3-VL Intel Xe (GPU.0)" `
    -GatewayScript "Stop-VisionGateway.ps1" `
    -BackendScript "Stop-VisionBackend.ps1"