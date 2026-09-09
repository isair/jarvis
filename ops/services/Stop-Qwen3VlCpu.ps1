<#
.SYNOPSIS
    Stop the on-demand Qwen3-VL CPU vision lane.

.DESCRIPTION
    Stops the gateway on 8103 first, then the backend on 20002. Scoped to the
    tracked PID and port owner, so the resident Intel Xe lane on 8102/20000 is
    unaffected.

    The CPU backend holds roughly 17 GB resident, so stopping this lane when it
    is not in use returns a material amount of memory.

.EXAMPLE
    .\scripts\services\Stop-Qwen3VlCpu.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
. (Join-Path (Split-Path -Parent $PSCommandPath) "lib\LocalService.Common.ps1")

Stop-VisionLane `
    -LaneName "Qwen3-VL CPU (on demand)" `
    -GatewayScript "Stop-VisionCpuGateway.ps1" `
    -BackendScript "Stop-VisionCpuBackend.ps1"