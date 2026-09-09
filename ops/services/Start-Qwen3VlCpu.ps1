<#
.SYNOPSIS
    Start the on-demand Qwen3-VL vision lane on the CPU.

.DESCRIPTION
    Same model and server as the Intel Xe lane, pinned to the CPU device and to
    its own ports so both lanes can run simultaneously:

        backend 20002, gateway 8103

    This lane is on demand: nothing auto-starts it. It is useful as a second
    engine for batch work, and as a correctness oracle, because the CPU plugin
    bounds-checks the DeepStack scatter that the GPU plugin does not.

    It is materially slower than the Xe lane for interactive use - measured on
    this stack at roughly 11 s per frame versus 9 s, and it degrades a further
    ~45% while the Xe lane is saturated, since both compete for the same memory
    bandwidth.

.EXAMPLE
    .\scripts\services\Start-Qwen3VlCpu.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
. (Join-Path (Split-Path -Parent $PSCommandPath) "lib\LocalService.Common.ps1")

Start-VisionLane `
    -LaneName "Qwen3-VL CPU (on demand)" `
    -BackendScript "Start-VisionCpuBackend.ps1" `
    -GatewayScript "Start-VisionCpuGateway.ps1" `
    -BackendPort 20002 `
    -GatewayPort 8103