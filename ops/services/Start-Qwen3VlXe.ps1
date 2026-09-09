<#
.SYNOPSIS
    Start the canonical Qwen3-VL vision lane on the Intel Xe iGPU (GPU.0).

.DESCRIPTION
    Starts the two processes that Vision Angels actually talks to:

        1. openvino_vlm_server (OpenVINO GenAI ContinuousBatching) on 20000,
           device GPU.0
        2. qwen3_vl_gateway (OpenAI-compatible) on 8102

    The backend is awaited before the gateway is launched, so the readiness
    probe does not observe a gateway proxying to a model that is still loading.

    This replaces the earlier `qwen3-vl-intel-xe-openvino` entry, which pointed
    at an ONNX launcher retired on 2026-08-21 while still claiming ports 8102 and
    20000 - so its row reported this lane's health, and its Stop button tore this
    lane down.

.EXAMPLE
    .\scripts\services\Start-Qwen3VlXe.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
. (Join-Path (Split-Path -Parent $PSCommandPath) "lib\LocalService.Common.ps1")

Start-VisionLane `
    -LaneName "Qwen3-VL Intel Xe (GPU.0)" `
    -BackendScript "Start-VisionBackend.ps1" `
    -GatewayScript "Start-VisionGateway.ps1" `
    -BackendPort 20000 `
    -GatewayPort 8102
