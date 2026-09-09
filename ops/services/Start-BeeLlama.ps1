<#
.SYNOPSIS
    Start the BeeLlama local LLM server (llama.cpp) as a detached daemon.

.DESCRIPTION
    Delegates to the canonical launcher scripts/local-model/start-server.ps1 in a
    detached PowerShell window so the server survives the calling process
    (VS Code, extension host, terminal). Reuses a healthy existing instance.

.EXAMPLE
    .\scripts\services\Start-BeeLlama.ps1
    .\scripts\services\Start-BeeLlama.ps1 -LlamaPreset 128k -Port 8888
#>
[CmdletBinding()]
param(
    [Alias("LlamaModel")]
    [string]$ModelPath = $env:LOCAL_LLM_MODEL_PATH,
    [Alias("Preset")]
    [ValidateSet("128k", "96k-mtp", "64k-mtp", "parallel-2x64k", "parallel-2x48k")]
    [string]$LlamaPreset = "96k-mtp",
    [Alias("LlamaCppPath")]
    [string]$BeellamaPath = $env:BEELLAMA_PATH,
    [string]$ServerHost = $env:LOCAL_LLM_HOST,
    [int]$Port = 8888,
    [int]$ContextSize = 0,
    [int]$GpuLayers = 999,
    [string]$CacheTypeK = "",
    [string]$CacheTypeV = "",
    [switch]$EnableMTP,
    [int]$ParallelSlots = 0,
    [int]$MaxOutputTokens = 16384,
    [switch]$UseCurrentUserProfile
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $PSCommandPath
. (Join-Path $scriptRoot "lib\LocalService.Common.ps1")

$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptRoot)
$launcher = Join-Path $repoRoot "scripts\local-model\start-server.ps1"

$llamaPort = if ($Port -gt 0) { $Port } else { 8888 }
$healthUrl = "http://127.0.0.1:$llamaPort/health"
if (-not (Assert-ServiceNotRunning "BeeLlama" $healthUrl $llamaPort)) { return }

Write-Host "Starting BeeLlama (llama.cpp): $launcher (port $llamaPort, preset $LlamaPreset)" -ForegroundColor Cyan
Start-DetachedService -ScriptPath $launcher -WorkingDirectory (Split-Path -Parent $repoRoot) `
    -UseCurrentUserProfile:$UseCurrentUserProfile -Parameters @{
        ModelPath       = $ModelPath
        Preset          = $LlamaPreset
        LlamaCppPath    = $BeellamaPath
        ServerHost      = $ServerHost
        Port            = $llamaPort
        ContextSize     = $ContextSize
        GpuLayers       = $GpuLayers
        CacheTypeK      = $CacheTypeK
        CacheTypeV      = $CacheTypeV
        EnableMTP       = $EnableMTP.IsPresent
        ParallelSlots   = $ParallelSlots
        MaxOutputTokens = $MaxOutputTokens
    }
Write-Host "BeeLlama start requested; health endpoint: $healthUrl" -ForegroundColor Yellow
