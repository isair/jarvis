<#
.SYNOPSIS
    Start the Qdrant vector database (BLACKGLASS / IdeaWell substrate).

.DESCRIPTION
    Delegates to the canonical launcher infra/openvino-npu/start-qdrant.ps1, which
    brings up the `qdrant` Compose service and waits for readiness. Qdrant
    persists to the named Docker volume `qdrant_storage`, so evidence survives
    container restarts.

.EXAMPLE
    .\scripts\services\Start-Qdrant.ps1
    .\scripts\services\Start-Qdrant.ps1 -Port 6333
#>
[CmdletBinding()]
param(
    [string]$DockerPath = $env:DOCKER_PATH,
    [int]$Port = 6333,
    [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $PSCommandPath
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptRoot)
$launcher = Join-Path $repoRoot "infra\openvino-npu\start-qdrant.ps1"

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Canonical Qdrant launcher not found: $launcher"
}

Write-Host "Starting Qdrant via canonical launcher: $launcher (port $Port)" -ForegroundColor Cyan
& $launcher -DockerPath $DockerPath -Port $Port -TimeoutSeconds $TimeoutSeconds
if ($LASTEXITCODE -ne 0) { throw "Qdrant launcher failed ($LASTEXITCODE)" }
Write-Host "Qdrant ready at http://127.0.0.1:$Port/healthz (REST $Port, gRPC $($Port + 1))." -ForegroundColor Green
