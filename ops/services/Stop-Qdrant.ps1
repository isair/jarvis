<#
.SYNOPSIS
    Stop the Qdrant vector database Compose service.

.DESCRIPTION
    Stops the `qdrant` service from infra/openvino-npu/docker-compose.yml via
    `docker compose stop`. Data persists in the named volume `qdrant_storage`.

.EXAMPLE
    .\scripts\services\Stop-Qdrant.ps1
#>
[CmdletBinding()]
param(
    [string]$DockerPath = $env:DOCKER_PATH,
    [string]$ComposeFile,
    [int]$TimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $PSCommandPath
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptRoot)
if ([string]::IsNullOrWhiteSpace($ComposeFile)) {
    $ComposeFile = Join-Path $repoRoot "infra\openvino-npu\docker-compose.yml"
}
if (-not (Test-Path -LiteralPath $ComposeFile)) {
    throw "Qdrant Compose file was not found: $ComposeFile"
}

if ($DockerPath) {
    if (-not (Test-Path -LiteralPath $DockerPath)) { throw "Docker executable was not found: $DockerPath" }
    $docker = $DockerPath
} else {
    $dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerCommand) { throw "Docker was not found on PATH." }
    $docker = $dockerCommand.Source
}

Write-Host "Stopping Qdrant Compose service from $ComposeFile..." -ForegroundColor Cyan
& $docker compose -f $ComposeFile stop --timeout $TimeoutSeconds qdrant
if ($LASTEXITCODE -ne 0) { throw "Failed to stop the Qdrant Compose service ($LASTEXITCODE)." }
Write-Host "Qdrant stopped. Data persists in Docker volume 'qdrant_storage'." -ForegroundColor Green
