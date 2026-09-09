[CmdletBinding()]
param(
    [string]$DockerPath = $env:DOCKER_PATH,
    [string]$ComposeFile = (Join-Path $PSScriptRoot "docker-compose.yml"),
    [int]$Port = 6333,
    [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"

if ($Port -le 0) {
    throw "Qdrant port must be greater than zero."
}
if ($TimeoutSeconds -le 0) {
    throw "Qdrant timeout must be greater than zero seconds."
}
if (-not (Test-Path -LiteralPath $ComposeFile)) {
    throw "Qdrant Compose file was not found: $ComposeFile"
}

if ($DockerPath) {
    if (-not (Test-Path -LiteralPath $DockerPath)) {
        throw "Docker executable was not found: $DockerPath"
    }
    $docker = $DockerPath
} else {
    $dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerCommand) {
        throw "Docker was not found on PATH. Start Docker Desktop before launching Qdrant."
    }
    $docker = $dockerCommand.Source
    if (-not $docker) {
        $docker = $dockerCommand.Name
    }
}

$healthUrl = "http://127.0.0.1:$Port/healthz"

function Test-QdrantReady([string]$Uri) {
    try {
        $response = Invoke-WebRequest -Uri $Uri -Method Get -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Test-LocalPortInUse([int]$LocalPort) {
    try {
        return @(
            Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue
        ).Count -gt 0
    } catch {
        return $false
    }
}

if (Test-QdrantReady $healthUrl) {
    Write-Host "Qdrant is already ready at $healthUrl; reusing it." -ForegroundColor Green
    exit 0
}

if (Test-LocalPortInUse $Port) {
    throw "Qdrant is not healthy at $healthUrl, but local port $Port is occupied. Refusing to start a duplicate or unrelated service."
}

Write-Host "Starting Qdrant only from $ComposeFile (Docker OVMS is not started)." -ForegroundColor Cyan
& $docker compose -f $ComposeFile up -d qdrant
if ($LASTEXITCODE -ne 0) {
    throw "Failed to start the Qdrant Compose service."
}

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if (Test-QdrantReady $healthUrl) {
        Write-Host "Qdrant is ready at $healthUrl." -ForegroundColor Green
        exit 0
    }
    Start-Sleep -Seconds 2
}

throw "Qdrant did not become ready at $healthUrl within $TimeoutSeconds seconds."
