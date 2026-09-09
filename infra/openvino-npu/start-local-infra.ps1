[CmdletBinding()]
param(
    [switch]$SkipModelProbe,
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$composeFile = Join-Path $PSScriptRoot "docker-compose.yml"
$modelFile = Join-Path $PSScriptRoot "models\Qwen3-Embedding-0.6B-int4-cw-ov\1\openvino_model.xml"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found on PATH. Start Docker Desktop with WSL2/OpenVINO NPU support first."
}
if (-not (Test-Path -LiteralPath $modelFile)) {
    throw "OpenVINO model is missing: $modelFile. Run prepare-bge-openvino.ps1 to download the pinned Qwen OpenVINO IR artifacts."
}

docker compose -f $composeFile up -d
if ($LASTEXITCODE -ne 0) { throw "Failed to start OVMS and Qdrant." }

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$ovmsReady = $false
$qdrantReady = $false
while ((Get-Date) -lt $deadline -and (-not $ovmsReady -or -not $qdrantReady)) {
    try {
        $ready = Invoke-WebRequest -Uri "http://127.0.0.1:8000/v2/health/ready" -TimeoutSec 3 -UseBasicParsing
        $ovmsReady = $ready.StatusCode -eq 200
    } catch { $ovmsReady = $false }
    try {
        $health = Invoke-WebRequest -Uri "http://127.0.0.1:6333/healthz" -TimeoutSec 3 -UseBasicParsing
        $qdrantReady = $health.StatusCode -eq 200
    } catch { $qdrantReady = $false }
    if (-not $ovmsReady -or -not $qdrantReady) { Start-Sleep -Seconds 2 }
}

if (-not $ovmsReady) { throw "OVMS did not become ready at http://127.0.0.1:8000." }
if (-not $qdrantReady) { throw "Qdrant did not become ready at http://127.0.0.1:6333." }

$modelLoaded = $true
$latencyMs = 0
$dimensions = 0
if (-not $SkipModelProbe) {
    $body = @{ model = "Qwen3-Embedding-0.6B-int4-cw-ov"; input = @("Instruct: Given a software-engineering task, retrieve code passages, symbols, tests, configuration, and documentation relevant to implementing, debugging, or verifying the requested change.\nQuery:Kelvin Clyne NPU connectivity probe") } | ConvertTo-Json -Depth 5
    $started = Get-Date
    try {
        $embedding = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v3/embeddings" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 30
        $latencyMs = [int]((Get-Date) - $started).TotalMilliseconds
        $dimensions = @($embedding.data[0].embedding).Count
        $modelLoaded = $dimensions -gt 0
    } catch {
        $modelLoaded = $false
        Write-Warning "OVMS readiness succeeded but the embedding probe failed: $($_.Exception.Message)"
    }
}

$device = "NPU (declared by OVMS target_device; verify container logs for physical device binding)"
Write-Host ""
Write-Host "Embedding Runtime" -ForegroundColor Cyan
Write-Host "  Endpoint:     $(if ($ovmsReady) { 'Connected' } else { 'Unavailable' })"
Write-Host "  Model:        $(if ($modelLoaded) { 'Loaded' } else { 'Probe failed' })"
Write-Host "  Device:       $device"
Write-Host "  Dimensions:   $dimensions"
Write-Host "  Latency:      $latencyMs ms"
Write-Host "  Batch limit:  500 inputs/request; scheduler batch 8"
Write-Host ""
Write-Host "Context Injection Budget" -ForegroundColor Cyan
Write-Host "  8,000 tokens"
Write-Host "Maximum Search Results" -ForegroundColor Cyan
Write-Host "  50 candidates"
Write-Host "Maximum Injected Results" -ForegroundColor Cyan
Write-Host "  8 chunks"
Write-Host ""
Write-Host "Qdrant:         $(if ($qdrantReady) { 'Connected at http://127.0.0.1:6333' } else { 'Unavailable' })"
