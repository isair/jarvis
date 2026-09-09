<#
.SYNOPSIS
Starts the ThinkingCap Qwen3.6-27B BeeLlama server.

.DESCRIPTION
The presets are tuned for an RTX 4090. A two-slot preset is the supported
parallel configuration; do not combine two slots with the 96k or 128k presets
without first validating GPU memory on the target machine.

.PARAMETER Preset
One of 128k, 96k-mtp, 64k-mtp, parallel-2x64k, or parallel-2x48k.

.PARAMETER ModelPath
Path to the GGUF model. Defaults to LOCAL_LLM_MODEL_PATH or the repository
model path.

.PARAMETER BeellamaPath
Path to the BeeLlama checkout containing the CUDA build and llama-server.exe.

.PARAMETER ServerHost
Bind address. Defaults to LOCAL_LLM_HOST or 127.0.0.1.

.PARAMETER Port
HTTP port. Defaults to LOCAL_LLM_PORT or 8888.

.PARAMETER ContextSize
Custom context size when using an unknown preset.

.PARAMETER GpuLayers
Number of GPU layers. Defaults to all layers (999).

.PARAMETER CacheTypeK
Key-cache quantization. Defaults to q8_0.

.PARAMETER CacheTypeV
Value-cache quantization. Defaults to q8_0.

.PARAMETER EnableMTP
Enables draft-mtp speculative decoding when using a custom/unknown preset.

.PARAMETER ParallelSlots
Custom slot count when using an unknown preset.

.PARAMETER MaxOutputTokens
Default maximum generated tokens passed to llama-server as `--n-predict`.
Defaults to 16,384. This is a generation limit, not the context allocation.

.EXAMPLE
.\start-server.ps1 -Preset 96k-mtp

.EXAMPLE
.\start-server.ps1 -Preset parallel-2x64k
#>

param(
    [string]$Preset = "96k-mtp",
    [string]$ModelPath = $env:LOCAL_LLM_MODEL_PATH,
    [Alias("LlamaCppPath")]
    [string]$BeellamaPath = $env:BEELLAMA_PATH,
    [string]$ServerHost = $env:LOCAL_LLM_HOST,
    [int]$Port = 8888,
    [int]$ContextSize = $null,
    [int]$GpuLayers = $null,
    [string]$CacheTypeK = $null,
    [string]$CacheTypeV = $null,
    [switch]$EnableMTP = $true,
    [int]$ParallelSlots = $null,
    [int]$MaxOutputTokens = 16384,
    [switch]$AllowCpuFallback = $false
)

$ErrorActionPreference = "Stop"

Write-Host "=== Starting BeeLlama server for ThinkingCap Qwen3.6-27B ===" -ForegroundColor Cyan

# Define presets
$presets = @{
    "128k" = @{
        ContextSize = 131072
        ParallelSlots = 1
        EnableMTP = $false
        GpuLayers = 999
        Description = "Full 128k context, with MTP (max context)"
    }
    "96k-mtp" = @{
        ContextSize = 98304
        ParallelSlots = 1
        EnableMTP = $true
        GpuLayers = 999
        Description = "96k context with MTP enabled (balanced speed/context)"
    }
    "64k-mtp" = @{
        ContextSize = 65536
        ParallelSlots = 1
        GpuLayers = 999
        EnableMTP = $true
        Description = "64k context with MTP enabled (fast generation)"
    }
    "parallel-2x64k" = @{
        ContextSize = 65536
        ParallelSlots = 2
        GpuLayers = 999
        EnableMTP = $false
        Description = "2 slots with 64k context each (parallel processing)"
    }
    "parallel-2x48k" = @{
        ContextSize = 49152
        GpuLayers = 999
        ParallelSlots = 2
        EnableMTP = $false
        Description = "2 slots with 48k context each (parallel + headroom)"
    }
}

# Apply preset if specified
if ($presets.ContainsKey($Preset)) {
    $presetConfig = $presets[$Preset]
    Write-Host "Applying preset: $Preset - $($presetConfig.Description)" -ForegroundColor Green
    
    # Apply preset values (override parameters since user explicitly chose the preset)
    $ContextSize = $presetConfig.ContextSize
    $ParallelSlots = $presetConfig.ParallelSlots
    $GpuLayers = $presetConfig.GpuLayers
    $EnableMTP = $presetConfig.EnableMTP
} else {
    Write-Host "Warning: Preset '$Preset' not found. Available presets: $($presets.Keys -join ', ')" -ForegroundColor Yellow
    # Use custom values or defaults
    if ($null -eq $ContextSize) { $ContextSize = 131072 }
    if ($null -eq $ParallelSlots) { $ParallelSlots = 1 }
}

# Set defaults for any remaining null values
if (-not $ServerHost) { $ServerHost = "127.0.0.1" }
if (-not $Port) { $Port = 8888 }
if ($null -eq $GpuLayers) { $GpuLayers = 999 }
if (-not $CacheTypeK) { $CacheTypeK = "q8_0" }
if (-not $CacheTypeV) { $CacheTypeV = "q8_0" }

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Context per slot: $ContextSize tokens"
Write-Host "  Parallel slots: $ParallelSlots"
Write-Host "  Total context: $($ContextSize * $ParallelSlots) tokens"
Write-Host "  MTP enabled: $EnableMTP"
Write-Host ""

# Find model file
if (-not $ModelPath) {
    $ModelPath = "D:\_MODELS\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf"
}

if (-not (Test-Path $ModelPath)) {
    Write-Error "Model file not found: $ModelPath"
    Write-Host "Run .\download-model.ps1 first" -ForegroundColor Yellow
    exit 1
}

# Find the BeeLlama executable. The external checkout has its CUDA build in
# build-nram-cuda\bin; keep the repository build layouts as compatibility fallbacks.
if (-not $BeellamaPath) {
    $BeellamaPath = Join-Path $PSScriptRoot "..\..\..\beellama"
}

$buildBins = @(
    ("d:\_SATIN_AI_2\beellama\build-x64-windows-cuda-vulkan-release\bin"),
    (Join-Path $BeellamaPath "build\bin")
)
$serverCandidates = @()
foreach ($buildBin in $buildBins) {
    $serverCandidates += Join-Path $buildBin "llama-server.exe"
    $serverCandidates += Join-Path $buildBin "Release\llama-server.exe"
}
$serverExe = $serverCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $serverExe) {
    Write-Error "BeeLlama llama-server.exe was not found under $BeellamaPath"
    Write-Host "Build BeeLlama with its CUDA build script first: $BeellamaPath\scripts\build-win-cuda-13.1-sm_86.ps1" -ForegroundColor Yellow
    exit 1
}

$runtimeBin = Split-Path -Parent $serverExe
$cudaBackendCandidates = @((Join-Path $runtimeBin "ggml-cuda.dll"))
foreach ($buildBin in $buildBins) {
    $cudaBackendCandidates += Join-Path $buildBin "ggml-cuda.dll"
}
$cudaBackend = $cudaBackendCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $cudaBackend -and -not $AllowCpuFallback) {
    Write-Error "CUDA backend ggml-cuda.dll was not found beside the selected BeeLlama llama-server.exe. Rebuild BeeLlama with its CUDA build script or pass -AllowCpuFallback explicitly."
    exit 1
}

if ($cudaBackend) {
    $cudaBackendBin = Split-Path -Parent $cudaBackend
    $env:Path = "$cudaBackendBin;$env:Path"
    $cudaRoots = @(
        $env:CUDA_PATH,
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    ) | Where-Object { $_ -and (Test-Path (Join-Path $_ "bin")) }
    foreach ($cudaRoot in $cudaRoots) {
        $env:Path = "$(Join-Path $cudaRoot 'bin');$env:Path"
    }
    Write-Host "CUDA backend: $cudaBackend" -ForegroundColor Green
} elseif ($AllowCpuFallback) {
    Write-Warning "CUDA backend was not found; CPU fallback was explicitly allowed."
}

Write-Host "Server executable: $serverExe"
Write-Host "Model: $ModelPath"
Write-Host "Host: $ServerHost`:$Port"
Write-Host "Context size: $ContextSize tokens"
Write-Host "GPU layers: $GpuLayers"
Write-Host "KV cache: $CacheTypeK/$CacheTypeV"

# Build server arguments
$serverArgs = @(
    "--model", "`"$ModelPath`"",
    "--host", $ServerHost,
    "--port", $Port,
    "--ctx-size", $ContextSize,
    "--n-gpu-layers", $GpuLayers,
    "--parallel", $ParallelSlots,
    "--flash-attn", "on",
    "--cache-type-k", $CacheTypeK,
    "--cache-type-v", $CacheTypeV,
    "--n-predict", $MaxOutputTokens,
    # Speed optimization
    "--batch-size", 2048,           # Larger batch size for faster prompt processing
    "--ubatch-size", 2048,          # Micro-batch size for better throughput
    "--threads", 24,                # CPU threads for CPU operations
    "--cont-batching",              # Continuous batching for better throughput
    # Additional optimizations
    "--reasoning-preserve",         # Keep reasoning tokens (chain-of-thought)
    "--metrics",
    "--slots"
)

# Add MTP (Multi-Token Prediction) if enabled
if ($EnableMTP) {
    $serverArgs += "--spec-type"
    $serverArgs += "draft-mtp"
    $serverArgs += "--spec-draft-n-max"
    $serverArgs += "2"              # Number of tokens to draft (model has 1 nextn_predict_layer)
    Write-Host "MTP enabled with draft-mtp and 2-token drafting" -ForegroundColor Green
} else {
    # Even without an MTP/draft decoder we still set the draft depth ceiling so
    # the spec-decode scheduler is configured; with no --spec-type the decoder
    # stays idle but --spec-draft-n-max is honored if a draft is later attached.
    $serverArgs += "--spec-draft-n-max"
    $serverArgs += "4"
    Write-Host "Speculative decoder disabled; --spec-draft-n-max 4 set" -ForegroundColor Green
}

# === Prefix / prompt caching (BeeLlama b4688+) ===
# --cache-prompt is default-enabled; assert it explicitly for clarity.
$serverArgs += "--cache-prompt"
# Persist warmed slot KV to disk so a server restart restores the stable
# system/persona/tools prefix with zero prefill.
$slotSaveDir = Join-Path $PSScriptRoot "slot-cache"
if (-not (Test-Path $slotSaveDir)) { New-Item -ItemType Directory -Path $slotSaveDir | Out-Null }
$serverArgs += "--slot-save-path"
$serverArgs += "`"$slotSaveDir`""
# Save idle slot KV into the prompt cache on task switch.
$serverArgs += "--cache-idle-slots"
Write-Host "Prompt caching enabled; slot-save-path: $slotSaveDir" -ForegroundColor Green

Write-Host "`nStarting server with arguments:" -ForegroundColor Cyan
Write-Host ($serverArgs -join " ")

# Create log directory if it doesn't exist
$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logTimestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdoutLogFile = "$logDir/llama-server-$logTimestamp.stdout.log"
$stderrLogFile = "$logDir/llama-server-$logTimestamp.stderr.log"

# Start server
$process = Start-Process -FilePath $serverExe -WorkingDirectory $runtimeBin -ArgumentList $serverArgs -PassThru -NoNewWindow -RedirectStandardOutput $stdoutLogFile -RedirectStandardError $stderrLogFile

Write-Host "`nServer PID: $($process.Id)" -ForegroundColor Green
Write-Host "Waiting for server to start..." -ForegroundColor Yellow

# Wait for server to be ready
$maxWait = 60
$waited = 0
$ready = $false
$startupFailure = $null

while ($waited -lt $maxWait -and -not $ready) {
    Start-Sleep -Seconds 1
    $waited++
    
    try {
        $response = Invoke-RestMethod -Uri "http://$ServerHost`:$Port/health" -Method GET -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($response.status -eq "ok") {
            $ready = $true
        }
    } catch {
        # Server not ready yet
    }

    if (Test-Path $stderrLogFile) {
        $stderr = Get-Content -LiteralPath $stderrLogFile -Raw -ErrorAction SilentlyContinue
        if ($stderr -match "no usable GPU found|compiled without GPU support") {
            $startupFailure = "BeeLlama could not initialize a usable CUDA GPU backend. See $stderrLogFile."
            break
        }
    }

    if ($process.HasExited) {
        $startupFailure = "llama-server exited with code $($process.ExitCode). See $stderrLogFile."
        break
    }
    
    Write-Host "." -NoNewline
}

Write-Host ""

if ($ready) {
    Write-Host "Server is ready!" -ForegroundColor Green
    Write-Host "OpenAI-compatible endpoint: http://$ServerHost`:$Port/v1/chat/completions" -ForegroundColor Cyan
    Write-Host "Health endpoint: http://$ServerHost`:$Port/health" -ForegroundColor Cyan
    Write-Host "Metrics endpoint: http://$ServerHost`:$Port/metrics" -ForegroundColor Cyan
    Write-Host "`nPress Ctrl+C to stop the server" -ForegroundColor Yellow
    
    try {
        Wait-Process -Id $process.Id
    } catch {
        Write-Host "`nStopping server..." -ForegroundColor Yellow
        Stop-Process -Id $process.Id -Force
    }
} else {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
    if ($startupFailure) {
        Write-Error $startupFailure
    } else {
        Write-Error "Server failed to start within $maxWait seconds. See $stderrLogFile."
    }
    exit 1
}
