[CmdletBinding()]
param([string]$HuggingFaceCli = "huggingface-cli")

$ErrorActionPreference = "Stop"
$embeddingRoot = Join-Path $PSScriptRoot "models\Qwen3-Embedding-0.6B-int4-cw-ov\1"
$rerankerRoot = Join-Path $PSScriptRoot "models\Qwen3-Reranker-0.6B-int8-ov\1"

if (-not (Get-Command $HuggingFaceCli -ErrorAction SilentlyContinue) -and -not (Test-Path -LiteralPath $HuggingFaceCli)) {
    throw "huggingface-cli was not found. Install huggingface_hub or pass -HuggingFaceCli with its path."
}

New-Item -ItemType Directory -Force -Path $embeddingRoot, $rerankerRoot | Out-Null
& $HuggingFaceCli download OpenVINO/Qwen3-Embedding-0.6B-int4-cw-ov --local-dir $embeddingRoot
if ($LASTEXITCODE -ne 0) { throw "Failed to download the Qwen3 embedding OpenVINO IR." }
& $HuggingFaceCli download OpenVINO/Qwen3-Reranker-0.6B-int8-ov --local-dir $rerankerRoot
if ($LASTEXITCODE -ne 0) { throw "Failed to download the Qwen3 reranker OpenVINO IR." }
Write-Host "Qwen3 OpenVINO IR artifacts downloaded. Optimum Intel is not required by OVMS at runtime; it is only needed for the Python inference example." -ForegroundColor Green
