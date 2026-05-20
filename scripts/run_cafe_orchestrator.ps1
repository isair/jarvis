# Start cafe-agent orchestrator (Rust)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$env:Path = "$env:USERPROFILE\.cargo\bin;" + $env:Path

$vcvars = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (Test-Path $vcvars) {
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path "env:$($matches[1])" -Value $matches[2]
        }
    }
}

$CafeDir = Join-Path $Root "cafe-agent"
Set-Location $CafeDir
if (-not (Test-Path "config.toml")) {
    Copy-Item "config.example.toml" "config.toml"
    Write-Host "  Created cafe-agent/config.toml from example"
}
if ($env:ANTHROPIC_BASE_URL) {
    Write-Host "  ANTHROPIC_BASE_URL=$($env:ANTHROPIC_BASE_URL)"
} else {
    Write-Host "  ANTHROPIC_BASE_URL not set (default https://api.anthropic.com)"
}
if ($env:ANTHROPIC_API_KEY) {
    Write-Host "  ANTHROPIC_API_KEY is set"
} else {
    Write-Host "  WARN  ANTHROPIC_API_KEY not set — schedule_plan uses heuristic only"
}
cargo run -p orchestrator
