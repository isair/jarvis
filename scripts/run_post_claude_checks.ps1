# Post-integration checks (stack smoke + fast tests). Run with dashboard + cafe-agent up.
param(
    [switch]$AllowOffline,
    [switch]$SkipEvals
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "  ERROR  .venv not found. Create venv first."
    exit 1
}

Write-Host "  Post-Claude integration checks"
Write-Host ""

& "$Root\scripts\smoke_jarvis_stack.ps1" @PSBoundParameters
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "  pytest (unit smoke modules)"
& $py -m pytest -q -m unit `
    tests/test_daemon_lock.py `
    tests/test_settings_api.py `
    tests/test_dashboard_query.py `
    tests/test_cafe_agent_proxy.py `
    tests/test_shell_daemon.py `
    tests/test_cafe_jarvis_bridge.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipEvals) {
    Write-Host ""
    Write-Host "  pytest (cafe live evals)"
    & $py -m pytest evals/test_cafe_schedule_claude.py -v
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "  cargo test (cafe-agent)"
Push-Location (Join-Path $Root "cafe-agent")
cargo test --workspace -q
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { exit $code }

Write-Host ""
Write-Host "  All post-Claude checks passed."
