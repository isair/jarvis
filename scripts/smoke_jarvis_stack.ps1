param(
    [switch]$AllowOffline
)

# Quick health check for Jarvis stack (dashboard + optional cafe-agent).
$ErrorActionPreference = "Stop"
$Root = if ($env:JARVIS_ROOT) { $env:JARVIS_ROOT } else { Split-Path -Parent $PSScriptRoot }

function Test-PortOpen([int]$Port) {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $Port)
        $c.Close()
        return $true
    } catch {
        return $false
    }
}

function Get-Json([string]$Url) {
    return Invoke-RestMethod -Uri $Url -TimeoutSec 15
}

function Post-Json([string]$Url, [object]$Body) {
    return Invoke-RestMethod -Uri $Url -Method Post -Body ($Body | ConvertTo-Json -Depth 8) `
        -ContentType "application/json" -TimeoutSec 60
}

$failures = 0
$strict = -not $AllowOffline
Write-Host "  Jarvis stack smoke check"
Write-Host "  Repo: $Root"

$dashPort = 5050
$cafePort = 8787

if (-not (Test-PortOpen $dashPort)) {
    Write-Host "  WARN  Dashboard not on :$dashPort (start shell or memory_viewer)"
    if ($strict) { $failures++ }
} else {
    $vc = Get-Json "http://127.0.0.1:$dashPort/api/dashboard/voice-config"
    Write-Host "  OK    Dashboard voice-config (PTT=$($vc.ptt_hotkey_display), lazy=$($vc.whisper_lazy_load))"

    $meta = Get-Json "http://127.0.0.1:$dashPort/api/settings/metadata"
    if ($meta.ok) {
        Write-Host "  OK    Settings metadata ($($meta.fields.Count) fields)"
    } else {
        Write-Host "  FAIL  Settings metadata"
        $failures++
    }

    $health = Get-Json "http://127.0.0.1:$dashPort/api/cafe-agent/health"
    Write-Host "  OK    Cafe proxy health status=$($health.status)"

    if (Test-PortOpen $cafePort) {
        try {
            $task = Post-Json "http://127.0.0.1:$dashPort/api/cafe-agent/task" @{
                task = @{
                    type         = "schedule_plan"
                    week_start   = "2026-05-12"
                    persist      = $false
                }
            }
            $planner = $task.result.data.planner
            $days = $task.result.data.days.Count
            if ($task.ok -and $planner -and $days -ge 7) {
                Write-Host "  OK    schedule_plan via proxy (planner=$planner, days=$days)"
            } else {
                Write-Host "  FAIL  schedule_plan unexpected payload"
                $failures++
            }
        } catch {
            Write-Host "  FAIL  schedule_plan: $_"
            $failures++
        }
    } else {
        Write-Host "  SKIP  schedule_plan (cafe orchestrator offline on :$cafePort)"
    }

    try {
        $q = Post-Json "http://127.0.0.1:$dashPort/api/dashboard/query" @{ text = "smoke ping" }
        if ($q.ok -and $q.delivery) {
            Write-Host "  OK    dashboard/query delivery=$($q.delivery)"
        } else {
            Write-Host "  FAIL  dashboard/query"
            $failures++
        }
    } catch {
        Write-Host "  FAIL  dashboard/query: $_"
        $failures++
    }
}

if (-not (Test-PortOpen $cafePort)) {
    Write-Host "  WARN  Cafe orchestrator not on :$cafePort (shell: Start cafe agent)"
} else {
    $cafe = Get-Json "http://127.0.0.1:$cafePort/health"
    Write-Host "  OK    Cafe orchestrator sales_rows=$($cafe.sales_rows)"
}

$sample = Join-Path $Root "cafe-agent\data\sample_sales.csv"
if (Test-Path $sample) {
    Write-Host "  OK    Sample sales CSV present"
} else {
    Write-Host "  WARN  Missing $sample"
    $failures++
}

if ($failures -gt 0) {
    Write-Host "  Done with $failures failure(s)."
    exit 1
}
Write-Host "  Done (all checks passed)."
exit 0
