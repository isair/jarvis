<#
.SYNOPSIS
    Shared helpers for the local service start/stop scripts in scripts/services.

.DESCRIPTION
    Dot-sourced by every Start-*/Stop-* script in scripts/services. Provides the
    canonical primitives used across the local service stack:

      - endpoint health probing (HTTP GET with short timeout)
      - TCP listener detection
      - process discovery by command-line pattern
      - graceful process stop (port owner, then pattern match)
      - detached child PowerShell launch (service survives the caller)

    These helpers intentionally mirror the semantics of
    scripts/start-local-models.ps1 so the unified launcher and the per-service
    launchers agree on what "running" means.
#>

function ConvertTo-CommandLineArgument {
    param([Parameter(Mandatory)][string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Test-EndpointReady {
    param([Parameter(Mandatory)][string]$Uri, [int]$TimeoutSec = 2)
    try {
        Invoke-RestMethod -Uri $Uri -Method Get -TimeoutSec $TimeoutSec -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Get-EndpointPayload {
    param([Parameter(Mandatory)][string]$Uri, [int]$TimeoutSec = 2)
    try {
        return Invoke-RestMethod -Uri $Uri -Method Get -TimeoutSec $TimeoutSec -ErrorAction Stop
    } catch {
        return $null
    }
}

function Test-LocalPortInUse {
    param([Parameter(Mandatory)][int]$LocalPort)
    if ($LocalPort -le 0) { return $false }
    try {
        return @(
            Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue
        ).Count -gt 0
    } catch {
        return $false
    }
}

function Get-ProcessesByCommandLine {
    param(
        [Parameter(Mandatory)][string[]]$Patterns,
        [string]$NameFilter = $null
    )
    $query = if ($NameFilter) { "Name = '$NameFilter'" } else { $null }
    $processes = if ($query) {
        Get-CimInstance Win32_Process -Filter $query -ErrorAction SilentlyContinue
    } else {
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue
    }
    return @($processes | Where-Object {
        $cmd = $_.CommandLine
        if ([string]::IsNullOrEmpty($cmd)) { return $false }
        foreach ($p in $Patterns) {
            if ($cmd -match $p) { return $true }
        }
        return $false
    })
}

function Stop-ServiceProcesses {
    <#
    .SYNOPSIS
        Stops processes matched by command-line patterns, waiting for exit.
    #>
    param(
        [Parameter(Mandatory)][string[]]$Patterns,
        [string]$NameFilter = $null,
        [int]$WaitSeconds = 10,
        [string]$DisplayName = "service"
    )
    $targets = Get-ProcessesByCommandLine -Patterns $Patterns -NameFilter $NameFilter
    if ($targets.Count -eq 0) { return $true }
    foreach ($p in $targets) {
        $preview = $p.CommandLine
        if ($preview.Length -gt 80) { $preview = $preview.Substring(0, 80) }
        Write-Host "  Stopping $DisplayName PID $($p.ProcessId): $preview..." -ForegroundColor Yellow
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    $deadline = [datetime]::UtcNow.AddSeconds($WaitSeconds)
    while ([datetime]::UtcNow -lt $deadline) {
        if ((Get-ProcessesByCommandLine -Patterns $Patterns -NameFilter $NameFilter).Count -eq 0) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    return (Get-ProcessesByCommandLine -Patterns $Patterns -NameFilter $NameFilter).Count -eq 0
}

function Stop-PortOwner {
    <#
    .SYNOPSIS
        Stops the process owning a local TCP listener, if any.
    #>
    param([Parameter(Mandatory)][int]$LocalPort, [int]$WaitSeconds = 10)
    $connections = @(Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue)
    $pids = @($connections | Select-Object -ExpandProperty OwningProcess -Unique)
    foreach ($pidValue in $pids) {
        if ($pidValue -gt 0) {
            Write-Host "  Stopping port $LocalPort owner PID $pidValue..." -ForegroundColor Yellow
            Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
        }
    }
    if ($pids.Count -eq 0) { return $true }
    $deadline = [datetime]::UtcNow.AddSeconds($WaitSeconds)
    while ([datetime]::UtcNow -lt $deadline) {
        if (-not (Test-LocalPortInUse $LocalPort)) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return -not (Test-LocalPortInUse $LocalPort)
}

function Start-DetachedService {
    <#
    .SYNOPSIS
        Launches a service launcher script in a detached PowerShell window that
        survives the calling process (VS Code, extension host, terminal).
    #>
    param(
        [Parameter(Mandatory)][string]$ScriptPath,
        [Parameter(Mandatory)][hashtable]$Parameters,
        [string]$WorkingDirectory,
        [switch]$UseCurrentUserProfile
    )
    if (-not (Test-Path -LiteralPath $ScriptPath)) {
        throw "Service launcher not found: $ScriptPath"
    }
    $pwshCmd = Get-Command pwsh -ErrorAction SilentlyContinue
    if ($pwshCmd) { $pwsh = $pwshCmd.Source } else { $pwsh = (Get-Command powershell -ErrorAction Stop).Source }

    $arguments = [System.Collections.Generic.List[string]]::new()
    $arguments.Add("-NoLogo")
    if (-not $UseCurrentUserProfile) { $arguments.Add("-NoProfile") }
    $arguments.Add("-NoExit")
    $arguments.Add("-File")
    $arguments.Add((ConvertTo-CommandLineArgument $ScriptPath))
    foreach ($key in $Parameters.Keys) {
        $value = $Parameters[$key]
        if ($null -eq $value) { continue }
        if ($value -is [switch] -or $value -is [bool]) {
            if ([bool]$value) { $arguments.Add("-$key") }
            continue
        }
        $text = "$value"
        if ($text.Length -eq 0 -or $text -eq "0") { continue }
        $arguments.Add("-$key")
        $arguments.Add((ConvertTo-CommandLineArgument $text))
    }

    $startArgs = @{
        FilePath     = $pwsh
        ArgumentList = $arguments
    }
    if ($WorkingDirectory) { $startArgs.WorkingDirectory = $WorkingDirectory }
    Start-Process @startArgs | Out-Null
}

function Resolve-ServicePackageRoot {
    <#
    .SYNOPSIS
        Finds the directory that contains a marker subdirectory, walking up from
        this script.

    .DESCRIPTION
        The launchers used to assume "two levels above scripts/services is the
        repository root". That holds in a checkout and in the current VSIX
        staging layout, but it is a silent assumption: if the packaging layout
        ever changes, the launcher does not fail, it points at the wrong tree
        and reports a missing file the user cannot place.

        Walking up to a marker states the requirement instead of encoding a
        depth, so the same script works from a checkout, from the packaged
        extension, and from a copy placed anywhere else that keeps the
        marker's relative layout.
    #>
    param(
        [string]$From = (Split-Path -Parent $PSCommandPath),
        [string]$Marker = "services\qwen3-vl-xpu"
    )
    $probe = $From
    for ($i = 0; $i -lt 8 -and $probe; $i++) {
        if (Test-Path -LiteralPath (Join-Path $probe $Marker)) {
            return (Resolve-Path -LiteralPath $probe).Path
        }
        $parent = Split-Path -Parent $probe
        if ($parent -eq $probe) { break }
        $probe = $parent
    }
    throw "Could not locate '$Marker' from '$From'. The service scripts must keep their relative layout, or VISION_SERVICE_ROOT must be set."
}

function Start-VisionLane {
    <#
    .SYNOPSIS
        Starts a two-process vision lane (backend, then gateway) via the
        lifecycle scripts, waiting for the backend before the gateway.

    .DESCRIPTION
        The gateway proxies to the backend, so starting them simultaneously
        produces a gateway that answers 502 until the model finishes loading.
        The Integrations panel probes readiness immediately after Start, so
        that race shows up as a row that goes green and then red.
    #>
    param(
        [Parameter(Mandatory)][string]$LaneName,
        [Parameter(Mandatory)][string]$BackendScript,
        [Parameter(Mandatory)][string]$GatewayScript,
        [Parameter(Mandatory)][int]$BackendPort,
        [Parameter(Mandatory)][int]$GatewayPort,
        [int]$BackendTimeoutSec = 180
    )
    $lifecycle = Join-Path (Resolve-ServicePackageRoot) "services\qwen3-vl-xpu\scripts\lifecycle"
    foreach ($name in @($BackendScript, $GatewayScript)) {
        $path = Join-Path $lifecycle $name
        if (-not (Test-Path -LiteralPath $path)) { throw "Lifecycle script not found: $path" }
    }

    $pwshCmd = Get-Command pwsh -ErrorAction SilentlyContinue
    $pwsh = if ($pwshCmd) { $pwshCmd.Source } else { (Get-Command powershell -ErrorAction Stop).Source }

    Write-Host "Starting $LaneName backend (port $BackendPort)..." -ForegroundColor Cyan
    & $pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File (Join-Path $lifecycle $BackendScript)

    $backendHealth = "http://127.0.0.1:$BackendPort/readyz"
    $deadline = [datetime]::UtcNow.AddSeconds($BackendTimeoutSec)
    while ([datetime]::UtcNow -lt $deadline) {
        if (Test-EndpointReady $backendHealth 3) { break }
        Start-Sleep -Seconds 2
    }
    if (-not (Test-EndpointReady $backendHealth 3)) {
        Write-Warning "$LaneName backend did not report ready within $BackendTimeoutSec s; starting the gateway anyway."
    } else {
        Write-Host "  backend ready at $backendHealth" -ForegroundColor Green
    }

    Write-Host "Starting $LaneName gateway (port $GatewayPort)..." -ForegroundColor Cyan
    & $pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File (Join-Path $lifecycle $GatewayScript)
    Write-Host "$LaneName start requested; readiness endpoint: http://127.0.0.1:$GatewayPort/readyz" -ForegroundColor Yellow
}

function Stop-VisionLane {
    <#
    .SYNOPSIS
        Stops a vision lane's gateway then backend via the lifecycle scripts.

    .DESCRIPTION
        Gateway first, so in-flight requests fail fast at the edge instead of
        hitting a backend that is being torn down underneath them.

        These stops are PID- and port-scoped by the lifecycle module. The
        predecessors of this launcher pattern-killed every process whose command
        line matched `qwen3_vl_gateway.app:app`, which meant pressing Stop on one
        lane also killed the other lane's gateway.
    #>
    param(
        [Parameter(Mandatory)][string]$LaneName,
        [Parameter(Mandatory)][string]$GatewayScript,
        [Parameter(Mandatory)][string]$BackendScript
    )
    $lifecycle = Join-Path (Resolve-ServicePackageRoot) "services\qwen3-vl-xpu\scripts\lifecycle"
    $pwshCmd = Get-Command pwsh -ErrorAction SilentlyContinue
    $pwsh = if ($pwshCmd) { $pwshCmd.Source } else { (Get-Command powershell -ErrorAction Stop).Source }

    foreach ($name in @($GatewayScript, $BackendScript)) {
        $path = Join-Path $lifecycle $name
        if (-not (Test-Path -LiteralPath $path)) {
            Write-Warning "Lifecycle script not found, skipping: $path"
            continue
        }
        & $pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File $path
    }
    Write-Host "$LaneName stopped." -ForegroundColor Green
}

function Assert-ServiceNotRunning {
    <#
    .SYNOPSIS
        Returns $false when healthy (reuse), throws when the port is occupied
        by an unhealthy owner, returns $true when a start is required.
    #>
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$HealthUrl,
        [Parameter(Mandatory)][int]$LocalPort
    )
    if (Test-EndpointReady $HealthUrl) {
        Write-Host "$Name is already running at $HealthUrl; reusing it." -ForegroundColor Green
        return $false
    }
    if (Test-LocalPortInUse $LocalPort) {
        throw "$Name is not healthy at $HealthUrl, but local port $LocalPort is already occupied. Refusing to start a second instance."
    }
    return $true
}
