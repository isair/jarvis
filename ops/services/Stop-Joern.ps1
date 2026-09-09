<#
.SYNOPSIS
    Disable the BLACKGLASS reachability plane (Joern).

.DESCRIPTION
    There is no Joern daemon to stop. JoernCodeGraphAdapter runs the binary per
    query with a timeout and it exits on its own; between queries there is nothing
    listening and nothing resident. So "stopping" Joern means telling the fabric
    not to use it: ZOO_BLACKGLASS_DISABLE_JOERN=1, read by resolveBlackglassBackends
    at activation.

    The consequence is deliberate and reported: with the plane off, every
    reachability question answers UNKNOWN — never "not reachable" (§35) — and no
    attack chain can be promoted to STATIC_CONFIRMED. Turning this off makes the
    system say less, not say no.

    A stray JVM sweep is still available via -KillStrays, for the case where a
    query was cancelled and its child outlived the timeout. It is not the default,
    because a pattern-matched kill of `java` processes on a developer machine is a
    blunt instrument and the normal case has nothing to kill.

    Generated CPGs are NOT deleted. They are content-addressed by repository
    snapshot and expensive to rebuild; §36.6 makes deletion a scoped, explicit
    operation, not a side effect of turning a plane off. Use -PurgeCpgs for that.

.EXAMPLE
    .\scripts\services\Stop-Joern.ps1
    .\scripts\services\Stop-Joern.ps1 -KillStrays
    .\scripts\services\Stop-Joern.ps1 -PurgeCpgs
#>
[CmdletBinding()]
param(
    [switch]$KillStrays,
    [switch]$PurgeCpgs,
    [int]$WaitSeconds = 20
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $PSCommandPath
. (Join-Path $scriptRoot "lib\LocalService.Common.ps1")

Write-Host "Disabling the BLACKGLASS reachability plane (Joern)..." -ForegroundColor Cyan

[Environment]::SetEnvironmentVariable("ZOO_BLACKGLASS_DISABLE_JOERN", "1", "User")
$env:ZOO_BLACKGLASS_DISABLE_JOERN = "1"
Write-Host "Set ZOO_BLACKGLASS_DISABLE_JOERN=1 (User scope)." -ForegroundColor Green

if ($KillStrays) {
    # The JVM, not the .bat shim: a Joern run appears as `java ... io.joern...`,
    # and the shim has usually already exited by the time anyone looks.
    $patterns = @(
        'io\.joern',
        'joern[\\/]+(lib|bin)'
    )
    $stopped = Stop-ServiceProcesses -Patterns $patterns -WaitSeconds $WaitSeconds -DisplayName "Joern"
    if ($stopped) {
        Write-Host "Stray Joern JVMs cleared." -ForegroundColor Green
    } else {
        Write-Host "No stray Joern JVMs were running." -ForegroundColor DarkGray
    }
}

$workDir = Join-Path $env:LOCALAPPDATA "Kelvin-Clyne\blackglass\joern"

if ($PurgeCpgs) {
    if (Test-Path -LiteralPath $workDir) {
        Write-Host ""
        Write-Host "About to delete the Joern CPG cache:" -ForegroundColor Red
        Write-Host "  $workDir" -ForegroundColor Red
        Write-Host "Rebuilding a CPG for a large repository takes minutes to hours." -ForegroundColor Red
        $answer = Read-Host "Type PURGE to confirm"
        if ($answer -cne "PURGE") {
            Write-Host "Purge cancelled; CPG cache kept." -ForegroundColor Yellow
        } else {
            Remove-Item -LiteralPath $workDir -Recurse -Force
            Write-Host "CPG cache deleted." -ForegroundColor Green
        }
    } else {
        Write-Host "No CPG cache to purge at $workDir." -ForegroundColor DarkGray
    }
} else {
    Write-Host "Cached CPGs kept under $workDir." -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Reload the VS Code window to apply." -ForegroundColor Yellow
Write-Host "BLACKGLASS reachability will then report UNKNOWN (never 'not reachable')," -ForegroundColor DarkGray
Write-Host "and no attack chain can reach STATIC_CONFIRMED." -ForegroundColor DarkGray
