<#
.SYNOPSIS
    Detach the LadybugDB native module from the BLACKGLASS graph plane.

.DESCRIPTION
    There is no daemon to kill: LadybugDB is embedded in the extension host
    process, so the only way to "stop" it is to stop the extension host from
    loading it. This script sets that up by writing the environment override the
    backend resolver honours.

    It deliberately does NOT uninstall the module and does NOT delete the
    databases. Both would be destructive in a way the word "Stop" does not
    promise -- deleting workspace.db discards every verified finding, human
    decision and reachability projection recorded for this repository, and
    §36.6 makes deletion a scoped, explicit operation, not a side effect of
    turning a service off.

    Use -PurgeDatabases only when you actually mean "forget all BLACKGLASS
    evidence on this machine". It asks first.

.EXAMPLE
    .\scripts\services\Stop-LadybugDb.ps1
    .\scripts\services\Stop-LadybugDb.ps1 -PurgeDatabases
#>
[CmdletBinding()]
param(
    [switch]$PurgeDatabases
)

$ErrorActionPreference = "Stop"

# The resolver reads this at activation. Setting it at User scope survives the
# reload that is required for the change to take effect at all.
[Environment]::SetEnvironmentVariable("ZOO_BLACKGLASS_DISABLE_LADYBUG", "1", "User")
$env:ZOO_BLACKGLASS_DISABLE_LADYBUG = "1"

Write-Host "LadybugDB disabled for BLACKGLASS (ZOO_BLACKGLASS_DISABLE_LADYBUG=1, user scope)." -ForegroundColor Green
Write-Host "Reload the VS Code window to take effect." -ForegroundColor DarkGray
Write-Host "The security graph falls back to memory and will not survive the reload after that." -ForegroundColor Yellow
Write-Host "Re-enable with: .\scripts\services\Start-LadybugDb.ps1" -ForegroundColor DarkGray

if (-not $PurgeDatabases) {
    Write-Host ""
    Write-Host "Databases were kept. Pass -PurgeDatabases to delete them." -ForegroundColor DarkGray
    return
}

# Global storage lives under the VS Code profile; both stable and Insiders are
# checked because a developer machine commonly has both.
$candidates = @(
    (Join-Path $env:APPDATA "Code\User\globalStorage"),
    (Join-Path $env:APPDATA "Code - Insiders\User\globalStorage")
) | Where-Object { Test-Path -LiteralPath $_ }

$graphDirs = @()
foreach ($root in $candidates) {
    $graphDirs += @(Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName "blackglass\graph" } |
        Where-Object { Test-Path -LiteralPath $_ })
}

if ($graphDirs.Count -eq 0) {
    Write-Host "No BLACKGLASS graph databases found; nothing to purge." -ForegroundColor DarkGray
    return
}

Write-Host ""
Write-Host "About to permanently delete BLACKGLASS evidence:" -ForegroundColor Red
$graphDirs | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
$answer = Read-Host "Type PURGE to confirm"
if ($answer -cne "PURGE") {
    Write-Host "Aborted. Databases were not touched." -ForegroundColor Yellow
    return
}

foreach ($dir in $graphDirs) {
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction Continue
    Write-Host "  Deleted $dir" -ForegroundColor Yellow
}
Write-Host "BLACKGLASS graph databases purged." -ForegroundColor Green
