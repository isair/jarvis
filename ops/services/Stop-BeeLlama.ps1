<#
.SYNOPSIS
    Stop the BeeLlama local LLM server (llama.cpp).

.DESCRIPTION
    Stops the process owning the BeeLlama listener port, then any remaining
    llama-server processes launched by the BeeLlama stack. Safe to run when the
    service is not running.

.EXAMPLE
    .\scripts\services\Stop-BeeLlama.ps1
    .\scripts\services\Stop-BeeLlama.ps1 -Port 8888
#>
[CmdletBinding()]
param(
    [int]$Port = 8888
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $PSCommandPath
. (Join-Path $scriptRoot "lib\LocalService.Common.ps1")

Write-Host "Stopping BeeLlama (port $Port)..." -ForegroundColor Cyan
$patterns = @('llama-server', 'start-server\.ps1')
$stopped = Stop-ServiceProcesses -Patterns $patterns -DisplayName "BeeLlama"
$portReleased = Stop-PortOwner -LocalPort $Port
if ($stopped -and $portReleased) {
    Write-Host "BeeLlama stopped." -ForegroundColor Green
} else {
    throw "BeeLlama did not fully stop; manual intervention required."
}
