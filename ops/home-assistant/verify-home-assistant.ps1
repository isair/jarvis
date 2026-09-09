#Requires -Version 5.1
<#
.SYNOPSIS
    Verify the Talkie Toaster Home Assistant deployment (no writes, no changes).
#>
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$VM_NAME = 'TalkieToaster-HA'

function Get-VBoxManagePath {
    $cmd = Get-Command 'VBoxManage.exe' -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @("$env:ProgramFiles\Oracle\VirtualBox\VBoxManage.exe",
                     'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe')) {
        if (Test-Path -LiteralPath $p) { return $p }
    }
    return $null
}
function Test-HaEndpoint([string]$url) {
    try { return [int](Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5).StatusCode }
    catch { return $null }
}

$statePath = Join-Path $PSScriptRoot 'local-state.json'
if (Test-Path -LiteralPath $statePath) {
    Write-Host '--- local-state.json ---' -ForegroundColor Cyan
    Get-Content -LiteralPath $statePath -Raw
}

$vm = Get-VBoxManagePath
if (-not $vm) { Write-Host 'FAIL: VirtualBox (VBoxManage.exe) not found.' -ForegroundColor Red; exit 1 }

Write-Host '--- VirtualBox ---' -ForegroundColor Cyan
$vmList = & $vm list vms
$vminfoLine = $vmList | Where-Object { $_ -match "`"$([regex]::Escape($VM_NAME))`"" }
if (-not $vminfoLine) { Write-Host "FAIL: VM '$VM_NAME' is not registered." -ForegroundColor Red; exit 1 }
Write-Host "  $vminfoLine"
$state = (& $vm showvminfo $VM_NAME --machinereadable | Where-Object { $_ -match '^VMState="(.+)"$' })
if ($state -match 'VMState="(.+)"') { Write-Host "  VMState: $($Matches[1])" }
$ip = $null
$gp = & $vm guestproperty get $VM_NAME '/VirtualBox/GuestInfo/Net/0/V4/IP'
if ($gp -match 'Value:\s*([0-9.]+)') { $ip = $Matches[1]; Write-Host "  Guest IPv4: $ip" }

Write-Host '--- Home Assistant endpoint ---' -ForegroundColor Cyan
$found = $null
$urls = @('http://homeassistant.local:8123', 'http://homeassistant:8123')
if ($ip) { $urls += "http://${ip}:8123" }
foreach ($u in $urls) {
    $code = Test-HaEndpoint $u
    if ($null -ne $code) { Write-Host "  OK  $u -> $code" -ForegroundColor Green; $found = $u; break }
    else { Write-Host "  ..  $u -> no answer" }
}
if ($found) {
    Write-Host "`nPASS: Home Assistant reachable at $found" -ForegroundColor Green
} else {
    Write-Host "`nINCONCLUSIVE: no 8123 endpoint yet. If the VM just started, boot can take 1-2 min; re-run." -ForegroundColor Yellow
    exit 0
}
