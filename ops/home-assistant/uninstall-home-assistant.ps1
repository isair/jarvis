#Requires -Version 5.1
<#
.SYNOPSIS
    Remove the TalkieToaster-HA VM (power off + unregister incl. disk files) and local state.
    Does not remove the Oracle VirtualBox package itself (shared component).
    Only touches the VM named TalkieToaster-HA and its own folder; other VMs/adapters untouched.
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

$vm = Get-VBoxManagePath
if (-not $vm) { Write-Host 'FAIL: VBoxManage.exe not found; nothing to uninstall.' -ForegroundColor Red; exit 1 }

$vmList = & $vm list vms
if ($vmList -notmatch [regex]::Escape($VM_NAME)) {
    Write-Host "No VM named $VM_NAME. Nothing to remove."
} else {
    $stateLine = (& $vm showvminfo $VM_NAME --machinereadable | Where-Object { $_ -match '^VMState=' })
    if ($stateLine -match 'VMState="running"') {
        Write-Host "Powering off $VM_NAME ..."
        & $vm controlvm $VM_NAME poweroff | Out-Null
    }
    Write-Host "Unregistering $VM_NAME (including its disk) ..."
    & $vm unregistervm $VM_NAME --delete | Out-Null
    Write-Host 'Removed.'
}

# Clean the machine data folder + local state.
foreach ($root in @('C:\', 'D:\')) {
    $base = Join-Path $root 'TalkieToaster-HA'
    if (Test-Path -LiteralPath $base) { Remove-Item -LiteralPath $base -Recurse -Force; Write-Host "Deleted $base" }
}
$statePath = Join-Path $PSScriptRoot 'local-state.json'
if (Test-Path -LiteralPath $statePath) { Remove-Item -LiteralPath $statePath -Force; Write-Host "Deleted $statePath" }
Write-Host 'Uninstall complete.'
