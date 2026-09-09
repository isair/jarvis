#Requires -Version 5.1
<#
.SYNOPSIS
    Idempotent deployment of Home Assistant OS in Oracle VirtualBox for Talkie Toaster (Jarvis).
.DESCRIPTION
    Phase 1: Windows preflight, VirtualBox install via winget (only when absent), existing-VM resume.
    Phase 2: Download latest stable HAOS (non-prerelease) VirtualBox VDI, verify checksum,
             clone into a permanent (non-temp) folder, create headless VM with bridged networking,
             wait up to 15 minutes for Home Assistant on port 8123.
    Non-secret state is written to ops/home-assistant/local-state.json. Secrets (HA long-lived
    token, Amazon 2SV key) are NEVER written here; they are stored by the Jarvis config system.
#>

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# ---------------------------------------------------------------- helpers
function Get-VBoxManagePath {
    $cmd = Get-Command 'VBoxManage.exe' -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @(
        "$env:ProgramFiles\Oracle\VirtualBox\VBoxManage.exe",
        'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe'
    )) {
        if (Test-Path -LiteralPath $p) { return $p }
    }
    $reg = 'HKLM:\SOFTWARE\Oracle\VirtualBox'
    if (Test-Path -LiteralPath $reg) {
        $loc = (Get-ItemProperty -LiteralPath $reg -ErrorAction SilentlyContinue).'(default)'
        if ($loc) {
            $p = Join-Path $loc 'VBoxManage.exe'
            if (Test-Path -LiteralPath $p) { return $p }
        }
    }
    return $null
}

function Write-State([hashtable]$merge) {
    $statePath = Join-Path $PSScriptRoot 'local-state.json'
    $state = @{}
    if (Test-Path -LiteralPath $statePath) {
        try {
            $loaded = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
            foreach ($prop in $loaded.PSObject.Properties) { $state[$prop.Name] = $prop.Value }
        } catch { }
    }
    foreach ($k in $merge.Keys) { $state[$k] = $merge[$k] }
    $state['last_run_utc'] = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    ($state | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath $statePath -Encoding UTF8
}

$VM_NAME = 'TalkieToaster-HA'
$HA_PORTS = 8123
$MAX_WAIT_SEC = 900

Write-Host '=== Talkie Toaster / Home Assistant deployment ===' -ForegroundColor Cyan

# ---------------------------------------------------------------- 1. Preflight
Write-Host "`n[Preflight] Windows / hardware checks" -ForegroundColor Cyan

$os = Get-CimInstance Win32_OperatingSystem
Write-Host "  Windows: $($os.Caption)"
if ($os.Caption -notmatch 'Windows 1[01]|Windows 8.1') {
    Write-Host "  NOTE: unexpected Windows edition '$($os.Caption)'; continuing if x64 + Hyper-V/VirtualBox capable."
}

$cs = Get-CimInstance Win32_ComputerSystem
$virt = $cs | Select-Object -ExpandProperty HyperVVisPresentationSupported -ErrorAction SilentlyContinue
if (-not $virt) { $virt = $cs.HyperVisorPresent }
Write-Host "  Hypervisor/virtualization active: $virt"
if (-not $virt) {
    Write-Host '  FAIL: hardware virtualization is not enabled/active.' -ForegroundColor Red
    Write-Host '  Checkpoint: enable Intel VT-x/AMD-V in BIOS, then re-run this script.'
    exit 1
}

$freeRamMB = [int][math]::Round($os.FreePhysicalMemory / 1024)
$totalRamMB = [int][math]::Round($os.TotalVisibleMemorySize / 1024)
Write-Host "  RAM: free ${freeRamMB} MB (total ${totalRamMB} MB, need >= 8192 MB free)"
if ($freeRamMB -lt 8192) { Write-Host "  FAIL: need >= 8192 MB free RAM." -ForegroundColor Red; exit 1 }

$vmDrive = (Split-Path -Qualifier ((Resolve-Path $PSScriptRoot).Path))
$dsk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$vmDrive'"
$freeDiskGB = [math]::Round($dsk.Size / 1GB, 1)
$freeDiskGBFree = [math]::Round($dsk.FreeSpace / 1GB, 1)
Write-Host "  Disk ${vmDrive}: ${freeDiskGBFree} GB free (need >= 40 GB)"
if ($dsk.FreeSpace -lt 40GB) { Write-Host "  FAIL: drive ${vmDrive} has < 40 GB free." -ForegroundColor Red; exit 1 }

# Machine data folder (persistent, outside temp): <drive root>\TalkieToaster-HA
$MACHINE_FOLDER = "$vmDrive\TalkieToaster-HA"
New-Item -ItemType Directory -Force -Path $MACHINE_FOLDER | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $MACHINE_FOLDER 'staging') | Out-Null
Write-Host "  Persistent data folder: $MACHINE_FOLDER (staging inside it)"

function Get-ActiveAdapter {
    # Prefer the physical Ethernet adapter that owns the default route;
    # fall back to Up Wi-Fi; never fall back to NAT.
    $defIf = (Get-NetRoute -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue |
        Sort-Object RouteMetric | Select-Object -First 1).ifIndex
    $up = @(Get-NetAdapter | Where-Object { $_.Status -eq 'Up' -and $_.InterfaceDescription })
    if ($defIf) {
        $a = $up | Where-Object { $_.ifIndex -eq $defIf } | Select-Object -First 1
        if ($a) { return $a }
    }
    $up | Sort-Object InterfaceMetric | Select-Object -First 1
}
$adapter = Get-ActiveAdapter
if (-not $adapter) { Write-Host '  FAIL: no active physical network adapter.' -ForegroundColor Red; exit 1 }
Write-Host "  Network: $($adapter.Name) ($($adapter.PhysicalMediaType), ifIndex $($adapter.ifIndex))"

# ---------------------------------------------------------------- 2. VirtualBox
Write-Host "`n[VirtualBox] presence check" -ForegroundColor Cyan
$vm = Get-VBoxManagePath
if (-not $vm) {
    Write-Host '  VirtualBox not found -> installing via winget.'
    winget install -e --id Oracle.VirtualBox --accept-package-agreements --accept-source-agreements
    $vm = Get-VBoxManagePath
    if (-not $vm) {
        # winget may require a shell refresh; try the known default once more.
        $p = 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe'
        if (Test-Path -LiteralPath $p) { $vm = $p }
    }
    if (-not $vm) {
        Write-Host '  FAIL: winget finished but VBoxManage.exe not found.' -ForegroundColor Red
        Write-Host '  Checkpoint: if a reboot was requested, reboot, then re-run this script (idempotent).'
        exit 1
    }
} else {
    Write-Host "  Found: $vm"
}
$vboxVersion = & $vm --version
Write-Host "  VirtualBox version: $vboxVersion"

# ---------------------------------------------------------------- 3. Existing VM resume
Write-Host "`n[VM] existing instance check" -ForegroundColor Cyan
& $vm list vms | ForEach-Object { Write-Host "  $_" }
$vms = @(& $vm list vms)
$existing = @($vms | ForEach-Object { if ($_ -match '"([^"]+)"') { $Matches[1] } } | Where-Object { $_ -eq $VM_NAME })
Write-Host "  TalkieToaster-HA present: $($existing.Count -gt 0)"

$state = @{}
if ($existing) {
    Write-Host "  Resuming existing VM '$VM_NAME'." -ForegroundColor Green
    $state['vm_name'] = $VM_NAME
    $uuidOut = & $vm list vms | Where-Object { $_ -match "`"$([regex]::Escape($VM_NAME))`"" }
    if ($uuidOut -match '\{([^}]+)\}') { $state['vm_uuid'] = $Matches[1] }
} else {
    # ------------------------------------------------------------ download HAOS
    Write-Host "`n[HAOS] resolving latest stable release" -ForegroundColor Cyan
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $rel = Invoke-RestMethod -Uri 'https://api.github.com/repos/home-assistant/operating-system/releases/latest' -UseBasicParsing
    if ($rel.prerelease) { Write-Host '  FAIL: latest release marked prerelease.' -ForegroundColor Red; exit 1 }
    $tag = $rel.tag_name
    Write-Host "  Version: $tag"
    $asset = $rel.assets | Where-Object { $_.name -eq "haos_ova-$tag.vdi.zip" } | Select-Object -First 1
    if (-not $asset) { Write-Host "  FAIL: haos_ova-$tag.vdi.zip not found in release $tag." -ForegroundColor Red; exit 1 }
    $assetUrl = $asset.browser_download_url
    $digest = ($asset.digest -replace '^sha256:', '')
    Write-Host "  Asset: $($asset.name)"
    Write-Host "  URL:   $assetUrl"

    $staging = Join-Path $MACHINE_FOLDER 'staging'
    $zipPath = Join-Path $staging $asset.name
    if (-not (Test-Path -LiteralPath $zipPath)) {
        Write-Host '  Downloading...'
        Invoke-WebRequest -Uri $assetUrl -OutFile $zipPath -UseBasicParsing
    } else {
        Write-Host '  Download already staged.'
    }
    if ($digest) {
        $actual = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $digest.ToLower()) {
            Write-Host "  FAIL: checksum mismatch.$([char]10)  expected: $($digest.ToLower())$([char]10)  actual:   $actual" -ForegroundColor Red
            exit 1
        }
        Write-Host '  Checksum verified.'
    } else {
        Write-Host '  NOTE: no digest published on asset; relying on TLS + GitHub release metadata.'
    }

    # Extract VDI from the zip.
    $extractDir = Join-Path $staging "extracted-$tag"
    if (-not (Test-Path (Join-Path $extractDir "haos_ova-$tag.vdi"))) {
        Expand-Archive -LiteralPath $zipPath -DestinationPath $extractDir -Force
    }

    # ------------------------------------------------------------ clone to permanent disk
    $vmFolder = Join-Path $MACHINE_FOLDER $VM_NAME
    New-Item -ItemType Directory -Force -Path $vmFolder | Out-Null
    $diskPath = Join-Path $vmFolder "TalkieToaster-HA.vdi"
    if (-not (Test-Path -LiteralPath $diskPath)) {
        Copy-Item -LiteralPath (Join-Path $extractDir "haos_ova-$tag.vdi") -Destination $diskPath -Force
        Write-Host "  Disk cloned: $diskPath"
    } else {
        Write-Host "  Disk already present: $diskPath"
    }

    # ------------------------------------------------------------ createVM
    Write-Host "`n[VM] creating '$VM_NAME'" -ForegroundColor Cyan
    & $vm createvm --name $VM_NAME --ostype Linux_64 --basefolder $MACHINE_FOLDER --register | ForEach-Object { Write-Host "  $_" }
    # Resolve the VirtualBox-visible Name for the active physical adapter (VBox 7.2 bridge-adapter1 uses the Name string).
    $bridgeLines = & $vm list bridgedifs
    $curName = $null; $bridgeName = $null
    foreach ($line in $bridgeLines) {
        if ($line -match '^\s*Name:\s*(.+)$') { $n = $Matches[1].Trim()
            $n1 = ($adapter.Name -replace '\s+$','')
            $n2 = ("$($adapter.InterfaceDescription)" -replace '\s+$','')
            if ($n -eq $n1 -or ($n2 -and $n -eq $n2)) { $bridgeName = $n }
        }
    }
    if (-not $bridgeName) {
        # second chance: match by GUID order
        $guids = @{}
        $pending = $null
        foreach ($line in $bridgeLines) {
            if ($line -match '^\s*Name:\s*(.+)$') { $pending = $Matches[1].Trim() }
            elseif ($line -match '^\s*GUID:\s*\{?([0-9A-Fa-f\-]+)\}?') { $guids[$Matches[1]] = $pending }
        }
        Write-Host "  FAIL: active adapter '$($adapter.Name)' not present in VBoxManage list bridgedifs." -ForegroundColor Red
        Write-Host '  No silent NAT fallback: enable the adapter (or its "Host-only" bridge list) and re-run.'
        exit 1
    }
    Write-Host "  Bridging via '$bridgeName' ($($adapter.Name))"
    # 2 vCPU, 4 GB, EFI, UTC clock (rtc-use-utc on VBox 7.2), bridged NIC on active adapter.
    & $vm modifyvm $VM_NAME --memory 4096 --cpus 2 --firmware efi --rtc-use-utc on
    & $vm modifyvm $VM_NAME --nic1 bridged --bridge-adapter1 $bridgeName
    # SATA/AHCI controller + cloned disk.
    & $vm storagectl $VM_NAME --name 'SATA' --add sata --controller IntelAhci --portcount 1 --bootable on
    & $vm storageattach $VM_NAME --storagectl 'SATA' --port 0 --device 0 --type hdd --medium $diskPath
    # Headless autostart with the VirtualBox host process (VBox 7.2: --autostart-enabled).
    & $vm modifyvm $VM_NAME --autostart-enabled on
    # Ensure dynamic disk is at least 32 GB virtual size.
    $info = & $vm showmediuminfo $diskPath
    $vsizeLine = $info | Where-Object { $_ -match '^Virtual size:\s+(\d+)' }
    if ($vsizeLine -match '^Virtual size:\s+(\d+)') {
        $vmb = [long][math]::Ceiling([long]$Matches[1] / 1MB)
        if ($vmb -lt 32768) { & $vm modifymedium disk $diskPath --resize 32768 | Out-Null }
    }
    $uuidOut2 = & $vm list vms | Where-Object { $_ -match "`"$([regex]::Escape($VM_NAME))`"" }
    if ($uuidOut2 -match '\{([^}]+)\}') { $state['vm_uuid'] = $Matches[1] }
    $state['vm_name'] = $VM_NAME
    $state['haos_version'] = $tag
    $state['asset_url'] = $assetUrl
    $state['asset_sha256'] = $digest
    $state['bridge_adapter'] = $adapter.Name
}

# ---------------------------------------------------------------- 4. start headless
Write-Host "`n[VM] start headless" -ForegroundColor Cyan
$status = (& $vm showvminfo $VM_NAME --machinereadable | Where-Object { $_ -match '^VMState=' })
if ($status -notmatch 'running') {
    & $vm startvm $VM_NAME --type headless | Out-Null
    Write-Host '  Started.'
} else {
    Write-Host '  Already running.'
}

# NEM (Hyper-V hosted) fallback: EFI CpuMpPei GP exception with 2 CPUs -> retry with 1.
$consoleLog = Join-Path $MACHINE_FOLDER 'console.log'
$nCpus = 2
for ($i = 0; $i -lt 6; $i++) {
    Start-Sleep -Seconds 5
    if (Test-Path -LiteralPath $consoleLog) {
        $txt = Get-Content -LiteralPath $consoleLog -Raw -ErrorAction SilentlyContinue
        if ($txt -and $txt -match 'Linux version') { break }
        if ($txt -and $txt -match 'X64 Exception Type') {
            & $vm controlvm $VM_NAME poweroff | Out-Null
            & $vm modifyvm $VM_NAME --cpus 1 | Out-Null
            & $vm startvm $VM_NAME --type headless | Out-Null
            $nCpus = 1
            break
        }
    }
}
$eff = (& $vm showvminfo $VM_NAME --machinereadable | Where-Object { $_ -match '^cpus=' }) -replace 'cpus=',''
Write-Host "  Effective vCPU count: $eff"
$state['vcpu'] = $eff

# ---------------------------------------------------------------- 5. wait for HA on 8123
Write-Host "`n[HA] waiting up to $([int]($MAX_WAIT_SEC/60)) minutes for http://homeassistant.local:8123 ..." -ForegroundColor Cyan

function Test-HaEndpoint([string]$url) {
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        return [int]$resp.StatusCode
    } catch { return $null }
}

$deadline = (Get-Date).AddSeconds($MAX_WAIT_SEC)
$workingUrl = $null
while ((Get-Date) -lt $deadline) {
    foreach ($u in @('http://homeassistant.local:8123', 'http://homeassistant:8123')) {
        if (Test-HaEndpoint $u) { $workingUrl = $u; break }
    }
    # DHCP-discovered address via VirtualBox guest property.
    $ip = $null
    $gp = & $vm guestproperty get $VM_NAME '/VirtualBox/GuestInfo/Net/0/V4/IP'
    if ($gp -match 'Value:\s*([0-9.]+)') { $ip = $Matches[1] }
    if ($ip) {
        $u = "http://${ip}:8123"
        if (Test-HaEndpoint $u) { $workingUrl = $u; $ipFound = $ip }
    }
    if ($workingUrl) { break }
    Start-Sleep -Seconds 5
}

if ($workingUrl) {
    Write-Host "`n  Home Assistant is reachable at: $workingUrl" -ForegroundColor Green
    $state['ha_url'] = $workingUrl
    if ($ip) { $state['guest_ip'] = $ip }
    $state['phase'] = 'onboarding-pending'
    Write-Host ''
    Write-Host 'ACTION REQUIRED: Create the Home Assistant owner account in the opened browser.' -ForegroundColor Yellow
    Write-Host "  URL: $workingUrl"
    Write-Host '  Do not send the credentials to the agent. Resume when onboarding is complete.'
} else {
    Write-Host '  TIMEOUT: Home Assistant not reachable yet. The VM boots; re-run verify-home-assistant.ps1 later.' -ForegroundColor Red
    $state['phase'] = 'booting-timeout'
}
Write-State $state
Write-Host "`nState written: $(Join-Path $PSScriptRoot 'local-state.json')"
