[CmdletBinding()]
param(
    [string]$Python = "python",
    [int]$Port = 8010,
    [switch]$KeepCache
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$cacheRoot = Join-Path $root "cache"
$logs = Join-Path $root "cache-verification"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Stop-NativeServer {
    $processes = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "native_server\\app.py" }
    foreach ($process in $processes) {
        taskkill.exe /PID $process.ProcessId /T /F | Out-Null
    }
    Start-Sleep -Seconds 2
}

function Start-AndMeasure([string]$Label) {
    $out = Join-Path $logs "$Label.out.log"
    $err = Join-Path $logs "$Label.err.log"
    Remove-Item $out,$err -Force -ErrorAction SilentlyContinue
    $started = [Diagnostics.Stopwatch]::StartNew()
    $process = Start-Process pwsh -ArgumentList "-NoProfile","-File",(Join-Path $root "start-native-server.ps1"),"-Python",$Python,"-Port",$Port -WorkingDirectory (Get-Location) -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    do {
        Start-Sleep -Seconds 5
        try { $ready = Invoke-RestMethod "http://127.0.0.1:$Port/readyz" -TimeoutSec 2 } catch { $ready = $null }
    } while (-not $ready.ready -and -not $process.HasExited -and $started.Elapsed.TotalSeconds -lt 900)
    $started.Stop()
    if (-not $ready.ready) { Get-Content $out -Tail 50; Get-Content $err -Tail 50; throw "$Label did not become ready." }
    [pscustomobject]@{ label = $Label; ready_seconds = [math]::Round($started.Elapsed.TotalSeconds, 2); stdout = $out; stderr = $err }
}

Stop-NativeServer
if (-not $KeepCache) { Remove-Item $cacheRoot -Recurse -Force -ErrorAction SilentlyContinue }
$cold = Start-AndMeasure "cold"
Stop-NativeServer
$warm = Start-AndMeasure "warm"
Stop-NativeServer

$records = foreach ($label in "cold","warm") {
    $path = Join-Path $logs "$label.out.log"
    Get-Content $path | ForEach-Object {
        try { $_ | ConvertFrom-Json } catch { $null }
    } | Where-Object { $_.event -eq "compiled_model_ready" -and $_.profile }
}
$result = [pscustomobject]@{ cold = $cold; warm = $warm; profiles = $records }
$result | ConvertTo-Json -Depth 8 | Tee-Object (Join-Path $logs "summary.json")
