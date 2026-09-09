<#
.SYNOPSIS
    Provision the LadybugDB native module (BLACKGLASS persistent graph plane).

.DESCRIPTION
    LadybugDB is embedded, not a daemon: there is no port and no process to
    supervise. "Starting" it means making `@ladybugdb/core` resolvable from the
    extension package so BlackglassBackends can load it on the next window
    reload. That is why this script installs and verifies rather than launching.

    Why it matters: without the native module the security hypergraph lives in
    `InMemorySecurityGraphStore`, which means every finding, advisory, projection
    and human decision is discarded when the window reloads. The fabric stays
    *correct* — §39 requires it to degrade rather than crash — but it stops being
    memory.

    The install is deliberately `--save-optional`. A hard dependency would make
    a machine without prebuilt binaries for its platform fail the whole
    extension install; an optional one fails only this plane, which is exactly
    the blast radius the adapter's guarded dynamic import already assumes.

.EXAMPLE
    .\scripts\services\Start-LadybugDb.ps1
    .\scripts\services\Start-LadybugDb.ps1 -Force
#>
[CmdletBinding()]
param(
    [string]$Version = $env:ZOO_LADYBUG_VERSION,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $PSCommandPath
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptRoot)
$extensionRoot = Join-Path $repoRoot "src"
$moduleName = "@ladybugdb/core"

if (-not (Test-Path -LiteralPath (Join-Path $extensionRoot "package.json"))) {
    throw "Extension package not found at $extensionRoot. Run this from a Kelvin-Clyne checkout."
}

function Test-LadybugResolvable {
    # Resolution is asked of Node from the extension root, which is the same
    # question the adapter's dynamic import asks at runtime. Checking for a
    # directory under node_modules would pass on a half-extracted install.
    $probe = "try{require.resolve('$moduleName');process.exit(0)}catch(e){process.exit(1)}"
    & node -e $probe 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

Push-Location $extensionRoot
try {
    if ((Test-LadybugResolvable) -and -not $Force) {
        Write-Host "$moduleName is already installed and resolvable." -ForegroundColor Green
        Write-Host "Reload the VS Code window to attach the persistent BLACKGLASS graph." -ForegroundColor DarkGray
        return
    }

    $spec = if ($Version) { "$moduleName@$Version" } else { $moduleName }
    Write-Host "Installing $spec into $extensionRoot (optional dependency)..." -ForegroundColor Cyan

    $pnpm = Get-Command "pnpm" -ErrorAction SilentlyContinue
    if ($pnpm) {
        & pnpm add --save-optional $spec
    } else {
        Write-Host "pnpm not found; falling back to npm." -ForegroundColor Yellow
        & npm install --save-optional $spec
    }
    if ($LASTEXITCODE -ne 0) {
        throw @"
Installing $spec failed.

LadybugDB ships prebuilt native binaries per platform. If none exists for this
machine, BLACKGLASS keeps working on the in-memory graph -- the security
hypergraph simply will not survive a window reload.
"@
    }

    if (-not (Test-LadybugResolvable)) {
        throw "$moduleName installed but is still not resolvable from $extensionRoot. The native binding likely failed to build for this platform."
    }

    Write-Host "$moduleName installed and resolvable." -ForegroundColor Green
    Write-Host "Databases will be created under the extension global storage:" -ForegroundColor DarkGray
    Write-Host "  blackglass\graph\corpus.db     (shared synthetic corpus)" -ForegroundColor DarkGray
    Write-Host "  blackglass\graph\workspace.db  (private repository evidence)" -ForegroundColor DarkGray
    Write-Host "Reload the VS Code window to attach the persistent BLACKGLASS graph." -ForegroundColor Cyan
} finally {
    Pop-Location
}
