# P0: fork isair/jarvis, push develop, open PR into upstream develop.
# Prereqs: GitHub fork at https://github.com/<YOUR_USER>/jarvis (one-time, in browser).
# Optional: GitHub CLI (gh) for PR creation — winget install GitHub.cli

param(
    [string]$GitHubUser = "JansonsJr",
    [string]$Upstream = "isair/jarvis",
    [string]$Branch = "develop"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$ForkUrl = "https://github.com/$GitHubUser/jarvis.git"
$ForkApi = "https://api.github.com/repos/$GitHubUser/jarvis"

Write-Host "  P0 ship — fork, push, PR"
Write-Host "  Repo: $Root"
Write-Host "  Fork:  $GitHubUser/jarvis"
Write-Host "  Upstream PR base: $Upstream branch $Branch"
Write-Host ""

try {
    Invoke-RestMethod -Uri $ForkApi -TimeoutSec 15 | Out-Null
    Write-Host "  OK    Fork exists on GitHub"
} catch {
    Write-Host "  STOP  Fork not found: $ForkApi"
    Write-Host "        Open in browser: https://github.com/$Upstream/fork"
    Write-Host "        Create the fork, then re-run this script."
    exit 1
}

if (-not (git rev-parse --is-inside-work-tree 2>$null)) {
    Write-Host "  STOP  Not a git repository"
    exit 1
}

$current = git branch --show-current
if ($current -ne $Branch) {
    Write-Host "  STOP  Expected branch '$Branch', on '$current'"
    exit 1
}

git remote remove fork 2>$null
git remote add fork $ForkUrl 2>$null
if (-not (git remote get-url fork 2>$null)) {
    git remote add fork $ForkUrl
}

if (-not (git remote get-url upstream 2>$null)) {
    git remote add upstream "https://github.com/$Upstream.git"
    Write-Host "  OK    Added upstream remote"
}

Write-Host "  Pushing $Branch to fork..."
git push -u fork $Branch
if ($LASTEXITCODE -ne 0) {
    Write-Host "  STOP  git push failed (auth or permissions)"
    exit $LASTEXITCODE
}
Write-Host "  OK    Pushed to fork"

$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $gh) {
    Write-Host ""
    Write-Host "  gh CLI not installed. Open PR manually:"
    Write-Host "  https://github.com/$Upstream/compare/$Branch...${GitHubUser}:jarvis:$Branch"
    Write-Host ""
    Write-Host "  Title: feat: shell cafe integration, daemon lock, and Anthropic base URL"
    Write-Host "  Body:  docs/pr_release_smoke_body.md"
    exit 0
}

$bodyFile = Join-Path $Root "docs/pr_release_smoke_body.md"
gh pr create `
    --repo $Upstream `
    --base $Branch `
    --head "${GitHubUser}:$Branch" `
    --title "feat: shell cafe integration, daemon lock, and cafe-agent glue" `
    --body-file $bodyFile

if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK    PR created. Run /review-pr on the PR URL."
} else {
    Write-Host "  WARN  gh pr create failed — use compare URL above"
}
