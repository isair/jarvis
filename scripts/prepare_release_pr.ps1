# Create a release PR on develop (run from a git clone with origin configured).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".git")) {
    Write-Host "  ERROR  Not a git repository. Clone https://github.com/isair/jarvis.git first."
    exit 1
}

$branch = "feat/shell-cafe-release-$(Get-Date -Format 'yyyyMMdd')"
git checkout -b $branch 2>$null
if ($LASTEXITCODE -ne 0) {
    git checkout $branch
}

git add -A
git status

Write-Host ""
Write-Host "  Review staged files, then:"
Write-Host "  git commit -m `"feat: shell settings, daemon lock, cafe CI and smoke`""
Write-Host "  git push -u origin HEAD"
Write-Host "  gh pr create --base develop --title `"feat: shell cafe integration and release smoke`" --body-file docs/pr_release_smoke_body.md"
Write-Host "  # Then run the review-pr skill on the PR URL"
