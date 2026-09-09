# Build for the existing env: kill jarvis processes, clean, PyInstaller onedir, no tests.
Set-Location $PSScriptRoot/..
$env:PYTHONPATH = "$PWD\src"

# Free the mapped .pyd/.dll files so Remove-Item can delete dist\.
Get-Process -Name Jarvis, python, python313 -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -ne $PID } |
    ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
Start-Sleep -Milliseconds 700

if (Test-Path build) { Remove-Item -LiteralPath build -Recurse -Force }
if (Test-Path dist) { Remove-Item -LiteralPath dist -Recurse -Force }

# No test run here: pytest is 1 (skip) or 5 (no tests), both non-zero by design.
& C:\Users\lukes.COREI9\Miniconda3\python.exe -W ignore -m PyInstaller --noconfirm jarvis_desktop.spec
$pyi = $LASTEXITCODE
if ($pyi -ne 0) { exit $pyi }

& .\dist\Jarvis\Jarvis.exe --smoke-test
exit $LASTEXITCODE
