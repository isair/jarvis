Describe "Native OpenVINO server launcher" {
    BeforeAll {
        $launcherPath = Join-Path $PSScriptRoot "..\start-native-server.ps1"
        $launcherText = Get-Content -LiteralPath $launcherPath -Raw
    }

    It "captures Python commands before OpenVINO setupvars can replace PATH" {
        ($launcherText -match '\$pythonCandidatesBeforeSetup = @\("python", "py"\)') | Should Be $true
        ($launcherText -match 'setupvars\.ps1 can replace PATH') | Should Be $true
        ($launcherText -match '\$pythonCandidatesBeforeSetup') | Should Be $true
    }

    It "resolves the Python launcher to a real interpreter before setupvars" {
        ($launcherText -match '\$pythonLauncherBeforeSetup = Get-Command "py"') | Should Be $true
        ($launcherText -match 'not itself an interpreter') | Should Be $true
        ($launcherText -match '\$launcherPath -3 -c') | Should Be $true
        ($launcherText -match '\$pythonFromLauncherBeforeSetup') | Should Be $true
    }

    It "uses command paths and common Windows Python installation paths" {
        ($launcherText -match '\$_\.Path') | Should Be $true
        ($launcherText -match 'C:\\Python\*\\python\.exe') | Should Be $true
        ($launcherText -match 'LOCALAPPDATA.*Programs\\Python') | Should Be $true
    }

    It "uses OPENVINO_ROOT when the retrieval wrapper does not receive a root" {
        $retrievalPath = Join-Path $PSScriptRoot "..\start-native-retrieval.ps1"
        $retrievalText = Get-Content -LiteralPath $retrievalPath -Raw

        ($retrievalText -match '\$env:OPENVINO_ROOT') | Should Be $true
        ($retrievalText -match '\$defaultOpenVinoRoot') | Should Be $true
        ($retrievalText -match 'if \(-not \$OpenVinoRoot\)') | Should Be $true
    }

    It "still supports an explicit Python executable" {
        ($launcherText -match 'if \(\$RequestedPython\)') | Should Be $true
        ($launcherText -match 'The requested Python executable was not found') | Should Be $true
    }

    It "runs the launcher path with the selected Python and OpenVINO setup" {
        $fixtureRoot = Join-Path ([IO.Path]::GetTempPath()) ("openvino-launcher-" + [guid]::NewGuid())
        $openVinoRoot = Join-Path $fixtureRoot "openvino"
        $pythonPath = Join-Path $fixtureRoot "python.cmd"
        $setupMarker = Join-Path $fixtureRoot "setup-marker.txt"

        New-Item -ItemType Directory -Path $openVinoRoot -Force | Out-Null
        @"
param([string]`$python_version)
Set-Content -LiteralPath '$setupMarker' -Value `$python_version
"@ | Set-Content -LiteralPath (Join-Path $openVinoRoot "setupvars.ps1")
        @"
@echo off
echo 3.13
exit /b 0
"@ | Set-Content -LiteralPath $pythonPath -Encoding ASCII

        try {
            & $launcherPath -OpenVinoRoot $openVinoRoot -Python $pythonPath -Port 8011 | Out-Null
            $exitCode = $LASTEXITCODE

            $exitCode | Should Be 0
            (Get-Content -LiteralPath $setupMarker -Raw).Trim() | Should Be "3.13"
        }
        finally {
            Remove-Item -LiteralPath $fixtureRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
