Describe "Qdrant launcher contract" {
    BeforeAll {
        $helperPath = Join-Path $PSScriptRoot "..\start-qdrant.ps1"
        $unifiedLauncherPath = Join-Path $PSScriptRoot "..\..\..\scripts\start-local-models.ps1"
        $helperText = Get-Content -LiteralPath $helperPath -Raw
        $unifiedLauncherText = Get-Content -LiteralPath $unifiedLauncherPath -Raw
    }

    It "starts only the qdrant Compose service" {
        ($helperText -match '& \$docker compose -f \$ComposeFile up -d qdrant') | Should Be $true
        (-not ($helperText -match '& \$docker compose -f \$ComposeFile up -d\s*\r?\n')) | Should Be $true
    }

    It "checks readiness before and after startup" {
        ($helperText -match 'Test-QdrantReady \$healthUrl') | Should Be $true
        ($helperText -match 'Qdrant did not become ready at \$healthUrl') | Should Be $true
    }

    It "wires Qdrant into the unified launcher" {
        ($unifiedLauncherText -match '\$qdrantScript = Join-Path') | Should Be $true
        ($unifiedLauncherText -match '\$qdrantHealthUrl = "http://127\.0\.0\.1:\$QdrantPort/healthz"') | Should Be $true
        ($unifiedLauncherText -match 'Starting Qdrant dependency') | Should Be $true
    }

    It "provides the native OpenVINO installation when the environment variable is unset" {
        ($unifiedLauncherText -match '\[string\]\$NativeNpuOpenVinoRoot = \$\(if \(\[string\]::IsNullOrWhiteSpace\(\$env:OPENVINO_ROOT\)\)') | Should Be $true
        ($unifiedLauncherText -match 'openvino_2026\.2\.1\\openvino_toolkit_windows_2026\.2\.1\.21919\.ede283a88e3_x86_64') | Should Be $true
    }

    It "passes the local Qwen profile parameters through the XPU launchers" {
        ($unifiedLauncherText -match '\[string\]\$NativeXpuModelId = "Qwen/Qwen3-VL-8B-Thinking"') | Should Be $true
        ($unifiedLauncherText -match '\[ValidateSet\("auto", "xpu", "cpu"\)\]') | Should Be $true
        ($unifiedLauncherText -match '\[int\]\$NativeXpuMaxNewTokens = 16384') | Should Be $true
        ($unifiedLauncherText -match 'Add-ParameterIfSet \$xpuGatewayArguments "-ModelId" \$NativeXpuModelId') | Should Be $true
        ($unifiedLauncherText -match 'Add-ParameterIfSet \$xpuBackendArguments "-MaxNewTokens" \$NativeXpuMaxNewTokens') | Should Be $true
    }
}
