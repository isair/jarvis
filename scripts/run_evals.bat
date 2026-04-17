@echo off
setlocal EnableDelayedExpansion
REM Run Jarvis evaluation suite on Windows
REM
REM Usage:
REM   run_evals.bat              Run all evals with both models (live + judge enabled)
REM   run_evals.bat weather      Run only weather-related evals
REM   run_evals.bat -v           Verbose output
REM   run_evals.bat --no-live    Exclude live LLM tests
REM   run_evals.bat --no-judge   Exclude LLM-as-judge tests
REM   run_evals.bat --no-report  Skip EVALS.md generation
REM   run_evals.bat --single     Run with single model only (EVAL_JUDGE_MODEL)
REM
REM Environment variables:
REM   EVAL_JUDGE_MODEL    - Model to use for LLM-as-judge (default: gpt-oss:20b)
REM   EVAL_JUDGE_BASE_URL - Ollama base URL (default: http://localhost:11434)
REM   EVAL_REPEAT_COUNT   - Number of times to run each test (default: 3)

REM Navigate to project root
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
set "SCRIPT_DIR=%~dp0"
cd /d "%PROJECT_ROOT%"

REM Resolve mamba env: prefer this checkout's own, fall back to the main
REM repo's when running from a git worktree (worktrees share one env).
set "MAMBA_ENV=%PROJECT_ROOT%\.mamba_env"
if not exist "!MAMBA_ENV!\python.exe" (
    for /f "usebackq delims=" %%G in (`git -C "%PROJECT_ROOT%" rev-parse --git-common-dir 2^>nul`) do (
        for %%I in ("%%G\..") do (
            if exist "%%~fI\.mamba_env\python.exe" set "MAMBA_ENV=%%~fI\.mamba_env"
        )
    )
)

if not exist "!MAMBA_ENV!\python.exe" (
    echo ERROR: Mamba environment not found.
    echo    Looked in: %PROJECT_ROOT%\.mamba_env
    echo    And the main repo's .mamba_env ^(if this is a git worktree^).
    echo Please run the setup script first.
    pause
    exit /b 1
)

set "PYTHON=!MAMBA_ENV!\python.exe"
set "PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%"

REM Officially supported models (from config.py)
set "MODEL_SMALL=gemma4:e2b"
set "MODEL_LARGE=gpt-oss:20b"

echo.
echo +------------------------------------------------------------+
echo ^|                  Jarvis Evaluation Suite                   ^|
echo +------------------------------------------------------------+
echo.

REM Check if Ollama is available
set "OLLAMA_AVAILABLE=false"
if defined EVAL_JUDGE_BASE_URL (
    set "OLLAMA_URL=!EVAL_JUDGE_BASE_URL!"
) else (
    set "OLLAMA_URL=http://localhost:11434"
)
curl -s "!OLLAMA_URL!/api/tags" >nul 2>&1
if not errorlevel 1 (
    set "OLLAMA_AVAILABLE=true"
    echo   Ollama detected at !OLLAMA_URL!
) else (
    echo   WARNING: Ollama not detected at !OLLAMA_URL!
    echo      LLM-as-judge tests will be skipped
)
echo.

REM Parse arguments
set "PYTEST_ARGS=-v"
set "FILTER="
set "INCLUDE_LIVE=true"
set "INCLUDE_JUDGE=true"
set "GENERATE_REPORT=true"
set "MULTI_MODEL=true"

:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="--no-live" (
    set "INCLUDE_LIVE=false"
    shift
    goto parse_args
)
if /i "%~1"=="--no-judge" (
    set "INCLUDE_JUDGE=false"
    shift
    goto parse_args
)
if /i "%~1"=="--no-report" (
    set "GENERATE_REPORT=false"
    shift
    goto parse_args
)
if /i "%~1"=="--single" (
    set "MULTI_MODEL=false"
    shift
    goto parse_args
)
if /i "%~1"=="--live" (
    set "INCLUDE_LIVE=true"
    shift
    goto parse_args
)
if /i "%~1"=="--judge" (
    set "INCLUDE_JUDGE=true"
    shift
    goto parse_args
)
if /i "%~1"=="-v" (
    set "PYTEST_ARGS=!PYTEST_ARGS! -v"
    shift
    goto parse_args
)
if /i "%~1"=="--verbose" (
    set "PYTEST_ARGS=!PYTEST_ARGS! -v"
    shift
    goto parse_args
)
if /i "%~1"=="-vv" (
    set "PYTEST_ARGS=!PYTEST_ARGS! -vv"
    shift
    goto parse_args
)
set "_FIRST_CHAR=%~1"
if "!_FIRST_CHAR:~0,2!"=="--" (
    set "PYTEST_ARGS=!PYTEST_ARGS! %~1"
    shift
    goto parse_args
)
set "FILTER=%~1"
shift
goto parse_args
:done_args

set "EXCLUDE_PATTERNS="
if "!INCLUDE_LIVE!"=="false" (
    set "EXCLUDE_PATTERNS=Live"
    echo   Skipping live LLM tests ^(remove --no-live to include^)
)

if "!GENERATE_REPORT!"=="true" (
    echo   Report will be saved to EVALS.md
)

set "FINAL_EXIT_CODE=0"
set "RUN_MULTI=false"
if "!MULTI_MODEL!"=="true" if "!OLLAMA_AVAILABLE!"=="true" set "RUN_MULTI=true"

if "!RUN_MULTI!"=="true" (
    echo   Running evals with both supported models for comparison

    set "TEMP_DIR=%TEMP%\jarvis_evals_%RANDOM%_%RANDOM%"
    mkdir "!TEMP_DIR!" >nul 2>&1

    set "EVAL_REPORT_PATH=!TEMP_DIR!\evals_small.md"
    call :run_evals_for_model "!MODEL_SMALL!" "_small"
    if errorlevel 1 set "FINAL_EXIT_CODE=1"

    echo   Unloading models before switching...
    curl -s "!OLLAMA_URL!/api/generate" -d "{\"model\":\"!MODEL_SMALL!\",\"keep_alive\":0}" >nul 2>&1
    timeout /t 2 /nobreak >nul

    set "EVAL_REPORT_PATH=!TEMP_DIR!\evals_large.md"
    call :run_evals_for_model "!MODEL_LARGE!" "_large"
    if errorlevel 1 set "FINAL_EXIT_CODE=1"

    if "!GENERATE_REPORT!"=="true" (
        "!PYTHON!" "!SCRIPT_DIR!merge_eval_reports.py" ^
            "!TEMP_DIR!\evals_small.md" "!MODEL_SMALL!" ^
            "!TEMP_DIR!\evals_large.md" "!MODEL_LARGE!" ^
            > "!PROJECT_ROOT!\EVALS.md"
        echo.
        echo   Combined report saved to EVALS.md
    )

    rmdir /s /q "!TEMP_DIR!" >nul 2>&1
) else (
    if not defined EVAL_JUDGE_MODEL set "EVAL_JUDGE_MODEL=!MODEL_LARGE!"
    set "EVAL_REPORT_PATH=!PROJECT_ROOT!\EVALS.md"
    call :run_evals_for_model "!EVAL_JUDGE_MODEL!" ""
    if errorlevel 1 set "FINAL_EXIT_CODE=1"
)

echo.
echo ----------------------------------------------------------------
if "!FINAL_EXIT_CODE!"=="0" (
    echo   All evaluations passed!
) else (
    echo   WARNING: Some evaluations failed ^(exit code: !FINAL_EXIT_CODE!^)
)
echo.
echo   Legend:
echo      PASSED  -^> Test passed
echo      FAILED  -^> Test failed
echo      SKIPPED -^> Test skipped ^(missing dependencies^)
echo      XFAIL   -^> Expected failure ^(documents known limitation^)
echo      XPASS   -^> Bug fixed! ^(expected failure now passes^)
echo.
if "!GENERATE_REPORT!"=="true" (
    echo   Full report: EVALS.md
    echo.
)
echo ----------------------------------------------------------------

exit /b !FINAL_EXIT_CODE!


:run_evals_for_model
REM %~1 = model, %~2 = report suffix
set "_MODEL=%~1"
set "_REPORT_SUFFIX=%~2"
set "EVAL_JUDGE_MODEL=!_MODEL!"

echo.
echo ================================================================
echo   Running evals with model: !_MODEL!
echo ================================================================
echo.

if defined EVAL_REPEAT_COUNT (
    set "_REPEAT_COUNT=!EVAL_REPEAT_COUNT!"
) else (
    set "_REPEAT_COUNT=3"
)

set "_CMD="!PYTHON!" -m pytest evals/ !PYTEST_ARGS! --tb=short --count=!_REPEAT_COUNT!"

if not "!FILTER!"=="" (
    if not "!EXCLUDE_PATTERNS!"=="" (
        set "_CMD=!_CMD! -k "!FILTER! and not !EXCLUDE_PATTERNS!""
    ) else (
        set "_CMD=!_CMD! -k "!FILTER!""
    )
) else if not "!EXCLUDE_PATTERNS!"=="" (
    set "_CMD=!_CMD! -k "not !EXCLUDE_PATTERNS!""
)

echo   Command: !_CMD!
echo.

if "!GENERATE_REPORT!"=="true" (
    set "EVAL_GENERATE_REPORT=1"
    set "EVAL_REPORT_SUFFIX=!_REPORT_SUFFIX!"
)

call !_CMD!
exit /b !errorlevel!
