#!/bin/bash
# Run Jarvis evaluation suite
#
# Usage:
#   ./scripts/run_evals.sh              # Run all evals (live + judge enabled)
#   ./scripts/run_evals.sh weather      # Run only weather-related evals
#   ./scripts/run_evals.sh -v           # Verbose output
#   ./scripts/run_evals.sh --no-live    # Exclude live LLM tests
#   ./scripts/run_evals.sh --no-judge   # Exclude LLM-as-judge tests
#   ./scripts/run_evals.sh --no-report  # Skip EVALS.md generation
#
# Environment variables:
#   EVAL_JUDGE_MODEL    - Model to use for LLM-as-judge (default: gpt-oss:20b)
#   EVAL_JUDGE_BASE_URL - Ollama base URL (default: http://localhost:11434)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Set default judge model
export EVAL_JUDGE_MODEL="${EVAL_JUDGE_MODEL:-gpt-oss:20b}"

echo ""
echo "┌────────────────────────────────────────────────────────────┐"
echo "│                  🧪 Jarvis Evaluation Suite                │"
echo "└────────────────────────────────────────────────────────────┘"
echo ""

# Check if Ollama is available
OLLAMA_AVAILABLE=false
OLLAMA_URL="${EVAL_JUDGE_BASE_URL:-http://localhost:11434}"
if curl -s "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    OLLAMA_AVAILABLE=true
    echo "  ✅ Ollama detected at ${OLLAMA_URL}"
    echo "  🤖 Judge model: ${EVAL_JUDGE_MODEL}"
else
    echo "  ⚠️  Ollama not detected at ${OLLAMA_URL}"
    echo "     LLM-as-judge tests will be skipped"
fi
echo ""

# Parse arguments (defaults: live=true, judge=true, report=true)
PYTEST_ARGS="-v"
FILTER=""
INCLUDE_LIVE=true
INCLUDE_JUDGE=true
GENERATE_REPORT=true

for arg in "$@"; do
    case $arg in
        --no-live)
            INCLUDE_LIVE=false
            ;;
        --no-judge)
            INCLUDE_JUDGE=false
            ;;
        --no-report)
            GENERATE_REPORT=false
            ;;
        --live)
            INCLUDE_LIVE=true
            ;;
        --judge)
            INCLUDE_JUDGE=true
            ;;
        -v|--verbose)
            PYTEST_ARGS="$PYTEST_ARGS -v"
            ;;
        -vv)
            PYTEST_ARGS="$PYTEST_ARGS -vv"
            ;;
        --*)
            PYTEST_ARGS="$PYTEST_ARGS $arg"
            ;;
        *)
            FILTER="$arg"
            ;;
    esac
done

# Enable report generation
if [ "$GENERATE_REPORT" = true ]; then
    export EVAL_GENERATE_REPORT=1
    echo "  📄 Report will be saved to EVALS.md"
fi

# Build the pytest command (--tb=short for cleaner tracebacks, -s to capture stdout for judge notes)
CMD="python -m pytest evals/ $PYTEST_ARGS --tb=short"

# Build exclusion filter
EXCLUDE_PATTERNS=""
if [ "$INCLUDE_LIVE" = false ]; then
    EXCLUDE_PATTERNS="Live"
    echo "  ⏭️  Skipping live LLM tests (remove --no-live to include)"
fi

if [ -n "$FILTER" ]; then
    if [ -n "$EXCLUDE_PATTERNS" ]; then
        CMD="$CMD -k '$FILTER and not $EXCLUDE_PATTERNS'"
    else
        CMD="$CMD -k '$FILTER'"
    fi
    echo "  🔍 Filter: $FILTER"
elif [ -n "$EXCLUDE_PATTERNS" ]; then
    CMD="$CMD -k 'not $EXCLUDE_PATTERNS'"
fi

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "  📂 Project: $PROJECT_ROOT"
echo "  🚀 Command: $CMD"
echo "────────────────────────────────────────────────────────────────"
echo ""

eval $CMD
EXIT_CODE=$?

echo ""
echo "────────────────────────────────────────────────────────────────"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ All evaluations passed!"
else
    echo "  ⚠️  Some evaluations failed (exit code: $EXIT_CODE)"
fi
echo ""
echo "  📖 Legend:"
echo "     PASSED  → Test passed"
echo "     FAILED  → Test failed"
echo "     SKIPPED → Test skipped (missing dependencies)"
echo "     XFAIL   → Expected failure (documents known limitation)"
echo "     XPASS   → Bug fixed! (expected failure now passes)"
echo ""
if [ "$GENERATE_REPORT" = true ]; then
    echo "  📄 Full report: EVALS.md"
    echo ""
fi
echo "────────────────────────────────────────────────────────────────"

exit $EXIT_CODE
