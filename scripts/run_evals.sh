#!/bin/bash
# Run Jarvis evaluation suite
# 
# Usage:
#   ./scripts/run_evals.sh              # Run all evals (skips live LLM tests)
#   ./scripts/run_evals.sh weather      # Run only weather-related evals
#   ./scripts/run_evals.sh -v           # Verbose output
#   ./scripts/run_evals.sh --live       # Include live LLM tests (requires Ollama)
#   ./scripts/run_evals.sh --judge      # Include LLM-as-judge tests (requires Ollama + model)
#
# Environment variables:
#   EVAL_JUDGE_MODEL    - Model to use for LLM-as-judge (default: qwen3)
#   EVAL_JUDGE_BASE_URL - Ollama base URL (default: http://localhost:11434)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "🧪 Jarvis Evaluation Suite"
echo "============================================================"
echo ""

# Check if Ollama is available
OLLAMA_AVAILABLE=false
if curl -s "http://localhost:11434/api/tags" > /dev/null 2>&1; then
    OLLAMA_AVAILABLE=true
    echo "✓ Ollama detected at localhost:11434"
else
    echo "⚠️  Ollama not detected - LLM-as-judge tests will be skipped"
fi

# Parse arguments
PYTEST_ARGS="-v"
FILTER=""
INCLUDE_LIVE=false
EXCLUDE_JUDGE=true

for arg in "$@"; do
    case $arg in
        --live)
            INCLUDE_LIVE=true
            ;;
        --judge)
            EXCLUDE_JUDGE=false
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

# Build the pytest command
CMD="python -m pytest evals/ $PYTEST_ARGS"

# Build exclusion filter
EXCLUDE_PATTERNS=""
if [ "$INCLUDE_LIVE" = false ]; then
    EXCLUDE_PATTERNS="Live"
    echo "⏭️  Skipping live LLM tests (use --live to include)"
fi

if [ -n "$FILTER" ]; then
    if [ -n "$EXCLUDE_PATTERNS" ]; then
        CMD="$CMD -k '$FILTER and not $EXCLUDE_PATTERNS'"
    else
        CMD="$CMD -k '$FILTER'"
    fi
    echo "🔍 Filter: $FILTER"
elif [ -n "$EXCLUDE_PATTERNS" ]; then
    CMD="$CMD -k 'not $EXCLUDE_PATTERNS'"
fi

echo "📂 Running from: $PROJECT_ROOT"
echo "🚀 Command: $CMD"
echo ""
echo "------------------------------------------------------------"
echo ""

eval $CMD

echo ""
echo "============================================================"
echo "✅ Evaluation complete"
echo ""
echo "Legend:"
echo "  PASSED  - Test passed"
echo "  XFAIL   - Expected failure (documents known bug)"
echo "  SKIPPED - Test skipped (missing dependencies)"
echo "  XPASS   - Bug was fixed! (expected fail now passes)"

