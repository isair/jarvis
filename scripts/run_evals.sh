#!/usr/bin/env bash
# Run the offline eval suite and any available local model evaluations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON:-python}"
MODEL_SMALL="gemma4:e2b"
MODEL_LARGE="gpt-oss:20b"
BASE_URL="${EVAL_JUDGE_BASE_URL:-http://localhost:11434}"
REPEAT_COUNT="${EVAL_REPEAT_COUNT:-1}"
REPORT=true
MULTI=true
INCLUDE_LIVE=true
INCLUDE_JUDGE=true
FILTER=""
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --no-report) REPORT=false ;;
        --single) MULTI=false ;;
        --no-live) INCLUDE_LIVE=false ;;
        --no-judge) INCLUDE_JUDGE=false ;;
        --live) INCLUDE_LIVE=true ;;
        --judge) INCLUDE_JUDGE=true ;;
        -v|--verbose|-vv) EXTRA_ARGS+=("$arg") ;;
        --*) EXTRA_ARGS+=("$arg") ;;
        *) FILTER="$arg" ;;
    esac
done

echo "🧪 Jarvis evaluation suite"
echo "  🔎 Endpoint: $BASE_URL"
PROBED_PROVIDER=""

probe_model() {
    local model="$1"
    local probe_output
    if probe_output="$("$PYTHON_BIN" "$SCRIPT_DIR/eval_provider.py" --base-url "$BASE_URL" --model "$model")"; then
        PROBED_PROVIDER="$(printf '%s' "$probe_output" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("provider") or "")')"
        echo "  ✅ $model available"
        return 0
    fi
    local reason
    reason="$(printf '%s' "$probe_output" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["reason"])')"
    PROBED_PROVIDER="$(printf '%s' "$probe_output" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("provider") or "")')"
    echo "  ⚠️  $model unavailable ($reason)"
    return 1
}

run_model() {
    local model="$1"
    local availability="$2"
    local report_path="$3"
    local scenario_path="$4"
    local expression="$FILTER"
    local args=(-m pytest evals/ -v --tb=short "--count=$REPEAT_COUNT")
    if [ "$INCLUDE_LIVE" = false ]; then
        expression="${expression:+$expression and }not Live"
    fi
    if [ "$INCLUDE_JUDGE" = false ]; then
        expression="${expression:+$expression and }not Judge"
    fi
    if [ -n "$expression" ]; then
        args+=(-k "$expression")
    fi
    args+=("${EXTRA_ARGS[@]}")
    echo "  🤖 Model: $model ($availability)"
    local result=0
    EVAL_JUDGE_MODEL="$model" \
    EVAL_JUDGE_BASE_URL="$BASE_URL" \
    EVAL_PROVIDER="$PROBED_PROVIDER" \
    EVAL_MODEL_AVAILABILITY="$availability" \
    EVAL_GENERATE_REPORT="$([ "$REPORT" = true ] && echo 1 || echo 0)" \
    EVAL_REPORT_PATH="$report_path" \
    EVAL_SCENARIO_REPORT_PATH="$scenario_path" \
        "$PYTHON_BIN" "${args[@]}" || result=$?
    return "$result"
}

exit_code=0
small_available=false
large_available=false
if [ "$MULTI" = true ]; then
    probe_model "$MODEL_SMALL" && small_available=true
    probe_model "$MODEL_LARGE" && large_available=true
fi
if [ "$MULTI" = true ] && [ "$small_available" = true ] && [ "$large_available" = true ]; then
    eval_tmp="$(mktemp -d)"
    trap 'rm -rf "$eval_tmp"' EXIT
    run_model "$MODEL_SMALL" available "$eval_tmp/small.md" "$([ "$REPORT" = true ] && echo "$PROJECT_ROOT/EVALS_SCENARIOS_small.json")" || exit_code=$?
    run_model "$MODEL_LARGE" available "$eval_tmp/large.md" "$([ "$REPORT" = true ] && echo "$PROJECT_ROOT/EVALS_SCENARIOS_large.json")" || exit_code=$?
    if [ "$REPORT" = true ]; then
        "$PYTHON_BIN" "$SCRIPT_DIR/merge_eval_reports.py" \
            "$eval_tmp/small.md" "$MODEL_SMALL" \
            "$eval_tmp/large.md" "$MODEL_LARGE" > "$PROJECT_ROOT/EVALS.md"
        echo "  📄 Combined report: EVALS.md"
    fi
else
    selected="${EVAL_JUDGE_MODEL:-}"
    if [ -z "$selected" ]; then
        if [ "$small_available" = true ]; then
            selected="$MODEL_SMALL"
        elif [ "$large_available" = true ]; then
            selected="$MODEL_LARGE"
        else
            selected="$MODEL_SMALL"
        fi
    fi
    availability=unavailable
    if probe_model "$selected"; then
        availability=available
    fi
    run_model "$selected" "$availability" "$PROJECT_ROOT/EVALS.md" "$([ "$REPORT" = true ] && echo "$PROJECT_ROOT/EVALS_SCENARIOS.json")" || exit_code=$?
fi

if [ "$exit_code" -eq 0 ]; then
    echo "  ✅ Executed evaluations passed"
else
    echo "  ⚠️  Executed evaluations failed (exit code $exit_code)"
fi
exit "$exit_code"
