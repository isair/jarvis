#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON_BIN="${JARVIS_PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if command -v python3.13 >/dev/null 2>&1; then
    PYTHON_BIN="python3.13"
  else
    PYTHON_BIN="python3"
  fi
fi

if [ ! -d .venv ]; then
  "${PYTHON_BIN}" -m venv .venv
fi
source .venv/bin/activate

REQ_HASH_FILE=".venv/.requirements.sha"
CURRENT_REQ_HASH="$(shasum requirements.txt | awk '{print $1}')"
INSTALLED_REQ_HASH=""
if [ -f "${REQ_HASH_FILE}" ]; then
  INSTALLED_REQ_HASH="$(cat "${REQ_HASH_FILE}")"
fi

if [ "${CURRENT_REQ_HASH}" != "${INSTALLED_REQ_HASH}" ]; then
  pip install -r requirements.txt
  printf '%s\n' "${CURRENT_REQ_HASH}" > "${REQ_HASH_FILE}"
fi

# Build Swift capture helper (scaffold)
if [ -d mac/CaptureCLI ]; then
  (cd mac/CaptureCLI && swift build -c release)
fi

export PYTHONPATH="$REPO_ROOT/src"
# Allow override via JARVIS_CONFIG_PATH; otherwise use default search path in code
export JARVIS_VOICE_DEBUG=${JARVIS_VOICE_DEBUG:-0}
python -m jarvis.daemon
