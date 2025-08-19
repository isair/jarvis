#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH="$REPO_ROOT/src"
# Point to example JSON config; keep debug via env var optional
export JARVIS_CONFIG_PATH="$REPO_ROOT/examples/config.stub.json"
export JARVIS_VOICE_DEBUG=${JARVIS_VOICE_DEBUG:-0}
python -m jarvis.daemon < examples/sample.log
