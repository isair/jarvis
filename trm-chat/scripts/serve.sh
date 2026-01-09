#!/bin/bash
# Start the TRM-Chat inference server

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export TRM_MODEL_PATH="${TRM_MODEL_PATH:-./models/qwen2.5-3b}"
export TRM_HEADS_PATH="${TRM_HEADS_PATH:-}"

# Find latest checkpoint if TRM_HEADS_PATH not set
if [ -z "$TRM_HEADS_PATH" ] && [ -d "./checkpoints" ]; then
    LATEST=$(ls -t checkpoints/trm_heads_step_*.safetensors 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        # Remove .safetensors extension for the path
        export TRM_HEADS_PATH="${LATEST%.safetensors}"
        echo "Using latest checkpoint: $TRM_HEADS_PATH"
    fi
fi

echo "=================================="
echo "TRM-Chat Inference Server"
echo "=================================="
echo "Model: $TRM_MODEL_PATH"
echo "TRM Heads: ${TRM_HEADS_PATH:-Not loaded (base model only)}"
echo ""
echo "Server will be available at http://localhost:8000"
echo "API endpoint: POST http://localhost:8000/api/chat"
echo ""
echo "To use with Jarvis, update config.json:"
echo '  "ollama_base_url": "http://127.0.0.1:8000"'
echo '  "ollama_chat_model": "trm-chat"'
echo ""

# Start server
uvicorn src.inference.server:app --host 0.0.0.0 --port 8000
