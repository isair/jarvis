#!/bin/bash
# Setup script for TRM-Chat
# This script installs dependencies and downloads the base model

set -e

echo "=================================="
echo "TRM-Chat Setup"
echo "=================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Clone nano-trm for reference (if not already cloned)
if [ ! -d "reference/nano-trm" ]; then
    echo ""
    echo "Cloning nano-trm for reference..."
    mkdir -p reference
    git clone https://github.com/olivkoch/nano-trm.git reference/nano-trm
fi

# Download base model
echo ""
echo "Downloading and converting Qwen2.5-3B to MLX format..."
echo "This may take a few minutes..."
mkdir -p models

python -m mlx_lm.convert \
    --hf-path Qwen/Qwen2.5-3B-Instruct \
    --mlx-path ./models/qwen2.5-3b \
    --quantize \
    --q-bits 8

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Prepare training data:"
echo "   python -m src.data.prepare --output-dir ./data"
echo ""
echo "2. Train TRM heads:"
echo "   python -m src.training.train_heads --train-data ./data/train.jsonl"
echo ""
echo "3. Start inference server:"
echo "   TRM_MODEL_PATH=./models/phi-3-mini uvicorn src.inference.server:app --port 8000"
