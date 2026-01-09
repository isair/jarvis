#!/bin/bash
# Training script for TRM-Chat
# Run this after setup.sh and data preparation

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate virtual environment
source venv/bin/activate

echo "=================================="
echo "TRM-Chat Training"
echo "=================================="

# Check if data exists
if [ ! -f "./data/train.jsonl" ]; then
    echo "Training data not found. Running data preparation..."
    python -m src.data.prepare --output-dir ./data
fi

# Train TRM heads
echo ""
echo "Starting TRM heads training..."
python -m src.training.train_heads \
    --base-model ./models/qwen2.5-3b \
    --train-data ./data/train.jsonl \
    --val-data ./data/val.jsonl \
    --output-dir ./checkpoints \
    --epochs 3 \
    --batch-size 2 \
    --lr 1e-4

echo ""
echo "Training complete!"
echo "Checkpoints saved to ./checkpoints/"
