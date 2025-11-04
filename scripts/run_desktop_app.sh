#!/bin/bash

# Test script for the Jarvis Desktop App

echo "🔧 Testing Jarvis Desktop App locally..."
echo ""

# Check Python version
echo "📋 Checking Python version..."
python --version || python3 --version
echo ""

# Install desktop dependencies if needed
echo "📦 Installing desktop app dependencies..."
pip install PyQt6 psutil pillow || pip3 install PyQt6 psutil pillow
echo ""

# Generate icons
echo "🎨 Generating icons..."
cd "$(dirname "$0")/.." || exit
python src/jarvis/desktop_assets/generate_icons.py
echo ""

# Run the desktop app
echo "🚀 Starting desktop app..."
echo "   Click the system tray icon to open menu"
echo "   Select 'Start Listening' from menu to begin"
echo "   Or press Ctrl+C to quit"
echo ""

# Set PYTHONPATH to include src directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

python -m jarvis.desktop_app

