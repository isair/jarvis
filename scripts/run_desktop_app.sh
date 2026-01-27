#!/bin/bash

# Test script for the Jarvis Desktop App

# Parse arguments
VOICE_DEBUG=0
for arg in "$@"; do
    case $arg in
        --voice-debug)
            VOICE_DEBUG=1
            shift
            ;;
    esac
done

# Navigate to project root first
cd "$(dirname "$0")/.." || exit

echo "🔧 Testing Jarvis Desktop App locally..."
if [ "$VOICE_DEBUG" = "1" ]; then
    echo "   📋 Voice debug: ENABLED"
fi
echo ""

# Check Python version
echo "📋 Checking Python version..."
python --version || python3 --version
echo ""

# Install dependencies from requirements.txt
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt || pip3 install -q -r requirements.txt
echo ""

# Generate icons
echo "🎨 Generating icons..."
python src/desktop_app/desktop_assets/generate_icons.py
echo ""

# Run the desktop app
echo "🚀 Starting desktop app..."
echo "   Click the system tray icon to open menu"
echo "   Select 'Start Listening' from menu to begin"
echo "   Or press Ctrl+C to quit"
echo ""

# Set PYTHONPATH to include src directory (already at project root)
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Set voice debug environment variable if requested
if [ "$VOICE_DEBUG" = "1" ]; then
    export JARVIS_VOICE_DEBUG=1
fi

python -m desktop_app

