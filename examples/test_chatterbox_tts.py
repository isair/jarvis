#!/usr/bin/env python3
"""
Test script for Chatterbox TTS functionality.

Usage:
    python examples/test_chatterbox_tts.py
    
This script will test both system TTS and Chatterbox TTS if available.
"""

import sys
import os
import time

# Add src to path so we can import jarvis modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jarvis.tts import TextToSpeech, ChatterboxTTS, create_tts_engine


def test_system_tts():
    """Test the system TTS engine."""
    print("🔊 Testing System TTS...")
    
    tts = TextToSpeech(enabled=True)
    tts.start()
    
    test_text = "Hello! This is a test of the system text-to-speech engine."
    print(f"Speaking: {test_text}")
    
    tts.speak(test_text)
    
    # Wait for speech to complete
    while tts.is_speaking():
        time.sleep(0.1)
    
    tts.stop()
    print("✅ System TTS test completed")


def test_chatterbox_tts():
    """Test the Chatterbox TTS engine."""
    print("🎤 Testing Chatterbox TTS...")
    
    try:
        tts = ChatterboxTTS(enabled=True, exaggeration=0.7, cfg_weight=0.5)
        tts.start()
        
        test_text = "Hello! This is a test of Chatterbox, the state-of-the-art open source text-to-speech engine."
        print(f"Speaking: {test_text}")
        
        tts.speak(test_text)
        
        # Wait for speech to complete
        while tts.is_speaking():
            time.sleep(0.1)
        
        tts.stop()
        print("✅ Chatterbox TTS test completed")
        
    except Exception as e:
        print(f"❌ Chatterbox TTS test failed: {e}")
        print("   Make sure you have installed: pip install chatterbox-tts pygame")


def test_factory_function():
    """Test the TTS factory function."""
    print("🏭 Testing TTS factory function...")
    
    # Test system engine
    system_tts = create_tts_engine(engine="system", enabled=True)
    print(f"System engine type: {type(system_tts).__name__}")
    
    # Test chatterbox engine
    chatterbox_tts = create_tts_engine(engine="chatterbox", enabled=True)
    print(f"Chatterbox engine type: {type(chatterbox_tts).__name__}")
    
    print("✅ Factory function test completed")


def main():
    """Run all TTS tests."""
    print("🧪 Jarvis TTS Test Suite")
    print("=" * 50)
    
    try:
        test_factory_function()
        print()
        
        test_system_tts()
        print()
        
        test_chatterbox_tts()
        print()
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("🏁 Test suite completed")


if __name__ == "__main__":
    main()
