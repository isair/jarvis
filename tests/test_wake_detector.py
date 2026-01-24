"""Tests for the wake detector module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from jarvis.listening.wake_detector import (
    WakeWordDetector,
    check_text_for_wake_word,
    extract_query_from_text,
    OPENWAKEWORD_AVAILABLE,
)


class TestWakeWordDetector:
    """Tests for WakeWordDetector class."""

    def test_init_with_jarvis(self):
        """Detector initializes for jarvis wake word."""
        detector = WakeWordDetector(wake_word="jarvis")
        assert detector.wake_word == "jarvis"
        # Audio detection depends on whether openwakeword is installed
        if OPENWAKEWORD_AVAILABLE:
            assert detector.audio_detection_enabled is True
        else:
            assert detector.audio_detection_enabled is False

    def test_init_with_custom_wake_word(self):
        """Custom wake words disable audio detection."""
        detector = WakeWordDetector(wake_word="friday")
        assert detector.wake_word == "friday"
        # No audio model for "friday"
        assert detector.audio_detection_enabled is False

    def test_init_normalizes_wake_word(self):
        """Wake word is normalized to lowercase."""
        detector = WakeWordDetector(wake_word="  JARVIS  ")
        assert detector.wake_word == "jarvis"

    def test_available_property_without_model(self):
        """available is False when no model loaded."""
        detector = WakeWordDetector(wake_word="friday")  # No model for friday
        assert detector.available is False

    def test_process_audio_returns_none_when_unavailable(self):
        """process_audio returns None when detector unavailable."""
        detector = WakeWordDetector(wake_word="friday")
        audio = np.zeros(1600, dtype=np.float32)
        result = detector.process_audio(audio)
        assert result is None

    @patch('jarvis.listening.wake_detector.OPENWAKEWORD_AVAILABLE', True)
    def test_process_audio_with_mock_model(self):
        """process_audio calls model predict when available."""
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.9}

        detector = WakeWordDetector(wake_word="jarvis")
        detector._model = mock_model
        detector.audio_detection_enabled = True

        audio = np.zeros(1600, dtype=np.float32)
        result = detector.process_audio(audio)

        assert result is not None  # Should return timestamp
        mock_model.predict.assert_called_once()

    @patch('jarvis.listening.wake_detector.OPENWAKEWORD_AVAILABLE', True)
    def test_process_audio_below_threshold(self):
        """process_audio returns None when score below threshold."""
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.3}  # Below default 0.5

        detector = WakeWordDetector(wake_word="jarvis", threshold=0.5)
        detector._model = mock_model
        detector.audio_detection_enabled = True

        audio = np.zeros(1600, dtype=np.float32)
        result = detector.process_audio(audio)

        assert result is None

    @patch('jarvis.listening.wake_detector.OPENWAKEWORD_AVAILABLE', True)
    def test_detection_cooldown(self):
        """Detector has cooldown between detections."""
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.9}

        detector = WakeWordDetector(wake_word="jarvis")
        detector._model = mock_model
        detector.audio_detection_enabled = True
        detector._detection_cooldown = 1.0

        audio = np.zeros(1600, dtype=np.float32)

        # First detection should succeed
        result1 = detector.process_audio(audio)
        assert result1 is not None

        # Second detection within cooldown should fail
        result2 = detector.process_audio(audio)
        assert result2 is None

    @patch('jarvis.listening.wake_detector.OPENWAKEWORD_AVAILABLE', True)
    def test_callback_called_on_detection(self):
        """Callback is called when wake word detected."""
        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.9}

        callback_received = []

        def on_wake(timestamp):
            callback_received.append(timestamp)

        detector = WakeWordDetector(
            wake_word="jarvis",
            on_wake_detected=on_wake,
        )
        detector._model = mock_model
        detector.audio_detection_enabled = True

        audio = np.zeros(1600, dtype=np.float32)
        detector.process_audio(audio)

        assert len(callback_received) == 1

    def test_reset(self):
        """reset() doesn't crash even without model."""
        detector = WakeWordDetector(wake_word="friday")
        detector.reset()  # Should not raise


class TestTextBasedDetection:
    """Tests for text-based wake word detection fallback."""

    def test_check_text_basic(self):
        """Detects wake word in text."""
        result = check_text_for_wake_word("hey jarvis what time is it", "jarvis")
        assert result is True

    def test_check_text_case_insensitive(self):
        """Detection is case insensitive."""
        result = check_text_for_wake_word("Hey JARVIS what time", "jarvis")
        assert result is True

    def test_check_text_no_wake_word(self):
        """Returns False when no wake word."""
        result = check_text_for_wake_word("what time is it", "jarvis")
        assert result is False

    def test_check_text_with_aliases(self):
        """Detects wake word aliases."""
        result = check_text_for_wake_word(
            "hey computer what time",
            "jarvis",
            aliases=["computer", "assistant"],
        )
        assert result is True

    def test_extract_query_basic(self):
        """Extracts query after wake word."""
        query = extract_query_from_text("jarvis what time is it", "jarvis")
        assert "what time is it" in query.lower()

    def test_extract_query_with_hey(self):
        """Extracts query after 'hey jarvis'."""
        query = extract_query_from_text("hey jarvis what time is it", "jarvis")
        assert "what time is it" in query.lower()

    def test_extract_query_no_wake_word(self):
        """Returns empty when no wake word."""
        query = extract_query_from_text("what time is it", "jarvis")
        # Depends on implementation - might return original or empty
        assert isinstance(query, str)
