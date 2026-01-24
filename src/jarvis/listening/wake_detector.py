"""Audio-level wake word detection using openWakeWord.

This module provides audio-level wake word detection for instant response
when "hey jarvis" is spoken. Falls back to text-based detection for other
wake words or when openWakeWord is unavailable.
"""

import time
from typing import Optional, Callable
import numpy as np

from ..debug import debug_log

# Try to import openWakeWord
try:
    import openwakeword
    from openwakeword.model import Model as OWWModel
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    OWWModel = None


class WakeWordDetector:
    """Audio-level wake word detection using openWakeWord.

    This detector runs in parallel with the main audio processing,
    providing instant wake word detection at the audio level for
    "hey jarvis". For other wake words, it falls back to text-based
    detection in the transcript.

    Attributes:
        available: Whether audio-level detection is available
        wake_word: The configured wake word
        audio_detection_enabled: Whether audio-level detection is enabled
    """

    # The only wake word we have an audio model for
    SUPPORTED_WAKE_WORDS = {"jarvis"}
    MODEL_NAME = "hey_jarvis"

    def __init__(
        self,
        wake_word: str = "jarvis",
        threshold: float = 0.5,
        sample_rate: int = 16000,
        on_wake_detected: Optional[Callable[[float], None]] = None,
    ):
        """Initialize the wake word detector.

        Args:
            wake_word: The wake word to detect (audio-level only works for "jarvis")
            threshold: Detection confidence threshold (0.0-1.0)
            sample_rate: Audio sample rate in Hz
            on_wake_detected: Optional callback when wake word detected, receives timestamp
        """
        self.wake_word = wake_word.lower().strip()
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.on_wake_detected = on_wake_detected

        self._model: Optional[OWWModel] = None
        self._last_detection_time: float = 0.0
        self._detection_cooldown: float = 1.0  # Minimum seconds between detections

        # Only enable audio-level detection for supported wake words
        self.audio_detection_enabled = (
            OPENWAKEWORD_AVAILABLE and
            self.wake_word in self.SUPPORTED_WAKE_WORDS
        )

        if self.audio_detection_enabled:
            self._initialize_model()
        else:
            if not OPENWAKEWORD_AVAILABLE:
                debug_log("wake detector: openWakeWord not installed, using text-based only", "voice")
            elif self.wake_word not in self.SUPPORTED_WAKE_WORDS:
                debug_log(f"wake detector: no audio model for '{wake_word}', using text-based only", "voice")

    def _initialize_model(self) -> None:
        """Initialize the openWakeWord model."""
        try:
            # Load the hey_jarvis model
            self._model = OWWModel(
                wakeword_models=[self.MODEL_NAME],
                inference_framework="onnx",
            )
            debug_log(f"wake detector: loaded '{self.MODEL_NAME}' model", "voice")
        except Exception as e:
            debug_log(f"wake detector: failed to load model: {e}", "voice")
            self._model = None
            self.audio_detection_enabled = False

    @property
    def available(self) -> bool:
        """Check if audio-level wake detection is available."""
        return self.audio_detection_enabled and self._model is not None

    def process_audio(self, audio_chunk: np.ndarray) -> Optional[float]:
        """Process an audio chunk for wake word detection.

        Args:
            audio_chunk: Audio samples as numpy array (int16 or float32)

        Returns:
            Timestamp of wake word detection, or None if not detected
        """
        if not self.available:
            return None

        # Convert to int16 if needed (openWakeWord expects int16)
        if audio_chunk.dtype == np.float32:
            audio_chunk = (audio_chunk * 32767).astype(np.int16)

        try:
            # Run prediction
            prediction = self._model.predict(audio_chunk)

            # Check if wake word was detected
            score = prediction.get(self.MODEL_NAME, 0.0)

            if score >= self.threshold:
                current_time = time.time()

                # Apply cooldown to prevent multiple detections
                if current_time - self._last_detection_time < self._detection_cooldown:
                    return None

                self._last_detection_time = current_time
                debug_log(f"wake detector: '{self.MODEL_NAME}' detected (score={score:.3f})", "voice")

                # Call callback if registered
                if self.on_wake_detected:
                    self.on_wake_detected(current_time)

                return current_time

        except Exception as e:
            debug_log(f"wake detector: prediction error: {e}", "voice")

        return None

    def reset(self) -> None:
        """Reset the detector state.

        Call this after processing a wake word to clear internal buffers.
        """
        if self._model is not None:
            try:
                self._model.reset()
            except Exception:
                pass


def check_text_for_wake_word(
    text: str,
    wake_word: str,
    aliases: Optional[list] = None,
    fuzzy_threshold: float = 0.78,
) -> bool:
    """Check if text contains the wake word (text-based fallback).

    This is the fallback detection method used when audio-level
    detection is unavailable or for wake words without audio models.

    Args:
        text: Transcribed text to check
        wake_word: Primary wake word
        aliases: Optional list of wake word aliases
        fuzzy_threshold: Fuzzy matching threshold (0.0-1.0)

    Returns:
        True if wake word was detected in text
    """
    from .wake_detection import is_wake_word_detected

    text_lower = text.lower().strip()
    wake_word_lower = wake_word.lower().strip()
    aliases_lower = [a.lower().strip() for a in (aliases or [])]

    all_aliases = set(aliases_lower) | {wake_word_lower}
    return is_wake_word_detected(text_lower, wake_word_lower, list(all_aliases), fuzzy_threshold)


def extract_query_from_text(
    text: str,
    wake_word: str,
    aliases: Optional[list] = None,
) -> str:
    """Extract query text after the wake word (text-based fallback).

    Args:
        text: Transcribed text
        wake_word: Primary wake word
        aliases: Optional list of wake word aliases

    Returns:
        Query text after the wake word, or empty string
    """
    from .wake_detection import extract_query_after_wake

    text_lower = text.lower().strip()
    wake_word_lower = wake_word.lower().strip()
    aliases_lower = [a.lower().strip() for a in (aliases or [])]

    all_aliases = list(set(aliases_lower) | {wake_word_lower})
    return extract_query_after_wake(text_lower, wake_word_lower, all_aliases)
