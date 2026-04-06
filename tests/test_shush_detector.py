"""Tests for acoustic shush detection via spectral analysis."""

import numpy as np
import pytest

from jarvis.listening.shush_detector import ShushDetector


def _make_silence(n_samples: int = 320) -> np.ndarray:
    """Return a silent frame (all zeros)."""
    return np.zeros(n_samples, dtype=np.float32)


def _make_shush(
    n_samples: int = 320,
    sample_rate: int = 16_000,
    amplitude: float = 0.1,
) -> np.ndarray:
    """Synthesise a shush-like frame: band-limited noise in 2-8 kHz."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, n_samples).astype(np.float32)

    # Band-pass via FFT: keep only 2-8 kHz.
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    mask = (freqs >= 2_000) & (freqs <= 8_000)
    spectrum[~mask] = 0
    filtered = np.fft.irfft(spectrum, n=n_samples).astype(np.float32)

    # Normalise to desired amplitude.
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered = filtered * (amplitude / peak)
    return filtered


def _make_speech(
    n_samples: int = 320,
    sample_rate: int = 16_000,
    fundamental: float = 150.0,
    amplitude: float = 0.1,
) -> np.ndarray:
    """Synthesise a voiced-speech-like frame: harmonic with low-freq energy."""
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    # Fundamental + a few harmonics → strong low-frequency content.
    signal = np.zeros(n_samples, dtype=np.float32)
    for harmonic in range(1, 6):
        signal += np.sin(2 * np.pi * fundamental * harmonic * t).astype(np.float32) / harmonic
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal * (amplitude / peak)
    return signal


def _make_sibilant(
    n_samples: int = 320,
    sample_rate: int = 16_000,
    amplitude: float = 0.1,
) -> np.ndarray:
    """Synthesise a brief sibilant ('s' in speech) — same spectral shape as
    shush but only used for a few frames, not sustained."""
    return _make_shush(n_samples, sample_rate, amplitude)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestShushFrameDetection:
    """Verify single-frame spectral classification."""

    def test_silence_is_not_shush(self):
        det = ShushDetector()
        assert det._is_shush_frame(_make_silence()) is False

    def test_shush_frame_is_detected(self):
        det = ShushDetector()
        assert det._is_shush_frame(_make_shush()) is True

    def test_speech_frame_is_not_shush(self):
        det = ShushDetector()
        assert det._is_shush_frame(_make_speech()) is False

    def test_low_energy_shush_below_floor(self):
        """A shush quieter than the energy floor is ignored."""
        det = ShushDetector(energy_floor=0.5)
        frame = _make_shush(amplitude=0.01)
        assert det._is_shush_frame(frame) is False

    def test_custom_ratio_threshold(self):
        """A lenient threshold accepts marginal frames that a strict one rejects."""
        # Build a frame with moderate high/low ratio (~12) by mixing shush
        # with faint speech harmonics.
        shush = _make_shush(amplitude=0.1)
        speech = _make_speech(amplitude=0.005)
        mixed = shush + speech
        det_lenient = ShushDetector(high_low_ratio=1.5)
        det_strict = ShushDetector(high_low_ratio=20.0)
        assert det_lenient._is_shush_frame(mixed) is True
        assert det_strict._is_shush_frame(mixed) is False


@pytest.mark.unit
class TestSustainedShushDetection:
    """Verify the consecutive-frames streak logic."""

    def test_sustained_shush_triggers(self):
        """Feeding enough consecutive shush frames returns True."""
        n_required = 10
        det = ShushDetector(consecutive_frames=n_required)
        for i in range(n_required - 1):
            assert det.feed(_make_shush()) is False
        assert det.feed(_make_shush()) is True

    def test_interrupted_streak_resets(self):
        """A non-shush frame in the middle resets the counter."""
        n_required = 10
        det = ShushDetector(consecutive_frames=n_required)
        for _ in range(n_required - 2):
            det.feed(_make_shush())
        # Break the streak with a speech frame.
        det.feed(_make_speech())
        # Start again — need full n_required again.
        for i in range(n_required - 1):
            assert det.feed(_make_shush()) is False
        assert det.feed(_make_shush()) is True

    def test_silence_does_not_trigger(self):
        """Feeding silence never triggers."""
        det = ShushDetector(consecutive_frames=5)
        for _ in range(50):
            assert det.feed(_make_silence()) is False

    def test_brief_sibilant_does_not_trigger(self):
        """A short sibilant (< required frames) doesn't trigger."""
        n_required = 15
        det = ShushDetector(consecutive_frames=n_required)
        # 5 frames of sibilant (~100 ms) — too short.
        for _ in range(5):
            det.feed(_make_sibilant())
        # Then speech.
        for _ in range(20):
            det.feed(_make_speech())
        # Should never have triggered.
        assert det._streak == 0

    def test_reset_clears_streak(self):
        """Calling reset() drops the counter to zero."""
        det = ShushDetector(consecutive_frames=10)
        for _ in range(8):
            det.feed(_make_shush())
        assert det._streak == 8
        det.reset()
        assert det._streak == 0

    def test_continues_returning_true_after_threshold(self):
        """Once threshold is met, subsequent shush frames keep returning True."""
        n_required = 5
        det = ShushDetector(consecutive_frames=n_required)
        for _ in range(n_required):
            det.feed(_make_shush())
        # Next frame should still return True.
        assert det.feed(_make_shush()) is True
