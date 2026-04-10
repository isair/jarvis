"""Acoustic shush detector using spectral analysis.

Detects sustained fricative sounds ("shhh") by analysing the spectral
profile of audio frames.  A deliberate shush has a distinctive signature:
high-frequency energy (2-8 kHz) with very little low-frequency content
(< 500 Hz) and no harmonic structure.  By requiring the pattern to persist
across several consecutive frames we avoid false positives from brief
sibilants in normal speech or TTS output.

This runs directly on raw audio frames in the audio loop — no Whisper
transcription needed — giving sub-200 ms detection latency.
"""

from __future__ import annotations

import numpy as np

from ..debug import debug_log

# ---------------------------------------------------------------------------
# Defaults (overridable via config)
# ---------------------------------------------------------------------------
_DEFAULT_SAMPLE_RATE = 16_000
_DEFAULT_HIGH_LOW_RATIO = 3.0          # min high-band / low-band energy ratio
_DEFAULT_ENERGY_FLOOR = 0.002          # min RMS to ignore silence
_DEFAULT_CONSECUTIVE_FRAMES = 15       # ~300 ms at 20 ms frames
_DEFAULT_LOW_BAND_HZ = (0, 500)
_DEFAULT_HIGH_BAND_HZ = (2_000, 8_000)


class ShushDetector:
    """Detect sustained "shhh" sounds from raw float32 audio frames.

    Call :meth:`feed` with each 20 ms frame while TTS is playing.  It
    returns ``True`` the moment the shush pattern has been sustained long
    enough to be confident.  Call :meth:`reset` when TTS stops or the
    shush triggers an interrupt so the counter restarts.
    """

    def __init__(
        self,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        high_low_ratio: float = _DEFAULT_HIGH_LOW_RATIO,
        energy_floor: float = _DEFAULT_ENERGY_FLOOR,
        consecutive_frames: int = _DEFAULT_CONSECUTIVE_FRAMES,
        low_band_hz: tuple[int, int] = _DEFAULT_LOW_BAND_HZ,
        high_band_hz: tuple[int, int] = _DEFAULT_HIGH_BAND_HZ,
    ) -> None:
        self._sample_rate = sample_rate
        self._high_low_ratio = high_low_ratio
        self._energy_floor = energy_floor
        self._consecutive_frames = consecutive_frames
        self._low_band_hz = low_band_hz
        self._high_band_hz = high_band_hz

        # Running counter of consecutive shush-like frames.
        self._streak = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, frame: np.ndarray) -> bool:
        """Process one audio frame and return ``True`` if shush detected.

        Args:
            frame: 1-D float32 array (mono, 16 kHz, typically 320 samples
                   for a 20 ms frame).

        Returns:
            ``True`` once the shush pattern has persisted for
            *consecutive_frames* frames in a row; ``False`` otherwise.
        """
        if self._is_shush_frame(frame):
            self._streak += 1
            if self._streak >= self._consecutive_frames:
                debug_log(
                    f"shush detected after {self._streak} consecutive frames "
                    f"({self._streak * 20}ms)",
                    "voice",
                )
                return True
        else:
            if self._streak > 0:
                self._streak = 0
        return False

    def reset(self) -> None:
        """Reset the consecutive-frame counter."""
        self._streak = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_shush_frame(self, frame: np.ndarray) -> bool:
        """Return ``True`` if a single frame looks like a fricative."""
        flat = frame.flatten()

        # --- Energy gate: ignore silence ---
        rms = float(np.sqrt(np.mean(flat ** 2)))
        if rms < self._energy_floor:
            return False

        # --- Spectral analysis ---
        n = len(flat)
        if n < 2:
            return False

        spectrum = np.abs(np.fft.rfft(flat))
        freqs = np.fft.rfftfreq(n, d=1.0 / self._sample_rate)

        low_mask = (freqs >= self._low_band_hz[0]) & (freqs <= self._low_band_hz[1])
        high_mask = (freqs >= self._high_band_hz[0]) & (freqs <= self._high_band_hz[1])

        low_energy = float(np.mean(spectrum[low_mask] ** 2)) if np.any(low_mask) else 0.0
        high_energy = float(np.mean(spectrum[high_mask] ** 2)) if np.any(high_mask) else 0.0

        # Avoid division by zero — if low band is near-silent, any
        # meaningful high-band energy counts.
        if low_energy < 1e-12:
            return high_energy > 0
        ratio = high_energy / low_energy
        return ratio >= self._high_low_ratio
