"""Detect sustained broadband frication in microphone audio frames."""

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
_MIN_HIGH_BAND_OCCUPANCY = 0.12


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

    def feed(self, frame: np.ndarray, *, sample_rate: int | None = None) -> bool:
        """Process one audio frame and return ``True`` if shush detected.

        Args:
            frame: Mono audio samples from one capture frame.
            sample_rate: Actual capture rate when it differs from the configured rate.

        Returns:
            ``True`` once the shush pattern has persisted for
            *consecutive_frames* frames in a row; ``False`` otherwise.
        """
        if self._is_shush_frame(frame, sample_rate=sample_rate):
            self._streak += 1
            if self._streak >= self._consecutive_frames:
                debug_log(
                    f"shush detected after {self._streak} consecutive frames",
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

    def _is_shush_frame(self, frame: np.ndarray, *, sample_rate: int | None = None) -> bool:
        """Return ``True`` if a single frame looks like a fricative."""
        samples = frame.reshape(-1)
        if samples.size < 2 or np.sqrt(np.mean(samples ** 2)) < self._energy_floor:
            return False

        spectrum = np.abs(np.fft.rfft(samples)) ** 2
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / (sample_rate or self._sample_rate))
        low_band = spectrum[(freqs >= self._low_band_hz[0]) & (freqs <= self._low_band_hz[1])]
        high_band = spectrum[(freqs >= self._high_band_hz[0]) & (freqs <= self._high_band_hz[1])]
        high_power = float(high_band.sum())
        if not high_power:
            return False

        # Frication distributes power across frequencies. Tones can have a
        # large high/low ratio while concentrating energy in only a few bins.
        occupancy = high_power ** 2 / (high_band.size * float(high_band @ high_band))
        if occupancy < _MIN_HIGH_BAND_OCCUPANCY:
            return False
        return high_power / high_band.size >= self._high_low_ratio * float(low_band.mean())
