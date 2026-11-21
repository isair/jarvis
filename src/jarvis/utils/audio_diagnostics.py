"""Audio input metadata and bounded signal statistics, without audio storage."""
from math import sqrt


def input_device_label(sounddevice, device):
    """Describe an input without changing its selection or opening a stream."""
    name = device.get('name') or 'Unknown input'
    api_name = 'API unknown'
    try:
        candidate = sounddevice.query_hostapis(device['hostapi']).get('name')
        if isinstance(candidate, str) and candidate.strip():
            api_name = candidate.strip()
    except Exception:
        pass
    return f'{name} [{api_name}]'


class InputSignalLevels:
    """Accumulate one reporting window of channel and actual downmix levels."""

    def __init__(self):
        self.samples = 0
        self.channel_peak = None
        self._channel_squares = None
        self._mono_squares = 0.0
        self.mono_peak = 0.0

    def observe(self, source, mono):
        import numpy as np
        if not mono.size:
            return
        channels = np.asarray(source, dtype=np.float64)
        if channels.ndim == 1:
            channels = channels[:, None]
        if self.channel_peak is not None and len(self.channel_peak) != channels.shape[1]:
            self.__init__()
        peak = np.max(np.abs(channels), axis=0)
        squares = np.sum(channels * channels, axis=0)
        self.channel_peak = peak if self.channel_peak is None else np.maximum(self.channel_peak, peak)
        self._channel_squares = squares if self._channel_squares is None else self._channel_squares + squares
        mono_values = np.asarray(mono, dtype=np.float64)
        self._mono_squares += float(np.sum(mono_values * mono_values))
        self.mono_peak = max(self.mono_peak, float(np.max(np.abs(mono_values))))
        self.samples += len(mono_values)

    @property
    def max_channel_peak(self):
        return float(self.channel_peak.max()) if self.samples else 0.0

    def describe(self):
        if not self.samples:
            return 'samples=0, channel_peak=unavailable, channel_rms=unavailable, mono_peak=unavailable, mono_rms=unavailable'
        channel_count = len(self._channel_squares)
        rms = [sqrt(float(value) / self.samples) for value in self._channel_squares[:8]]
        peaks = ', '.join(f'{v:.6f}' for v in self.channel_peak[:8])
        levels = ', '.join(f'{v:.6f}' for v in rms[:8])
        return (
            f'samples={self.samples}, channels={channel_count}, shown={len(rms)}, '
            f'channel_peak=[{peaks}], channel_rms=[{levels}], '
            f'max_channel_peak={self.max_channel_peak:.6f}, mono_peak={self.mono_peak:.6f}, '
            f'mono_rms={sqrt(self._mono_squares / self.samples):.6f}'
        )
