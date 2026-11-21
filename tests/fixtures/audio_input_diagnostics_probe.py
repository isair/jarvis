"""Cross-platform input diagnostics with owned metadata and numerical audio."""
from types import SimpleNamespace
import builtins
from unittest.mock import patch

_original_import = builtins.__import__


def _without_numpy(name, *args, **kwargs):
    if name == 'numpy' or name.startswith('numpy.'):
        raise ImportError('Owned NumPy-unavailable control')
    return _original_import(name, *args, **kwargs)


with patch('builtins.__import__', _without_numpy):
    from jarvis.utils.audio_diagnostics import InputSignalLevels, input_device_label
    assert input_device_label(SimpleNamespace(query_hostapis=lambda index: {'name': 'MME'}),
                              {'name': 'Owned mic', 'hostapi': 0}) == 'Owned mic [MME]'
    assert 'unavailable' in InputSignalLevels().describe()

import numpy as np


def main():
    audio = SimpleNamespace(query_hostapis=lambda index: {'name': ['MME', 'Windows WASAPI'][index]})
    devices = [dict(name='Microphone Array (Owned)', hostapi=index) for index in range(2)]
    labels = [input_device_label(audio, device) for device in devices]
    assert labels[0] != labels[1] and 'MME' in labels[0] and 'Windows WASAPI' in labels[1]
    signal = np.linspace(-.5, .5, 960, dtype=np.float32)
    source = np.column_stack((signal, -signal))
    original = source.copy()
    mono = source.mean(axis=1)
    levels = InputSignalLevels()
    levels.observe(source, mono)
    assert levels.max_channel_peak == float(np.max(np.abs(original)))
    assert levels.mono_peak == 0.0
    assert 'mono_rms=0.000000' in levels.describe()
    expected = np.sqrt(np.mean(original.astype(np.float64) ** 2, axis=0))
    assert f'channel_rms=[{expected[0]:.6f}, {expected[1]:.6f}]' in levels.describe()
    source[:] = 0
    assert levels.max_channel_peak > 0
    fresh = InputSignalLevels()
    fresh.observe(source, source.mean(axis=1))
    assert 'channel_rms=[0.000000, 0.000000]' in fresh.describe()
    many = InputSignalLevels()
    data = np.ones((960, 32), dtype=np.float32)
    many.observe(data, data.mean(axis=1))
    assert 'channels=32, shown=8' in many.describe() and len(many.describe()) < 1000
    print('✅ API-labelled inputs and signal diagnostics passed with owned audio fakes')


if __name__ == '__main__':
    main()
