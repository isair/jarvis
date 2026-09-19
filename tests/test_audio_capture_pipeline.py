"""Native-rate microphone frames reach VAD without silent rejection or loss."""

from collections import deque
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.listening.listener import VoiceListener

pytestmark = pytest.mark.unit


def listener(rate=48000):
    obj = VoiceListener.__new__(VoiceListener)
    obj.cfg = SimpleNamespace(voice_min_energy=0.0045, voice_debug=False)
    obj._samplerate = 16000
    obj._stream_samplerate = rate
    obj._frame_samples = rate * 20 // 1000
    obj._pending_audio = None
    obj._recent_audio_energy = deque(maxlen=20)
    obj._vad_error_logged = False
    obj._vad = None
    obj._should_stop = obj._dictation_active = False
    obj._callback_count = 0
    obj._audio_q = queue.Queue(maxsize=2)
    obj._reset_audio_health(now=0)
    return obj


@pytest.mark.parametrize('rate', [16000, 44100, 48000])
def test_native_frames_use_supported_vad_format(rate):
    obj = listener(rate)

    class StrictVad:
        def is_speech(self, pcm, sample_rate):
            assert sample_rate == 16000
            assert len(pcm) == 640  # 20 ms, mono int16 at 16 kHz
            return True

    obj._vad = StrictVad()
    assert obj._is_speech_frame(np.ones(obj._frame_samples, dtype=np.float32) * .1)


def test_partial_callback_frames_are_not_discarded():
    obj = listener(44100)
    audio = np.linspace(-.1, .1, obj._frame_samples * 3 + 17, dtype=np.float32)
    output = []
    for block in np.array_split(audio, 13):
        output.extend(obj._audio_frames(block[:, None]))
    np.testing.assert_array_equal(np.concatenate(output), audio[:-17])
    np.testing.assert_array_equal(obj._pending_audio, audio[-17:])


def test_vad_failure_warns_once_and_uses_energy_gate(capsys):
    obj = listener()
    class BrokenVad:
        def is_speech(self, *args):
            raise ValueError('invalid frame')
    obj._vad = BrokenVad()
    assert obj._is_speech_frame(np.ones(obj._frame_samples) * .1)
    assert not obj._is_speech_frame(np.zeros(obj._frame_samples))
    assert capsys.readouterr().out.count('Speech detection failed') == 1


def test_capture_health_distinguishes_missing_callbacks_and_silent_samples(capsys):
    obj = listener()
    obj._check_audio_health(now=6)
    assert 'No microphone callbacks' in capsys.readouterr().out
    obj._callback_count = 1
    obj._last_audio_callback = 11
    obj._audio_frames(np.zeros((obj._frame_samples, 1), dtype=np.float32))
    obj._check_audio_health(now=12)
    assert 'silent samples' in capsys.readouterr().out


def test_callback_status_and_queue_overflow_are_visible(capsys):
    obj = listener()
    for _ in range(3):
        obj._on_audio(np.ones((960, 1), dtype=np.float32), 960, None, 'input overflow')
    obj._check_audio_health(now=6)
    output = capsys.readouterr().out
    assert 'input overflow' in output
    assert '1' in output and 'dropped' in output


def test_dictation_pause_does_not_report_capture_failure(capsys):
    obj = listener()
    obj._dictation_active = True
    obj._check_audio_health(now=30)
    assert not capsys.readouterr().out


def test_stalled_capture_warns_once_then_reports_recovery(capsys):
    obj = listener()
    obj._last_audio_callback = 2
    obj._check_audio_health(now=8)
    obj._check_audio_health(now=14)
    assert capsys.readouterr().out.count('No microphone callbacks') == 1
    obj._last_audio_callback = 19
    obj._audio_frames(np.ones((obj._frame_samples, 1), dtype=np.float32) * .1)
    obj._check_audio_health(now=20)
    assert 'arriving again' in capsys.readouterr().out


def test_callback_exception_is_not_silenced(capsys):
    obj = listener()
    class BrokenInput:
        def copy(self):
            raise RuntimeError('capture buffer failed')
    obj._on_audio(BrokenInput(), 960, None, None)
    obj._check_audio_health(now=6)
    assert 'capture buffer failed' in capsys.readouterr().out
