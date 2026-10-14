"""
Tests for the dictation engine (hold-to-dictate feature).
"""

import threading
import time
from unittest.mock import patch, MagicMock, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(**overrides):
    """Create a DictationEngine with sensible test defaults."""
    from jarvis.dictation.dictation_engine import DictationEngine

    defaults = dict(
        whisper_model_ref=lambda: MagicMock(),
        whisper_backend_ref=lambda: "faster-whisper",
        mlx_repo_ref=lambda: None,
        hotkey="ctrl+shift+d",
        sample_rate=16000,
        on_dictation_start=None,
        on_dictation_end=None,
        transcribe_lock=threading.Lock(),
    )
    defaults.update(overrides)
    return DictationEngine(**defaults)


# ---------------------------------------------------------------------------
# Beep generation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.usefixtures('no_network')
class TestBeepGeneration:
    """Notification tones are valid, non-silent WAV data with requested timing."""

    def test_notification_beeps_are_distinct_and_non_silent(self):
        import io
        import wave
        import numpy as np
        from jarvis.dictation.dictation_engine import _get_start_beep, _get_stop_beep

        beeps = (_get_start_beep(), _get_stop_beep())
        assert beeps[0] != beeps[1]
        for data in beeps:
            with wave.open(io.BytesIO(data)) as audio:
                assert audio.getnchannels() == 1
                assert audio.getsampwidth() == 2
                samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2')
            assert np.any(samples), 'Notification tones must contain a non-silent signal'

    @pytest.mark.parametrize('frequency,duration', [(400, 0.04), (1000, 0.05), (1200, 0.075)])
    def test_generated_beep_preserves_requested_timing_and_pitch(self, frequency, duration):
        import io
        import wave
        import numpy as np
        from jarvis.dictation.dictation_engine import _generate_beep_wav

        with wave.open(io.BytesIO(_generate_beep_wav(freq=frequency, duration=duration))) as audio:
            rate = audio.getframerate()
            samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2')
        assert len(samples) / rate == pytest.approx(duration, abs=1 / rate)
        bins = np.fft.rfftfreq(len(samples), d=1 / rate)
        peak = bins[np.argmax(np.abs(np.fft.rfft(samples)))]
        assert peak == pytest.approx(frequency, abs=rate / len(samples))


# ---------------------------------------------------------------------------
# Hotkey parsing
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.usefixtures('no_network')
class TestHotkeyParsing:
    """Configured shortcuts retain the requested modifiers and trigger key."""

    @pytest.fixture
    def keyboard(self, monkeypatch):
        from enum import Enum
        from jarvis.dictation import dictation_engine

        class Key(Enum):
            ctrl_l = 'control'
            shift_l = 'shift'
            alt_l = 'alt'
            cmd = 'command'
            f5 = 'function-five'

        class KeyCode:
            @staticmethod
            def from_char(character):
                return ('character', character)

        class Keyboard:
            pass

        Keyboard.Key, Keyboard.KeyCode = Key, KeyCode
        monkeypatch.setattr(dictation_engine, 'pynput_keyboard', Keyboard)
        return Keyboard

    @pytest.mark.parametrize('combo,modifiers,trigger', [
        ('ctrl+shift+d', ('ctrl_l', 'shift_l'), ('character', 'd')),
        ('ctrl+cmd', ('ctrl_l', 'cmd'), None),
        ('ctrl+alt', ('ctrl_l', 'alt_l'), None),
        ('ctrl+win', ('ctrl_l', 'cmd'), None),
        ('ctrl+super', ('ctrl_l', 'cmd'), None),
        ('alt+x', ('alt_l',), ('character', 'x')),
        ('f', (), ('character', 'f')),
        (' Ctrl + ALT ', ('ctrl_l', 'alt_l'), None),
        ('ctrl+ctrl+d', ('ctrl_l',), ('character', 'd')),
        ('ctrl+f5', ('ctrl_l',), 'f5'),
    ])
    def test_shortcut_preserves_requested_keys(self, keyboard, combo, modifiers, trigger):
        from jarvis.dictation.dictation_engine import parse_hotkey

        expected_modifiers = frozenset(getattr(keyboard.Key, key) for key in modifiers)
        expected_trigger = getattr(keyboard.Key, trigger) if isinstance(trigger, str) else trigger
        assert parse_hotkey(combo) == (expected_modifiers, expected_trigger)

    @pytest.mark.parametrize('combo', ['', 'ctrl+nonexistentkey'])
    def test_invalid_shortcut_is_rejected(self, keyboard, combo):
        from jarvis.dictation.dictation_engine import parse_hotkey
        with pytest.raises(ValueError):
            parse_hotkey(combo)

    def test_missing_keyboard_backend_is_reported(self, monkeypatch):
        from jarvis.dictation import dictation_engine
        monkeypatch.setattr(dictation_engine, 'pynput_keyboard', None)
        with pytest.raises(RuntimeError):
            dictation_engine.parse_hotkey('ctrl+alt')


# ---------------------------------------------------------------------------
# Recording state machine
# ---------------------------------------------------------------------------

class TestRecordingStateMachine:
    """Tests for the recording start/stop logic."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_deps(self):
        try:
            import pynput  # noqa: F401
            import sounddevice  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            pytest.skip("required dependencies not installed")

    def test_start_recording_checks_whisper_model(self):
        """Should not start recording if Whisper model is None (non-mlx)."""
        engine = _make_engine(whisper_model_ref=lambda: None)
        engine._start_recording()
        assert engine._recording is False

    def test_start_recording_allows_mlx_without_model(self):
        """MLX backend uses repo reference, not model object."""
        engine = _make_engine(
            whisper_model_ref=lambda: None,
            whisper_backend_ref=lambda: "mlx",
            mlx_repo_ref=lambda: "mlx-community/whisper-small-mlx",
        )
        with patch("jarvis.dictation.dictation_engine.sd") as mock_sd, \
             patch("jarvis.dictation.dictation_engine._play_beep"):
            mock_stream = MagicMock()
            mock_sd.InputStream.return_value = mock_stream
            engine._start_recording()
            assert engine._recording is True
            # Cleanup
            engine._stop_recording(discard=True)

    def test_named_stereo_headset_records_both_channels(self):
        """Dictation uses the selected headset even when mono capture is rejected."""
        import numpy as np

        engine = _make_engine(voice_device='Headset')
        engine._recording = True
        engine._session = 1

        with patch('jarvis.dictation.dictation_engine.sd') as mock_sd, \
             patch('jarvis.dictation.dictation_engine._play_beep'):
            devices = [
                {'name': None, 'max_input_channels': 1},
                {'name': 'Headset microphone', 'max_input_channels': 2,
                 'default_samplerate': 48000},
            ]
            mock_sd.query_devices.side_effect = lambda device=None, **kwargs: (
                devices if device is None and not kwargs else devices[device]
            )
            stream = MagicMock()

            def open_stream(**kwargs):
                assert kwargs['device'] == 1
                if kwargs['channels'] == 1:
                    raise RuntimeError('Invalid number of channels', -9998)
                assert kwargs['samplerate'] == 48000
                return stream

            mock_sd.InputStream.side_effect = open_stream
            engine._begin_recording(1)
            assert engine._stream is stream
            assert engine._stream_sample_rate == 48000

            audio = np.zeros((48000, 2), dtype=np.float32)
            audio[:, 1] = 0.4
            engine._audio_callback(engine._session, audio, len(audio), None, None)
            np.testing.assert_allclose(engine._audio_frames[-1], 0.2)
            with patch.object(engine, '_transcribe', return_value='') as transcribe:
                engine._transcribe_and_paste(engine._audio_frames, engine._stream_sample_rate)
            whisper_audio = transcribe.call_args.args[0]
            assert len(whisper_audio) == 16000
            np.testing.assert_allclose(whisper_audio, 0.2)
            engine._stop_recording(discard=True)

    def test_missing_named_microphone_does_not_use_default_for_dictation(self, capsys):
        ended = threading.Event()
        engine = _make_engine(voice_device='Disconnected Headset', on_dictation_end=ended.set)
        engine._recording = True
        engine._session = 1

        with patch('jarvis.dictation.dictation_engine.sd') as mock_sd, \
             patch('jarvis.dictation.dictation_engine._play_beep'):
            mock_sd.query_devices.return_value = [
                {'name': 'Built-in Microphone', 'max_input_channels': 1},
            ]
            engine._begin_recording(1)
            mock_sd.InputStream.assert_not_called()

        assert not engine._recording
        assert ended.is_set()
        assert 'Selected microphone not found' in capsys.readouterr().out

    def test_dictation_uses_whisper_rate_if_native_rate_is_rejected(self):
        engine = _make_engine(voice_device='1')
        engine._recording = True
        engine._session = 1

        with patch('jarvis.dictation.dictation_engine.sd') as mock_sd, \
             patch('jarvis.dictation.dictation_engine._play_beep'):
            mock_sd.query_devices.return_value = {
                'max_input_channels': 2, 'default_samplerate': 48000,
            }

            def open_stream(**kwargs):
                if kwargs['samplerate'] == 48000:
                    raise RuntimeError('Invalid sample rate', -9997)
                return MagicMock()

            mock_sd.InputStream.side_effect = open_stream
            engine._begin_recording(1)

            assert engine._stream is not None
            assert engine._stream_sample_rate == 16000
            engine._stop_recording(discard=True)

    def test_stop_recording_discard_clears_frames(self):
        engine = _make_engine()
        engine._recording = True
        engine._audio_frames = [MagicMock()]
        engine._stream = MagicMock()
        engine._stop_recording(discard=True)
        assert engine._audio_frames == []
        assert engine._recording is False

    def test_callback_accumulates_indefinitely_without_cap(self):
        """Without a max duration cap, the audio callback must continue
        accumulating frames regardless of duration."""
        import numpy as np
        end_called = threading.Event()
        engine = _make_engine(
            on_dictation_end=end_called.set,
            whisper_model_ref=lambda: None,
            whisper_backend_ref=lambda: "faster-whisper",
        )
        stream_mock = MagicMock()
        engine._recording = True
        engine._stream = stream_mock
        engine._audio_frames = []

        with patch("jarvis.dictation.dictation_engine._play_beep"):
            # Send many frames — well past a hypothetical 60s cap
            for _ in range(100):
                indata = np.random.randn(1600, 1).astype(np.float32)
                engine._audio_callback(engine._session, indata, 1600, None, None)

        # All frames accumulated; no stop was triggered
        assert len(engine._audio_frames) == 100
        assert engine._recording is True
        assert not end_called.is_set()

        # Cleanup
        engine._stop_recording(discard=True)

    def test_finalise_fires_on_dictation_end_when_beep_raises(self):
        """A failure in ``_play_beep`` must not strand the listener paused.

        Finalisation guarantees completion notification even if stream
        teardown or the stop beep fails before transcription.
        """
        import numpy as np
        end_called = threading.Event()
        engine = _make_engine(on_dictation_end=lambda: end_called.set())

        with patch(
            "jarvis.dictation.dictation_engine._play_beep",
            side_effect=RuntimeError("beep broken"),
        ):
            engine._finalise_and_transcribe(
                stream=None,
                audio_frames=[np.zeros(1600, dtype=np.float32)],
                start_time=time.monotonic(), stream_sample_rate=engine._stream_sample_rate, token=engine._session,
            )

        assert end_called.is_set(), (
            "_on_dictation_end must fire even when _play_beep raises"
        )

    def test_on_dictation_callbacks_called(self):
        """Start/end callbacks should be invoked."""
        start_called = threading.Event()
        end_called = threading.Event()

        engine = _make_engine(
            on_dictation_start=lambda: start_called.set(),
            on_dictation_end=lambda: end_called.set(),
        )

        with patch("jarvis.dictation.dictation_engine.sd") as mock_sd, \
             patch("jarvis.dictation.dictation_engine._play_beep"):
            mock_stream = MagicMock()
            mock_sd.InputStream.return_value = mock_stream
            engine._start_recording()
            assert start_called.is_set()

            engine._stop_recording(discard=True)
            assert end_called.is_set()


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

class TestTranscription:
    """Tests for the transcription logic."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_deps(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            pytest.skip("numpy not installed")

    def test_transcribe_faster_whisper(self):
        import numpy as np
        mock_model = MagicMock()
        mock_seg = MagicMock()
        mock_seg.text = " hello world "
        mock_model.transcribe.return_value = ([mock_seg], MagicMock())

        engine = _make_engine(
            whisper_model_ref=lambda: mock_model,
            whisper_backend_ref=lambda: "faster-whisper",
        )

        audio = np.zeros(16000, dtype=np.float32)
        result = engine._transcribe(audio)
        assert result == "hello world"

    def test_transcribe_empty_returns_empty(self):
        import numpy as np
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())

        engine = _make_engine(
            whisper_model_ref=lambda: mock_model,
            whisper_backend_ref=lambda: "faster-whisper",
        )

        audio = np.zeros(16000, dtype=np.float32)
        result = engine._transcribe(audio)
        assert result == ""

    def test_transcribe_no_model_returns_empty(self):
        import numpy as np
        engine = _make_engine(
            whisper_model_ref=lambda: None,
            whisper_backend_ref=lambda: "faster-whisper",
        )

        audio = np.zeros(16000, dtype=np.float32)
        result = engine._transcribe(audio)
        assert result == ""

    def test_transcribe_mlx(self):
        import sys
        import numpy as np
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {"text": "hello from mlx"}

        # Patch sys.modules so `import mlx_whisper` inside the method resolves
        with patch.dict(sys.modules, {"mlx_whisper": mock_mlx}):
            engine = _make_engine(
                whisper_model_ref=lambda: None,
                whisper_backend_ref=lambda: "mlx",
                mlx_repo_ref=lambda: "mlx-community/whisper-small-mlx",
            )

            audio = np.zeros(16000, dtype=np.float32)
            result = engine._transcribe(audio)
            assert result == "hello from mlx"


# ---------------------------------------------------------------------------
# Clipboard helpers
# ---------------------------------------------------------------------------

class TestClipboard:
    """Tests for clipboard/paste helper functions."""

    @patch("jarvis.dictation.dictation_engine.platform")
    @patch("jarvis.dictation.dictation_engine._clipboard_windows")
    @patch("jarvis.dictation.dictation_engine.pynput_keyboard")
    def test_clipboard_paste_windows(self, mock_kb, mock_clip_win, mock_platform):
        from jarvis.dictation.dictation_engine import _clipboard_paste
        mock_platform.system.return_value = "Windows"
        mock_ctrl = MagicMock()
        mock_kb.Controller.return_value = mock_ctrl
        mock_kb.Key.ctrl = MagicMock()

        _clipboard_paste("hello")
        mock_clip_win.assert_called_once_with("hello")

    @patch("jarvis.dictation.dictation_engine._paste_cgevent", return_value=True)
    @patch("jarvis.dictation.dictation_engine._check_macos_accessibility", return_value=True)
    @patch("jarvis.dictation.dictation_engine.platform")
    @patch("jarvis.dictation.dictation_engine._clipboard_macos")
    @patch("jarvis.dictation.dictation_engine.pynput_keyboard")
    def test_clipboard_paste_macos(
        self, mock_kb, mock_clip_mac, mock_platform, mock_ax, mock_cge
    ):
        from jarvis.dictation.dictation_engine import _clipboard_paste
        mock_platform.system.return_value = "Darwin"
        mock_ctrl = MagicMock()
        mock_kb.Controller.return_value = mock_ctrl
        mock_kb.Key.cmd = MagicMock()

        _clipboard_paste("hello mac")
        mock_clip_mac.assert_called_once_with("hello mac")
        # Guard: the real CGEvent paste and Accessibility check must never
        # fire during tests — they would emit a real Cmd+V into whatever
        # window has focus and pop open System Settings.
        mock_cge.assert_called_once()

    def test_clipboard_paste_empty_string_is_noop(self):
        from jarvis.dictation.dictation_engine import _clipboard_paste
        # Should return immediately without error
        _clipboard_paste("")
        _clipboard_paste(None)


# ---------------------------------------------------------------------------
# Transcribe-and-paste pipeline
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.usefixtures('no_network')
class TestTranscribeAndPaste:
    """Released dictation completes without losing or inventing output."""

    @pytest.mark.parametrize('duration,text,expected', [
        (0, 'Synthetic dictated sentence.', None),
        (0.1, 'Synthetic dictated sentence.', None),
        (0.5, '', None),
        (0.5, None, None),
        (0.5, 'Synthetic dictated sentence.', 'Synthetic dictated sentence.'),
    ], ids=['empty-audio', 'short-audio', 'empty-transcription', 'failed-transcription', 'valid'])
    def test_finalisation_delivery_and_persisted_history(self, monkeypatch, tmp_path, duration, text, expected):
        from types import SimpleNamespace
        import numpy as np
        from jarvis.dictation import dictation_engine
        from jarvis.dictation.history import DictationHistory

        model = MagicMock()
        if text is None:
            model.transcribe.side_effect = RuntimeError('Synthetic transcription failure')
        else:
            model.transcribe.return_value = ([SimpleNamespace(text=text)], SimpleNamespace())
        path = tmp_path / 'history.json'
        pasted, delivered, phases = [], [], []
        monkeypatch.setattr(dictation_engine, 'pynput_keyboard', None)
        monkeypatch.setattr(dictation_engine, 'parse_hotkey', lambda _: (frozenset(), None))
        monkeypatch.setattr(dictation_engine, '_clipboard_paste', pasted.append)
        monkeypatch.setattr(dictation_engine, '_play_beep', lambda *args: None)
        engine = _make_engine(
            whisper_model_ref=lambda: model, history=DictationHistory(path=path),
            on_dictation_result=delivered.append,
            on_dictation_processing_start=lambda: phases.append('processing'),
            on_dictation_end=lambda: phases.append('idle'),
        )
        frames = [np.zeros(int(engine._target_sample_rate * duration), dtype=np.float32)] if duration else []
        engine._finalise_and_transcribe(None, frames, time.monotonic() - duration,
                                       engine._target_sample_rate, engine._session)

        expected_texts = [] if expected is None else [expected]
        assert phases == ['processing', 'idle']
        assert pasted == expected_texts
        assert [entry['text'] for entry in delivered] == expected_texts
        assert [entry['duration'] for entry in delivered] == ([] if expected is None else [duration])
        assert DictationHistory(path=path).get_all() == delivered


# ---------------------------------------------------------------------------
# Paste-activation suppression
# ---------------------------------------------------------------------------

class TestPasteActivationSuppression:
    """Tests that synthetic key events during paste don't restart recording."""

    def _make_keys(self):
        """Get real pynput key objects for ctrl, shift, and d."""
        from pynput import keyboard
        import importlib
        import jarvis.dictation.dictation_engine as de
        importlib.reload(de)  # ensures _MODIFIER_MAP is populated
        return (
            keyboard.Key.ctrl_l,
            keyboard.Key.shift,
            keyboard.KeyCode.from_char("d"),
        )

    def test_paste_in_progress_suppresses_restart(self):
        """Synthetic key events during paste must not trigger new recording."""
        from pynput import keyboard as pynput_kb
        from jarvis.dictation.dictation_engine import DictationEngine

        engine = _make_engine(hotkey="ctrl+shift+d")
        ctrl, shift, d = self._make_keys()

        # Simulate post-stop state — not recording, paste in progress
        engine._recording = False
        engine._paste_in_progress = True
        engine._pressed_modifiers.clear()

        with patch.object(engine, "_start_recording") as mock_start:
            # Paste thread first presses modifier keys (Ctrl, then Shift)
            engine._on_key_press(ctrl)
            engine._on_key_press(shift)
            # Paste thread then taps 'd' (Ctrl+V)
            engine._on_key_press(d)
            # Must NOT start a new recording during paste
            mock_start.assert_not_called()

        # Now simulate paste completed — clear flag and release synthetic keys
        engine._paste_in_progress = False
        engine._on_key_release(d)
        engine._on_key_release(shift)

        # User is still holding the physical keys — simulate key repeat events
        # that the OS might generate for held modifiers
        engine._on_key_press(ctrl)
        engine._on_key_press(shift)

        with patch.object(engine, "_start_recording") as mock_start:
            engine._on_key_press(d)
            mock_start.assert_called_once()

        engine.stop()

    def test_paste_in_progress_does_not_block_normal_release_stop(self):
        """Normal release of hotkey must still stop recording during paste."""
        engine = _make_engine(hotkey="ctrl+shift+d")
        ctrl, shift, d = self._make_keys()

        # Simulate recording in progress
        engine._recording = True
        engine._paste_in_progress = False
        engine._pressed_modifiers = {ctrl, shift}

        with patch.object(engine, "_stop_recording") as mock_stop:
            # User releases a modifier
            engine._on_key_release(ctrl)
            mock_stop.assert_called_once()

        engine.stop()


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Tests that dictation config fields are present in Settings."""

    def test_settings_has_dictation_fields(self):
        from jarvis.config import Settings
        import inspect
        sig = inspect.signature(Settings)
        assert "dictation_enabled" in sig.parameters
        assert "dictation_hotkey" in sig.parameters

    def test_default_config_has_dictation(self):
        import sys
        from jarvis.config import get_default_config
        defaults = get_default_config()
        assert defaults["dictation_enabled"] is True
        # Platform-aware default (aligned with WisprFlow)
        if sys.platform == "win32":
            assert defaults["dictation_hotkey"] == "ctrl+cmd"
        else:
            assert defaults["dictation_hotkey"] == "ctrl+alt"

    def test_load_settings_includes_dictation(self):
        """load_settings should produce Settings with dictation fields."""
        from jarvis.config import load_settings
        settings = load_settings()
        assert hasattr(settings, "dictation_enabled")
        assert hasattr(settings, "dictation_hotkey")
        assert isinstance(settings.dictation_enabled, bool)
        assert isinstance(settings.dictation_hotkey, str)


# ---------------------------------------------------------------------------
# Face widget DICTATING state
# ---------------------------------------------------------------------------

class TestFaceWidgetDictatingState:
    """Tests that the DICTATING state exists and is handled."""

    def test_jarvis_state_has_dictating(self):
        from desktop_app.face_widget import JarvisState
        assert hasattr(JarvisState, "DICTATING")
        assert JarvisState.DICTATING.value == "dictating"

    def test_dictating_state_round_trips(self):
        """State manager should accept DICTATING state."""
        from desktop_app.face_widget import JarvisState
        state = JarvisState("dictating")
        assert state == JarvisState.DICTATING

    def test_jarvis_state_has_dictation_processing(self):
        from desktop_app.face_widget import JarvisState
        assert hasattr(JarvisState, "DICTATION_PROCESSING")
        assert JarvisState.DICTATION_PROCESSING.value == "dictation_processing"

    def test_dictation_processing_state_round_trips(self):
        from desktop_app.face_widget import JarvisState
        state = JarvisState("dictation_processing")
        assert state == JarvisState.DICTATION_PROCESSING


class TestDictationProcessingCallback:
    """Verifies the processing callback fires between recording stop and
    transcription, so the face can switch to a distinct 'processing' state
    once the user's voice input has been accepted."""

    def test_processing_callback_fires_before_end_callback(self):
        """End-to-end ordering: the processing callback must fire before the
        end callback during the full finalise → transcribe → paste chain."""
        from jarvis.dictation import dictation_engine as de

        events = []

        engine = _make_engine(
            on_dictation_processing_start=lambda: events.append("processing"),
            on_dictation_end=lambda: events.append("end"),
        )

        # Stub stream teardown and beep audio only. The real
        # _transcribe_and_paste runs with empty frames; finalisation still
        # reports processing and completion for the released session.
        with patch.object(de, "_close_stream"), patch.object(de, "_play_beep"):
            engine._finalise_and_transcribe(
                stream=MagicMock(), audio_frames=[], start_time=time.monotonic(), stream_sample_rate=engine._stream_sample_rate, token=engine._session
            )

        assert events == ["processing", "end"]

    def test_processing_callback_optional(self):
        """Engine must work when no processing callback is supplied."""
        from jarvis.dictation import dictation_engine as de

        engine = _make_engine(on_dictation_processing_start=None)

        with patch.object(de, "_close_stream"), \
             patch.object(de, "_play_beep"), \
             patch.object(engine, "_transcribe_and_paste"):
            # Should not raise
            engine._finalise_and_transcribe(stream=MagicMock(), audio_frames=[], start_time=time.monotonic(), stream_sample_rate=engine._stream_sample_rate, token=engine._session)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Tests for thread-safe transcription locking."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_numpy(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            pytest.skip("numpy not installed")

    def test_transcribe_acquires_lock(self):
        """Transcription should acquire the shared lock."""
        import numpy as np
        lock = threading.Lock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())

        engine = _make_engine(
            whisper_model_ref=lambda: mock_model,
            whisper_backend_ref=lambda: "faster-whisper",
            transcribe_lock=lock,
        )

        # Acquire the lock externally — transcribe should block
        lock.acquire()
        result_holder = [None]
        done = threading.Event()

        def do_transcribe():
            result_holder[0] = engine._transcribe(np.zeros(16000, dtype=np.float32))
            done.set()

        t = threading.Thread(target=do_transcribe)
        t.start()

        # Give thread a moment — it should be blocked
        time.sleep(0.1)
        assert not done.is_set()

        # Release the lock — thread should complete
        lock.release()
        done.wait(timeout=2.0)
        assert done.is_set()
        assert result_holder[0] == ""
        t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Listener capture pause
# ---------------------------------------------------------------------------

class TestListenerPauseFlag:
    """Tests for the independent dictation capture pause on VoiceListener."""

    @pytest.fixture()
    def listener(self):
        """Create a VoiceListener with mock dependencies."""
        from jarvis.listening.listener import VoiceListener
        cfg = MagicMock()
        cfg.sample_rate = 16000
        cfg.vad_enabled = False
        cfg.wake_aliases = []
        cfg.stop_commands = ["stop"]
        return VoiceListener(MagicMock(), cfg, MagicMock(), MagicMock())

    def test_voice_listener_starts_with_capture_unpaused(self, listener):
        """Voice capture is available before any consumer pauses it."""
        assert listener.capture_paused is False

    def test_voice_listener_has_transcribe_lock(self, listener):
        """VoiceListener should expose a transcribe_lock."""
        assert hasattr(listener, "transcribe_lock")
        assert isinstance(listener.transcribe_lock, type(threading.Lock()))


# ---------------------------------------------------------------------------
# format_hotkey_display
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.usefixtures('no_network')
class TestFormatHotkeyDisplay:
    """Configured shortcuts use familiar platform-specific modifier names."""

    @pytest.mark.parametrize('system,combo,expected', [
        ('Windows', 'ctrl+cmd', 'Ctrl + Win'),
        ('Windows', 'ctrl+super', 'Ctrl + Win'),
        ('Windows', 'ctrl+win', 'Ctrl + Win'),
        ('Darwin', 'ctrl+cmd', 'Ctrl + Cmd'),
        ('Darwin', 'ctrl+alt', 'Ctrl + Option'),
        ('Windows', 'ctrl+shift+d', 'Ctrl + Shift + D'),
        ('Linux', 'ctrl+alt', 'Ctrl + Alt'),
    ])
    def test_shortcut_uses_platform_modifier_names(self, monkeypatch, system, combo, expected):
        from jarvis.dictation import dictation_engine
        monkeypatch.setattr(dictation_engine.platform, 'system', lambda: system)
        assert dictation_engine.format_hotkey_display(combo) == expected


# ---------------------------------------------------------------------------
# _clipboard_windows ctypes correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.usefixtures('no_network')
class TestClipboardWindowsCtypes:
    """Windows clipboard delivery and ownership use an owned native ABI."""

    @pytest.fixture
    def clipboard(self, monkeypatch):
        import ctypes
        from types import SimpleNamespace

        state = {'failure': None, 'opened': False, 'allocated': set(),
                 'published': None, 'buffer': None}
        handle = (1 << 40) + 17
        callbacks = []

        def function(result_type, argument_types, callback):
            signature = ctypes.CFUNCTYPE(result_type, *argument_types)
            native = signature(callback)
            callbacks.append(native)
            # Unconfigured foreign functions return a C int and infer arguments.
            bound = ctypes.CFUNCTYPE(ctypes.c_int)(ctypes.cast(native, ctypes.c_void_p).value)
            return bound

        def allocate(flags, size):
            if state['failure'] == 'allocate':
                return 0
            state['buffer'] = ctypes.create_string_buffer(size)
            state['allocated'].add(handle)
            return handle

        def lock(value):
            if value not in state['allocated'] or state['failure'] == 'lock':
                return 0
            return ctypes.addressof(state['buffer'])

        def free(value):
            state['allocated'].discard(value)
            return 0

        def open_clipboard(owner):
            state['opened'] = state['failure'] != 'open'
            return int(state['opened'])

        def publish(kind, value):
            if kind != 13 or value not in state['allocated'] or state['failure'] == 'publish':
                return 0
            state['published'] = state['buffer'].raw
            state['allocated'].remove(value)
            return value

        def close_clipboard():
            state['opened'] = False
            return 1

        user = SimpleNamespace(
            OpenClipboard=function(ctypes.c_int, [ctypes.c_void_p], open_clipboard),
            CloseClipboard=function(ctypes.c_int, [], close_clipboard),
            EmptyClipboard=function(ctypes.c_int, [], lambda: 1),
            SetClipboardData=function(ctypes.c_void_p, [ctypes.c_uint, ctypes.c_void_p], publish),
        )
        kernel = SimpleNamespace(
            GlobalAlloc=function(ctypes.c_void_p, [ctypes.c_uint, ctypes.c_size_t], allocate),
            GlobalLock=function(ctypes.c_void_p, [ctypes.c_void_p], lock),
            GlobalUnlock=function(ctypes.c_int, [ctypes.c_void_p], lambda value: 1),
            GlobalFree=function(ctypes.c_void_p, [ctypes.c_void_p], free),
        )
        monkeypatch.setattr(ctypes, 'windll', SimpleNamespace(user32=user, kernel32=kernel), raising=False)
        copy = ctypes.memmove

        def owned_copy(destination, source, size):
            assert destination == ctypes.addressof(state['buffer']), 'Unowned or truncated destination'
            assert size <= ctypes.sizeof(state['buffer'])
            return copy(destination, source, size)

        monkeypatch.setattr(ctypes, 'memmove', owned_copy)
        yield state
        assert not state['opened'], 'Clipboard ownership must be released'
        assert not state['allocated'], 'Unpublished memory must be released'

    @pytest.mark.parametrize('text', ['dictation test', 'Dictée, Türkçe, 日本語 🎙️', ''])
    def test_unicode_text_reaches_the_clipboard_with_full_width_handles(self, clipboard, text):
        from jarvis.dictation.dictation_engine import _clipboard_windows

        _clipboard_windows(text)

        assert clipboard['published'] == text.encode('utf-16-le') + b'\x00\x00'

    @pytest.mark.parametrize('failure', ['open', 'allocate', 'lock', 'publish'])
    def test_failed_delivery_preserves_ownership_and_publishes_nothing(self, clipboard, failure):
        from jarvis.dictation.dictation_engine import _clipboard_windows

        clipboard['failure'] = failure
        with pytest.raises(OSError):
            _clipboard_windows('Synthetic dictated text')

        assert clipboard['published'] is None

@pytest.mark.unit
@pytest.mark.usefixtures('no_network')
class TestLlmCleanDictation:
    """Dictation delivers and stores complete cleanup or raw fallback text."""

    def assert_pipeline_result(self, monkeypatch, tmp_path, cfg, original, expected, thinking):
        from dataclasses import replace
        from types import SimpleNamespace
        import numpy as np
        from jarvis.dictation import dictation_engine
        from jarvis.dictation.history import DictationHistory

        cfg = replace(cfg, dictation_thinking_enabled=thinking)
        model = MagicMock()
        model.transcribe.return_value = ([SimpleNamespace(text=original)], SimpleNamespace())
        history_path = tmp_path / 'history.json'
        history = DictationHistory(path=history_path)
        pasted, delivered = [], []
        monkeypatch.setattr(dictation_engine, 'pynput_keyboard', None)
        monkeypatch.setattr(dictation_engine, 'parse_hotkey', lambda _: (frozenset(), None))
        monkeypatch.setattr(dictation_engine, '_clipboard_paste', pasted.append)
        engine = _make_engine(
            whisper_model_ref=lambda: model, history=history, cfg=cfg,
            filler_removal=True, chat_model=cfg.llm_chat_model,
            thinking=cfg.dictation_thinking_enabled, on_dictation_result=delivered.append,
        )
        frames = [np.zeros(int(engine._target_sample_rate * 0.5), dtype=np.float32)]
        engine._transcribe_and_paste(frames, engine._target_sample_rate)

        assert pasted == [expected]
        assert [entry['text'] for entry in delivered] == [expected]
        assert [entry['duration'] for entry in delivered] == [0.5]
        assert DictationHistory(path=history_path).get_all() == delivered

    @pytest.fixture(params=['ollama', 'openai_compatible'])
    def cfg(self, request):
        from dataclasses import replace
        from jarvis.config import load_settings

        return replace(load_settings(), llm_provider=request.param,
                       llm_base_url='http://127.0.0.1:1/v1', ollama_base_url='http://127.0.0.1:1',
                       llm_chat_model='synthetic-cleanup-model')

    @pytest.mark.parametrize('thinking', [False, True])
    @pytest.mark.parametrize('word_count', [2, 300])
    def test_complete_cleanup_uses_configured_transport_and_budget(self, monkeypatch, tmp_path, cfg, thinking, word_count):
        from contextlib import nullcontext
        from jarvis.llm import http

        provider = cfg.llm_provider
        words = ['alpha'] * word_count
        cleaned = ' '.join(words)
        original = 'um ' + cleaned

        def post(url, **kwargs):
            payload = kwargs['json']
            assert payload['model'] == cfg.llm_chat_model
            assert payload['messages'][-1]['content'] == original
            if provider == 'ollama':
                assert url == cfg.ollama_base_url + '/api/chat'
                assert payload['think'] is thinking
                budget = payload['options']['num_predict']
            else:
                assert url == cfg.llm_base_url + '/chat/completions'
                assert 'think' not in payload
                budget = payload['max_tokens']
            # This synthetic model has a one-word vocabulary and a bounded budget.
            answer = ' '.join(words[:budget])
            reason = 'stop' if budget >= len(words) else 'length'
            message = {'content': answer}
            response = MagicMock()
            response.json.return_value = (
                {'message': message, 'done': True, 'done_reason': reason}
                if provider == 'ollama' else
                {'choices': [{'message': message, 'finish_reason': reason}]}
            )
            return nullcontext(response)

        monkeypatch.setattr(http, 'post', post)
        self.assert_pipeline_result(monkeypatch, tmp_path, cfg, original, cleaned, thinking)

    @pytest.mark.parametrize('failure', ['empty', 'unavailable'])
    def test_failed_cleanup_preserves_the_full_transcription(self, monkeypatch, tmp_path, cfg, failure):
        from contextlib import nullcontext
        import requests
        from jarvis.llm import http

        original = 'Keep every word and the ending of this transcription.'

        def post(*args, **kwargs):
            if failure == 'unavailable':
                raise requests.ConnectionError('Synthetic unavailable model')
            response = MagicMock()
            response.json.return_value = (
                {'message': {'content': ''}} if cfg.llm_provider == 'ollama' else
                {'choices': [{'message': {'content': ''}}]}
            )
            return nullcontext(response)

        monkeypatch.setattr(http, 'post', post)
        self.assert_pipeline_result(monkeypatch, tmp_path, cfg, original, original,
                                    cfg.dictation_thinking_enabled)


@pytest.mark.unit
def test_unavailable_default_ends_dictation_without_recording(capsys):
    ended = threading.Event()
    with patch('jarvis.dictation.dictation_engine.parse_hotkey',
               return_value=(frozenset(), None)):
        engine = _make_engine(on_dictation_end=ended.set)
    engine._recording = True
    engine._session = 1
    with patch('jarvis.dictation.dictation_engine.sd') as audio, \
         patch('jarvis.dictation.dictation_engine._play_beep'):
        audio.query_devices.side_effect = RuntimeError('Error querying device -1')
        engine._begin_recording(1)
        audio.InputStream.assert_not_called()
    assert not engine._recording
    assert ended.is_set()
    assert 'default microphone' in capsys.readouterr().out


@pytest.mark.unit
def test_dictation_keeps_default_input_during_native_rate_query():
    with patch('jarvis.dictation.dictation_engine.parse_hotkey',
               return_value=(frozenset(), None)):
        engine = _make_engine()
    engine._recording = True
    engine._session = 1
    selected = {'index': 3, 'name': 'Selected microphone', 'max_input_channels': 1,
                'default_samplerate': 48000}
    opened = []
    with patch('jarvis.dictation.dictation_engine.sd') as audio, \
         patch('jarvis.dictation.dictation_engine._play_beep'):
        def query(device=None, *, kind=None):
            if kind == 'input':
                return selected.copy()
            assert device == 3
            selected['index'] = 7
            return dict(selected, index=device)

        def open_stream(**kwargs):
            opened.append((kwargs['device'], kwargs['samplerate']))
            return MagicMock()

        audio.query_devices.side_effect = query
        audio.InputStream.side_effect = open_stream
        engine._begin_recording(1)
        assert engine._stream is not None
        assert opened == [(3, selected['default_samplerate'])]
        engine._stop_recording(discard=True)
