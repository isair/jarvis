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
    from src.jarvis.dictation.dictation_engine import DictationEngine

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

class TestBeepGeneration:
    """Tests for beep WAV generation."""

    def test_start_beep_is_valid_wav(self):
        from src.jarvis.dictation.dictation_engine import _get_start_beep
        wav = _get_start_beep()
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"

    def test_stop_beep_is_valid_wav(self):
        from src.jarvis.dictation.dictation_engine import _get_stop_beep
        wav = _get_stop_beep()
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"

    def test_start_and_stop_beeps_differ(self):
        from src.jarvis.dictation.dictation_engine import _get_start_beep, _get_stop_beep
        assert _get_start_beep() != _get_stop_beep()

    def test_generate_beep_wav_custom_params(self):
        from src.jarvis.dictation.dictation_engine import _generate_beep_wav
        wav = _generate_beep_wav(freq=1000, duration=0.05)
        assert wav[:4] == b"RIFF"
        assert len(wav) > 44  # At least a header


# ---------------------------------------------------------------------------
# Hotkey parsing
# ---------------------------------------------------------------------------

class TestHotkeyParsing:
    """Tests for hotkey string → pynput key object parsing."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_pynput(self):
        try:
            import pynput  # noqa: F401
        except ImportError:
            pytest.skip("pynput not installed")

    def test_parse_ctrl_shift_d(self):
        from src.jarvis.dictation.dictation_engine import parse_hotkey
        mods, trigger = parse_hotkey("ctrl+shift+d")
        assert len(mods) == 2
        assert trigger is not None

    def test_parse_modifier_only_combo(self):
        """A modifier-only hotkey like 'ctrl+cmd' should be valid."""
        from src.jarvis.dictation.dictation_engine import parse_hotkey
        mods, trigger = parse_hotkey("ctrl+cmd")
        assert len(mods) == 2
        assert trigger is None

    def test_parse_ctrl_alt(self):
        """macOS/Linux default: ctrl+alt should parse as two modifiers."""
        from src.jarvis.dictation.dictation_engine import parse_hotkey
        mods, trigger = parse_hotkey("ctrl+alt")
        assert len(mods) == 2
        assert trigger is None

    def test_parse_empty_string_raises(self):
        from src.jarvis.dictation.dictation_engine import parse_hotkey
        with pytest.raises(ValueError):
            parse_hotkey("")

    def test_parse_unknown_key_raises(self):
        from src.jarvis.dictation.dictation_engine import parse_hotkey
        with pytest.raises(ValueError):
            parse_hotkey("ctrl+nonexistentkey")

    def test_parse_alt_modifier(self):
        from src.jarvis.dictation.dictation_engine import parse_hotkey
        mods, trigger = parse_hotkey("alt+x")
        assert len(mods) == 1
        assert trigger is not None

    def test_parse_single_letter(self):
        """A single letter without modifiers should work as trigger."""
        from src.jarvis.dictation.dictation_engine import parse_hotkey
        # Technically no modifiers, just a trigger
        mods, trigger = parse_hotkey("f")
        assert len(mods) == 0
        assert trigger is not None


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------

class TestEngineLifecycle:
    """Tests for DictationEngine start/stop behaviour."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_deps(self):
        try:
            import pynput  # noqa: F401
            import sounddevice  # noqa: F401
        except ImportError:
            pytest.skip("pynput or sounddevice not installed")

    @patch("src.jarvis.dictation.dictation_engine.platform")
    @patch("src.jarvis.dictation.dictation_engine.sys")
    @patch("src.jarvis.dictation.dictation_engine.pynput_keyboard")
    def test_start_creates_listener(self, mock_kb, mock_sys, mock_platform):
        # Force a platform where pynput is allowed (avoid macOS 26+ guard)
        mock_sys.platform = "linux"
        mock_listener_instance = MagicMock()
        mock_kb.Listener.return_value = mock_listener_instance
        mock_kb.Key = MagicMock()
        mock_kb.KeyCode = MagicMock()
        mock_kb.Key.ctrl_l = MagicMock()
        mock_kb.Key.shift = MagicMock()

        engine = _make_engine()
        engine.start()

        assert engine._started is True
        mock_listener_instance.start.assert_called_once()

        engine.stop()
        assert engine._started is False

    @patch("src.jarvis.dictation.dictation_engine.pynput_keyboard", None)
    def test_start_without_pynput_is_noop(self):
        """Engine should gracefully skip when pynput is missing."""
        from src.jarvis.dictation.dictation_engine import DictationEngine
        # We can't use _make_engine because parse_hotkey needs pynput.
        # Directly test the start() guard.
        engine = DictationEngine.__new__(DictationEngine)
        engine._started = False
        engine._listener = None
        engine._recording = False
        engine.start()
        assert engine._started is False

    @patch("src.jarvis.dictation.dictation_engine.sd", None)
    @patch("src.jarvis.dictation.dictation_engine.pynput_keyboard")
    def test_start_without_sounddevice_is_noop(self, mock_kb):
        """Engine should gracefully skip when sounddevice is missing."""
        mock_kb.Key = MagicMock()
        mock_kb.KeyCode = MagicMock()
        mock_kb.Key.ctrl_l = MagicMock()
        mock_kb.Key.shift = MagicMock()

        engine = _make_engine()
        engine.start()
        assert engine._started is False

    @patch("src.jarvis.dictation.dictation_engine.platform")
    @patch("src.jarvis.dictation.dictation_engine.sys")
    @patch("src.jarvis.dictation.dictation_engine.pynput_keyboard")
    def test_start_uses_subprocess_on_macos_26(self, mock_kb, mock_sys, mock_platform):
        """macOS 26+ should use subprocess helper instead of in-process pynput."""
        mock_sys.platform = "darwin"
        mock_sys.executable = "python3"
        mock_platform.mac_ver.return_value = ("26.2", ("", "", ""), "")
        mock_kb.Key = MagicMock()
        mock_kb.KeyCode = MagicMock()
        mock_kb.Key.ctrl_l = MagicMock()
        mock_kb.Key.shift = MagicMock()

        engine = _make_engine()

        # Mock the subprocess helper to succeed
        with patch.object(engine, "_start_subprocess_helper", return_value=True):
            engine.start()
            assert engine._started is True
            # In-process pynput listener must NOT be created
            mock_kb.Listener.assert_not_called()
            engine.stop()

    @patch("src.jarvis.dictation.dictation_engine.platform")
    @patch("src.jarvis.dictation.dictation_engine.sys")
    @patch("src.jarvis.dictation.dictation_engine.pynput_keyboard")
    def test_start_fails_gracefully_on_macos_26_if_helper_fails(
        self, mock_kb, mock_sys, mock_platform,
    ):
        """If the subprocess helper cannot start, dictation should be disabled."""
        mock_sys.platform = "darwin"
        mock_sys.executable = "python3"
        mock_platform.mac_ver.return_value = ("26.0", ("", "", ""), "")
        mock_kb.Key = MagicMock()
        mock_kb.KeyCode = MagicMock()
        mock_kb.Key.ctrl_l = MagicMock()
        mock_kb.Key.shift = MagicMock()

        engine = _make_engine()

        with patch.object(engine, "_start_subprocess_helper", return_value=False):
            engine.start()
            assert engine._started is False
            mock_kb.Listener.assert_not_called()

    @patch("src.jarvis.dictation.dictation_engine.platform")
    @patch("src.jarvis.dictation.dictation_engine.sys")
    @patch("src.jarvis.dictation.dictation_engine.pynput_keyboard")
    def test_start_allowed_on_macos_15(self, mock_kb, mock_sys, mock_platform):
        """pynput should still work on macOS 15 (Sequoia) and earlier."""
        mock_sys.platform = "darwin"
        mock_platform.mac_ver.return_value = ("15.4", ("", "", ""), "")
        mock_listener = MagicMock()
        mock_kb.Listener.return_value = mock_listener
        mock_kb.Key = MagicMock()
        mock_kb.KeyCode = MagicMock()
        mock_kb.Key.ctrl_l = MagicMock()
        mock_kb.Key.shift = MagicMock()

        engine = _make_engine()
        engine.start()
        assert engine._started is True
        mock_listener.start.assert_called_once()
        engine.stop()

    @patch("src.jarvis.dictation.dictation_engine.platform")
    @patch("src.jarvis.dictation.dictation_engine.sys")
    @patch("src.jarvis.dictation.dictation_engine.pynput_keyboard")
    def test_start_allowed_on_windows(self, mock_kb, mock_sys, mock_platform):
        """Windows should not be affected by the macOS guard."""
        mock_sys.platform = "win32"
        mock_listener = MagicMock()
        mock_kb.Listener.return_value = mock_listener
        mock_kb.Key = MagicMock()
        mock_kb.KeyCode = MagicMock()
        mock_kb.Key.ctrl_l = MagicMock()
        mock_kb.Key.shift = MagicMock()

        engine = _make_engine()
        engine.start()
        assert engine._started is True
        mock_listener.start.assert_called_once()
        engine.stop()


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
        with patch("src.jarvis.dictation.dictation_engine.sd") as mock_sd, \
             patch("src.jarvis.dictation.dictation_engine._play_beep"):
            mock_stream = MagicMock()
            mock_sd.InputStream.return_value = mock_stream
            engine._start_recording()
            assert engine._recording is True
            # Cleanup
            engine._stop_recording(discard=True)

    def test_stop_recording_discard_clears_frames(self):
        engine = _make_engine()
        engine._recording = True
        engine._audio_frames = [MagicMock()]
        engine._stream = MagicMock()
        engine._stop_recording(discard=True)
        assert engine._audio_frames == []
        assert engine._recording is False

    def test_on_dictation_callbacks_called(self):
        """Start/end callbacks should be invoked."""
        start_called = threading.Event()
        end_called = threading.Event()

        engine = _make_engine(
            on_dictation_start=lambda: start_called.set(),
            on_dictation_end=lambda: end_called.set(),
        )

        with patch("src.jarvis.dictation.dictation_engine.sd") as mock_sd, \
             patch("src.jarvis.dictation.dictation_engine._play_beep"):
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

    @patch("src.jarvis.dictation.dictation_engine.platform")
    @patch("src.jarvis.dictation.dictation_engine._clipboard_windows")
    @patch("src.jarvis.dictation.dictation_engine.pynput_keyboard")
    def test_clipboard_paste_windows(self, mock_kb, mock_clip_win, mock_platform):
        from src.jarvis.dictation.dictation_engine import _clipboard_paste
        mock_platform.system.return_value = "Windows"
        mock_ctrl = MagicMock()
        mock_kb.Controller.return_value = mock_ctrl
        mock_kb.Key.ctrl = MagicMock()

        _clipboard_paste("hello")
        mock_clip_win.assert_called_once_with("hello")

    @patch("src.jarvis.dictation.dictation_engine.platform")
    @patch("src.jarvis.dictation.dictation_engine._clipboard_macos")
    @patch("src.jarvis.dictation.dictation_engine.pynput_keyboard")
    def test_clipboard_paste_macos(self, mock_kb, mock_clip_mac, mock_platform):
        from src.jarvis.dictation.dictation_engine import _clipboard_paste
        mock_platform.system.return_value = "Darwin"
        mock_ctrl = MagicMock()
        mock_kb.Controller.return_value = mock_ctrl
        mock_kb.Key.cmd = MagicMock()

        _clipboard_paste("hello mac")
        mock_clip_mac.assert_called_once_with("hello mac")

    def test_clipboard_paste_empty_string_is_noop(self):
        from src.jarvis.dictation.dictation_engine import _clipboard_paste
        # Should return immediately without error
        _clipboard_paste("")
        _clipboard_paste(None)


# ---------------------------------------------------------------------------
# Audio callback
# ---------------------------------------------------------------------------

class TestAudioCallback:
    """Tests for the audio callback frame accumulation."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_numpy(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            pytest.skip("numpy not installed")

    def test_callback_accumulates_frames(self):
        import numpy as np
        engine = _make_engine()
        engine._recording = True
        engine._audio_frames = []
        engine._max_frames = 1_000_000

        indata = np.random.randn(1600, 1).astype(np.float32)
        engine._audio_callback(indata, 1600, None, None)
        assert len(engine._audio_frames) == 1
        assert len(engine._audio_frames[0]) == 1600

    def test_callback_ignores_when_not_recording(self):
        import numpy as np
        engine = _make_engine()
        engine._recording = False
        engine._audio_frames = []

        indata = np.random.randn(1600, 1).astype(np.float32)
        engine._audio_callback(indata, 1600, None, None)
        assert len(engine._audio_frames) == 0

    def test_callback_respects_max_duration(self):
        import numpy as np
        engine = _make_engine()
        engine._recording = True
        # Pre-fill near the max
        engine._max_frames = 100
        engine._audio_frames = [np.zeros(100, dtype=np.float32)]

        indata = np.random.randn(1600, 1).astype(np.float32)
        with patch.object(engine, "_stop_recording"):
            engine._audio_callback(indata, 1600, None, None)
            # Should not accumulate more frames
            assert len(engine._audio_frames) == 1


# ---------------------------------------------------------------------------
# Transcribe-and-paste pipeline
# ---------------------------------------------------------------------------

class TestTranscribeAndPaste:
    """Tests for the full transcribe → paste pipeline."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_numpy(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            pytest.skip("numpy not installed")

    def test_short_audio_skipped(self):
        """Audio shorter than 0.3s should be skipped."""
        import numpy as np
        engine = _make_engine()
        end_called = threading.Event()
        engine._on_dictation_end = lambda: end_called.set()

        # 0.1s of audio at 16kHz = 1600 samples (< 4800 needed for 0.3s)
        short_frames = [np.zeros(1600, dtype=np.float32)]
        engine._transcribe_and_paste(short_frames)
        assert end_called.is_set()

    def test_empty_frames_handled(self):
        engine = _make_engine()
        end_called = threading.Event()
        engine._on_dictation_end = lambda: end_called.set()

        engine._transcribe_and_paste([])
        assert end_called.is_set()

    @patch("src.jarvis.dictation.dictation_engine._clipboard_paste")
    def test_successful_transcription_pastes(self, mock_paste):
        import numpy as np
        mock_model = MagicMock()
        mock_seg = MagicMock()
        mock_seg.text = "hello world"
        mock_model.transcribe.return_value = ([mock_seg], MagicMock())

        engine = _make_engine(
            whisper_model_ref=lambda: mock_model,
            whisper_backend_ref=lambda: "faster-whisper",
        )

        frames = [np.zeros(8000, dtype=np.float32)]  # 0.5s
        engine._transcribe_and_paste(frames)
        mock_paste.assert_called_once_with("hello world")

    @patch("src.jarvis.dictation.dictation_engine._clipboard_paste")
    def test_empty_transcription_does_not_paste(self, mock_paste):
        import numpy as np
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())

        engine = _make_engine(
            whisper_model_ref=lambda: mock_model,
            whisper_backend_ref=lambda: "faster-whisper",
        )

        frames = [np.zeros(8000, dtype=np.float32)]
        engine._transcribe_and_paste(frames)
        mock_paste.assert_not_called()


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Tests that dictation config fields are present in Settings."""

    def test_settings_has_dictation_fields(self):
        from src.jarvis.config import Settings
        import inspect
        sig = inspect.signature(Settings)
        assert "dictation_enabled" in sig.parameters
        assert "dictation_hotkey" in sig.parameters

    def test_default_config_has_dictation(self):
        import sys
        from src.jarvis.config import get_default_config
        defaults = get_default_config()
        assert defaults["dictation_enabled"] is True
        # Platform-aware default (aligned with WisprFlow)
        if sys.platform == "win32":
            assert defaults["dictation_hotkey"] == "ctrl+cmd"
        else:
            assert defaults["dictation_hotkey"] == "ctrl+alt"

    def test_load_settings_includes_dictation(self):
        """load_settings should produce Settings with dictation fields."""
        from src.jarvis.config import load_settings
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
        from src.desktop_app.face_widget import JarvisState
        assert hasattr(JarvisState, "DICTATING")
        assert JarvisState.DICTATING.value == "dictating"

    def test_dictating_state_round_trips(self):
        """State manager should accept DICTATING state."""
        from src.desktop_app.face_widget import JarvisState
        state = JarvisState("dictating")
        assert state == JarvisState.DICTATING


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
# Listener pause flag
# ---------------------------------------------------------------------------

class TestListenerPauseFlag:
    """Tests for the dictation pause flag on VoiceListener."""

    @pytest.fixture()
    def listener(self):
        """Create a VoiceListener with mock dependencies."""
        from src.jarvis.listening.listener import VoiceListener
        cfg = MagicMock()
        cfg.sample_rate = 16000
        cfg.vad_enabled = False
        cfg.wake_aliases = []
        cfg.stop_commands = ["stop"]
        return VoiceListener(MagicMock(), cfg, MagicMock(), MagicMock())

    def test_voice_listener_has_dictation_active_flag(self, listener):
        """VoiceListener should initialise _dictation_active = False."""
        assert hasattr(listener, "_dictation_active")
        assert listener._dictation_active is False

    def test_voice_listener_has_transcribe_lock(self, listener):
        """VoiceListener should expose a transcribe_lock."""
        assert hasattr(listener, "transcribe_lock")
        assert isinstance(listener.transcribe_lock, type(threading.Lock()))


# ---------------------------------------------------------------------------
# Subprocess helper event handling
# ---------------------------------------------------------------------------

class TestSubprocessHelperEvents:
    """Tests for _on_helper_event mapping to recording logic."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_deps(self):
        try:
            import pynput  # noqa: F401
            import sounddevice  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            pytest.skip("required dependencies not installed")

    def test_hotkey_press_starts_recording(self):
        engine = _make_engine()
        with patch.object(engine, "_start_recording") as mock_start:
            engine._on_helper_event("hotkey_press")
            mock_start.assert_called_once()

    def test_hotkey_release_stops_recording(self):
        engine = _make_engine()
        engine._recording = True
        engine._record_start_time = time.time() - 1.0  # held for 1s
        with patch.object(engine, "_stop_recording") as mock_stop:
            engine._on_helper_event("hotkey_release")
            mock_stop.assert_called_once()

    def test_hotkey_release_ignored_when_not_recording(self):
        engine = _make_engine()
        engine._recording = False
        with patch.object(engine, "_stop_recording") as mock_stop:
            engine._on_helper_event("hotkey_release")
            mock_stop.assert_not_called()

    def test_hotkey_release_ignored_in_hands_free(self):
        """In hands-free mode, key release should NOT stop recording."""
        engine = _make_engine()
        engine._recording = True
        engine._hands_free = True
        with patch.object(engine, "_stop_recording") as mock_stop:
            engine._on_helper_event("hotkey_release")
            mock_stop.assert_not_called()

    def test_escape_stops_hands_free(self):
        engine = _make_engine()
        engine._recording = True
        engine._hands_free = True
        with patch.object(engine, "_stop_recording") as mock_stop:
            engine._on_helper_event("escape")
            mock_stop.assert_called_once()

    def test_escape_ignored_when_not_hands_free(self):
        engine = _make_engine()
        engine._recording = True
        engine._hands_free = False
        with patch.object(engine, "_stop_recording") as mock_stop:
            engine._on_helper_event("escape")
            mock_stop.assert_not_called()

    def test_hotkey_press_stops_hands_free_recording(self):
        """Re-pressing hotkey in hands-free mode should stop recording."""
        engine = _make_engine()
        engine._recording = True
        engine._hands_free = True
        with patch.object(engine, "_stop_recording") as mock_stop:
            engine._on_helper_event("hotkey_press")
            mock_stop.assert_called_once()

    def test_double_tap_activates_hands_free(self):
        """Two quick taps should activate hands-free mode."""
        engine = _make_engine()

        # Simulate first quick tap release
        engine._recording = True
        engine._record_start_time = time.time() - 0.1  # held 0.1s < 0.4s
        engine._last_hotkey_release_time = 0.0
        with patch.object(engine, "_stop_recording"):
            engine._on_helper_event("hotkey_release")
        assert engine._last_hotkey_release_time > 0

        # Simulate second quick tap
        engine._recording = True
        engine._hands_free = False
        engine._record_start_time = time.time() - 0.1  # held 0.1s
        engine._on_helper_event("hotkey_release")
        assert engine._hands_free is True


# ---------------------------------------------------------------------------
# Subprocess helper module
# ---------------------------------------------------------------------------

class TestHotkeySharedModule:
    """Tests for the shared hotkey-matching utilities."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_pynput(self):
        try:
            import pynput  # noqa: F401
        except ImportError:
            pytest.skip("pynput not installed")

    def test_parse_hotkey_ctrl_alt(self):
        from pynput import keyboard as kb
        from src.jarvis.dictation._hotkey_shared import parse_hotkey
        mods, trigger = parse_hotkey(kb, "ctrl+alt")
        assert len(mods) == 2
        assert trigger is None

    def test_parse_hotkey_with_trigger(self):
        from pynput import keyboard as kb
        from src.jarvis.dictation._hotkey_shared import parse_hotkey
        mods, trigger = parse_hotkey(kb, "ctrl+shift+d")
        assert len(mods) == 2
        assert trigger is not None

    def test_parse_hotkey_empty_raises(self):
        from pynput import keyboard as kb
        from src.jarvis.dictation._hotkey_shared import parse_hotkey
        with pytest.raises(ValueError):
            parse_hotkey(kb, "")

    def test_key_matches_same_key(self):
        from pynput import keyboard as kb
        from src.jarvis.dictation._hotkey_shared import key_matches
        key = kb.Key.ctrl_l
        assert key_matches(kb, key, key, kb.Key.ctrl_l) is True

    def test_key_matches_none_target(self):
        from pynput import keyboard as kb
        from src.jarvis.dictation._hotkey_shared import key_matches
        assert key_matches(kb, kb.Key.ctrl_l, kb.Key.ctrl_l, None) is False

    def test_all_modifiers_held(self):
        from pynput import keyboard as kb
        from src.jarvis.dictation._hotkey_shared import all_modifiers_held
        modifiers = frozenset({kb.Key.ctrl_l, kb.Key.alt_l})
        pressed = {kb.Key.ctrl_l, kb.Key.alt_l}
        assert all_modifiers_held(pressed, modifiers) is True

    def test_all_modifiers_not_held(self):
        from pynput import keyboard as kb
        from src.jarvis.dictation._hotkey_shared import all_modifiers_held
        modifiers = frozenset({kb.Key.ctrl_l, kb.Key.alt_l})
        pressed = {kb.Key.ctrl_l}
        assert all_modifiers_held(pressed, modifiers) is False

    def test_needs_subprocess_helper_detects_version(self):
        """_needs_subprocess_helper should return True for macOS 26+."""
        from src.jarvis.dictation.dictation_engine import DictationEngine
        with patch("src.jarvis.dictation.dictation_engine.platform") as mock_platform:
            mock_platform.mac_ver.return_value = ("26.0", ("", "", ""), "")
            assert DictationEngine._needs_subprocess_helper() is True

    def test_needs_subprocess_helper_false_for_older(self):
        from src.jarvis.dictation.dictation_engine import DictationEngine
        with patch("src.jarvis.dictation.dictation_engine.platform") as mock_platform:
            mock_platform.mac_ver.return_value = ("15.4", ("", "", ""), "")
            assert DictationEngine._needs_subprocess_helper() is False
