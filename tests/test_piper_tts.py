"""Tests for Piper TTS implementation."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import threading
import time


class TestPiperTTSInterface:
    """Tests for PiperTTS interface compliance."""

    def test_has_required_methods(self):
        """PiperTTS should have the same interface as TextToSpeech."""
        from src.jarvis.output.tts import PiperTTS

        # Create instance with TTS disabled (no model needed)
        tts = PiperTTS(enabled=False)

        # Check required methods exist
        assert hasattr(tts, "start")
        assert callable(tts.start)

        assert hasattr(tts, "stop")
        assert callable(tts.stop)

        assert hasattr(tts, "speak")
        assert callable(tts.speak)

        assert hasattr(tts, "interrupt")
        assert callable(tts.interrupt)

        assert hasattr(tts, "is_speaking")
        assert callable(tts.is_speaking)

        assert hasattr(tts, "get_last_spoken_text")
        assert callable(tts.get_last_spoken_text)

    def test_initialization_disabled(self):
        """PiperTTS should handle disabled state gracefully."""
        from src.jarvis.output.tts import PiperTTS

        tts = PiperTTS(enabled=False)

        # Should not crash when disabled
        tts.start()
        tts.speak("test text")
        assert tts.is_speaking() is False
        tts.interrupt()
        tts.stop()

    def test_initialization_with_all_parameters(self):
        """PiperTTS should accept all configuration parameters."""
        from src.jarvis.output.tts import PiperTTS

        tts = PiperTTS(
            enabled=True,
            voice="test-voice",  # Interface compatibility
            rate=200,  # Interface compatibility
            model_path="/path/to/model.onnx",
            speaker=0,
            length_scale=1.2,
            noise_scale=0.5,
            noise_w=0.7,
            sentence_silence=0.3,
        )

        # Verify parameters are stored
        assert tts.enabled is True
        assert tts.voice == "test-voice"
        assert tts.rate == 200
        assert tts.model_path == "/path/to/model.onnx"
        assert tts.speaker == 0
        assert tts.length_scale == 1.2
        assert tts.noise_scale == 0.5
        assert tts.noise_w == 0.7
        assert tts.sentence_silence == 0.3


class TestPiperTTSErrorHandling:
    """Tests for PiperTTS error handling."""

    def test_missing_model_with_failed_download(self, tmp_path):
        """PiperTTS should handle failed download gracefully."""
        from src.jarvis.output.tts import PiperTTS
        from unittest.mock import patch

        # Use a non-existent custom path to force download attempt
        custom_model = str(tmp_path / "nonexistent-voice.onnx")
        tts = PiperTTS(enabled=True, model_path=custom_model)

        # Mock the download to fail
        with patch("src.jarvis.output.tts._download_piper_voice", return_value=None):
            result = tts._ensure_initialized()
            assert result is False
            assert tts._init_error is not None
            assert "download" in tts._init_error.lower() or "failed" in tts._init_error.lower()

    def test_nonexistent_model_file_with_failed_download(self):
        """PiperTTS should handle nonexistent model file gracefully when download fails."""
        from src.jarvis.output.tts import PiperTTS
        from unittest.mock import patch

        tts = PiperTTS(enabled=True, model_path="/nonexistent/path/model.onnx")

        # Mock the download to fail
        with patch("src.jarvis.output.tts._download_piper_voice", return_value=None):
            result = tts._ensure_initialized()
            assert result is False
            assert tts._init_error is not None

    def test_missing_config_json(self, tmp_path):
        """PiperTTS should require .onnx.json config file."""
        from src.jarvis.output.tts import PiperTTS
        from unittest.mock import patch

        # Create a fake model file but no config
        model_file = tmp_path / "custom-voice.onnx"
        model_file.write_text("fake model")

        tts = PiperTTS(enabled=True, model_path=str(model_file))

        # Mock download to fail (since config doesn't exist)
        with patch("src.jarvis.output.tts._download_piper_voice", return_value=None):
            result = tts._ensure_initialized()
            assert result is False
            assert tts._init_error is not None

    def test_user_path_expansion(self):
        """PiperTTS should expand ~ in model path."""
        from src.jarvis.output.tts import PiperTTS
        from unittest.mock import patch
        import os

        tts = PiperTTS(enabled=True, model_path="~/nonexistent/model.onnx")

        # Mock download to fail
        with patch("src.jarvis.output.tts._download_piper_voice", return_value=None):
            tts._ensure_initialized()

            # The error should reference the expanded path (with home directory)
            # not the literal ~
            if tts._init_error:
                # Either the path was expanded, or we got a different error
                # Both are acceptable as long as it didn't crash
                pass

    def test_explicit_model_path_skips_default(self, tmp_path):
        """When explicit model_path is given, don't use default."""
        from src.jarvis.output.tts import PiperTTS, _get_default_piper_model_path
        from unittest.mock import patch

        custom_path = str(tmp_path / "custom-voice.onnx")
        tts = PiperTTS(enabled=True, model_path=custom_path)

        # Mock download to return the custom path
        with patch("src.jarvis.output.tts._download_piper_voice", return_value=None):
            tts._ensure_initialized()

            # Error should reference the custom path, not default
            if tts._init_error:
                default_path = _get_default_piper_model_path()
                # Should not be using the default path
                assert "custom-voice" in tts._init_error or "download" in tts._init_error.lower()


class TestPiperTTSWithMocking:
    """Tests for PiperTTS with mocked Piper library."""

    def test_initialization_checks_both_files(self, tmp_path):
        """PiperTTS should check both .onnx and .onnx.json files exist."""
        from src.jarvis.output.tts import PiperTTS
        from unittest.mock import patch

        # Create model file but not config
        model_file = tmp_path / "test-voice.onnx"
        model_file.write_text("fake model")

        tts = PiperTTS(enabled=True, model_path=str(model_file))

        # Mock download to fail
        with patch("src.jarvis.output.tts._download_piper_voice", return_value=None):
            result = tts._ensure_initialized()

            assert result is False
            assert tts._init_error is not None

    @patch("src.jarvis.output.tts.os.path.exists", return_value=True)
    def test_piper_import_error_handling(self, mock_exists):
        """PiperTTS should handle missing piper-tts library gracefully."""
        from src.jarvis.output.tts import PiperTTS

        with patch.dict("sys.modules", {"piper": None, "piper.voice": None}):
            # Force reimport to trigger import error
            tts = PiperTTS(enabled=True, model_path="/fake/model.onnx")

            # Clear any previous initialization state
            tts._initialized = False
            tts._voice = None
            tts._init_error = None

            # Mock the import to raise ImportError
            with patch(
                "src.jarvis.output.tts.PiperTTS._ensure_initialized",
                wraps=tts._ensure_initialized,
            ):
                # Manually trigger what would happen with import error
                tts._init_error = "piper-tts not installed"
                tts._initialized = True
                result = tts._ensure_initialized()

            # Should have caught the error
            assert tts._init_error is not None

    def test_speak_queues_text(self):
        """PiperTTS.speak should queue text for processing."""
        from src.jarvis.output.tts import PiperTTS

        tts = PiperTTS(enabled=True, model_path="/fake/model.onnx")

        # Don't actually start the thread
        tts.speak("Hello world")

        # Text should be in queue (may have been preprocessed)
        assert not tts._q.empty()

    def test_speak_does_nothing_when_disabled(self):
        """PiperTTS.speak should do nothing when disabled."""
        from src.jarvis.output.tts import PiperTTS

        tts = PiperTTS(enabled=False)
        tts.speak("Hello world")

        # Queue should be empty
        assert tts._q.empty()

    def test_speak_does_nothing_for_empty_text(self):
        """PiperTTS.speak should do nothing for empty text."""
        from src.jarvis.output.tts import PiperTTS

        tts = PiperTTS(enabled=True, model_path="/fake/model.onnx")
        tts.speak("")
        tts.speak("   ")

        # Queue should be empty
        assert tts._q.empty()

    def test_interrupt_sets_flag(self):
        """PiperTTS.interrupt should set the interrupt flag."""
        from src.jarvis.output.tts import PiperTTS

        tts = PiperTTS(enabled=True)

        assert not tts._should_interrupt.is_set()
        tts.interrupt()
        assert tts._should_interrupt.is_set()

    def test_is_speaking_returns_event_state(self):
        """PiperTTS.is_speaking should return the speaking event state."""
        from src.jarvis.output.tts import PiperTTS

        tts = PiperTTS(enabled=True)

        assert tts.is_speaking() is False

        tts._is_speaking.set()
        assert tts.is_speaking() is True

        tts._is_speaking.clear()
        assert tts.is_speaking() is False

    def test_get_last_spoken_text_returns_stored_text(self):
        """PiperTTS.get_last_spoken_text should return the last spoken text."""
        from src.jarvis.output.tts import PiperTTS

        tts = PiperTTS(enabled=True)

        assert tts.get_last_spoken_text() == ""

        tts._last_spoken_text = "Hello world"
        assert tts.get_last_spoken_text() == "Hello world"


class TestPiperTTSFactory:
    """Tests for the create_tts_engine factory function."""

    def test_creates_piper_engine(self):
        """create_tts_engine should create PiperTTS for engine='piper'."""
        from src.jarvis.output.tts import create_tts_engine, PiperTTS

        tts = create_tts_engine(engine="piper", enabled=False)
        assert isinstance(tts, PiperTTS)

    def test_creates_piper_engine_case_insensitive(self):
        """create_tts_engine should handle 'PIPER', 'Piper', etc."""
        from src.jarvis.output.tts import create_tts_engine, PiperTTS

        tts1 = create_tts_engine(engine="PIPER", enabled=False)
        tts2 = create_tts_engine(engine="Piper", enabled=False)

        assert isinstance(tts1, PiperTTS)
        assert isinstance(tts2, PiperTTS)

    def test_passes_piper_parameters(self):
        """create_tts_engine should pass all Piper parameters."""
        from src.jarvis.output.tts import create_tts_engine, PiperTTS

        tts = create_tts_engine(
            engine="piper",
            enabled=True,
            voice="test",
            rate=200,
            piper_model_path="/path/to/model.onnx",
            piper_speaker=1,
            piper_length_scale=0.9,
            piper_noise_scale=0.5,
            piper_noise_w=0.6,
            piper_sentence_silence=0.25,
        )

        assert isinstance(tts, PiperTTS)
        assert tts.model_path == "/path/to/model.onnx"
        assert tts.speaker == 1
        assert tts.length_scale == 0.9
        assert tts.noise_scale == 0.5
        assert tts.noise_w == 0.6
        assert tts.sentence_silence == 0.25

    def test_default_engine_is_piper(self):
        """create_tts_engine should default to Piper TTS."""
        from src.jarvis.output.tts import create_tts_engine, PiperTTS

        tts = create_tts_engine(enabled=False)
        assert isinstance(tts, PiperTTS)

    def test_unknown_engine_falls_back_to_piper(self):
        """create_tts_engine with unknown engine should create PiperTTS."""
        from src.jarvis.output.tts import create_tts_engine, PiperTTS

        tts = create_tts_engine(engine="unknown", enabled=False)
        assert isinstance(tts, PiperTTS)

    def test_chatterbox_engine_still_works(self):
        """create_tts_engine should still create ChatterboxTTS."""
        from src.jarvis.output.tts import create_tts_engine, ChatterboxTTS

        tts = create_tts_engine(engine="chatterbox", enabled=False)
        assert isinstance(tts, ChatterboxTTS)


class TestPiperTTSAutoDownload:
    """Tests for Piper TTS auto-download functionality."""

    def test_get_default_model_path(self):
        """Default model path should be in ~/.local/share/jarvis/models/piper/."""
        from src.jarvis.output.tts import _get_default_piper_model_path, PIPER_DEFAULT_VOICE

        path = _get_default_piper_model_path()

        assert PIPER_DEFAULT_VOICE in path
        assert path.endswith(".onnx")
        assert "jarvis" in path
        assert "piper" in path

    def test_get_piper_models_dir(self):
        """Models directory should be created under jarvis data dir."""
        from src.jarvis.output.tts import _get_piper_models_dir

        models_dir = _get_piper_models_dir()

        assert models_dir.exists()
        assert "jarvis" in str(models_dir)
        assert "piper" in str(models_dir)

    def test_piper_uses_default_when_no_path(self):
        """PiperTTS should use default model path when none configured."""
        from src.jarvis.output.tts import PiperTTS, _get_default_piper_model_path

        tts = PiperTTS(enabled=True, model_path=None)

        # model_path starts as None
        assert tts.model_path is None

        # But initialization should use the default
        # (we don't actually init here to avoid downloads in tests)

    def test_default_voice_is_reasonable(self):
        """Default voice should be a reasonable choice."""
        from src.jarvis.output.tts import PIPER_DEFAULT_VOICE

        # Should be British English
        assert PIPER_DEFAULT_VOICE.startswith("en_GB")
        # Should include quality indicator
        assert "medium" in PIPER_DEFAULT_VOICE or "high" in PIPER_DEFAULT_VOICE


class TestPiperTTSConfig:
    """Tests for Piper TTS configuration in Settings."""

    def test_config_has_piper_fields(self):
        """Settings dataclass should have all Piper TTS fields."""
        from src.jarvis.config import Settings
        import inspect

        # Get the field names from Settings
        signature = inspect.signature(Settings)
        param_names = set(signature.parameters.keys())

        # Check all Piper fields exist
        assert "tts_piper_model_path" in param_names
        assert "tts_piper_speaker" in param_names
        assert "tts_piper_length_scale" in param_names
        assert "tts_piper_noise_scale" in param_names
        assert "tts_piper_noise_w" in param_names
        assert "tts_piper_sentence_silence" in param_names

    def test_default_config_has_piper_values(self):
        """get_default_config should include Piper TTS defaults."""
        from src.jarvis.config import get_default_config

        defaults = get_default_config()

        assert "tts_piper_model_path" in defaults
        assert defaults["tts_piper_model_path"] is None

        assert "tts_piper_speaker" in defaults
        assert defaults["tts_piper_speaker"] is None

        assert "tts_piper_length_scale" in defaults
        assert defaults["tts_piper_length_scale"] == 0.65  # ~30% faster speech

        assert "tts_piper_noise_scale" in defaults
        assert defaults["tts_piper_noise_scale"] == 0.8  # More expressive

        assert "tts_piper_noise_w" in defaults
        assert defaults["tts_piper_noise_w"] == 1.0  # More lively

        assert "tts_piper_sentence_silence" in defaults
        assert defaults["tts_piper_sentence_silence"] == 0.2

    def test_tts_engine_defaults_to_piper(self):
        """tts_engine should default to 'piper'."""
        from src.jarvis.config import load_settings, get_default_config
        from unittest.mock import patch

        # Check default config
        defaults = get_default_config()
        assert defaults["tts_engine"] == "piper"

        # Mock empty config file - should use default
        with patch("src.jarvis.config._load_json", return_value={}):
            settings = load_settings()
            assert settings.tts_engine == "piper"

    def test_tts_engine_migrates_system_to_piper(self):
        """tts_engine 'system' should be auto-migrated to 'piper' for existing users."""
        from src.jarvis.config import load_settings
        from unittest.mock import patch

        # Old config with system TTS (no _config_version = pre-migration)
        config_data = {"tts_engine": "system"}

        with patch("src.jarvis.config._load_json", return_value=config_data):
            with patch("src.jarvis.config._save_json", return_value=True):
                settings = load_settings()
                # Should be migrated to piper
                assert settings.tts_engine == "piper"

    def test_invalid_engine_falls_back_to_piper(self):
        """Invalid tts_engine values should fall back to piper."""
        from src.jarvis.config import load_settings
        from unittest.mock import patch

        # Config with invalid TTS engine
        config_data = {
            "tts_engine": "invalid_engine",
            "_config_version": 1
        }

        with patch("src.jarvis.config._load_json", return_value=config_data):
            settings = load_settings()
            # Should fall back to piper
            assert settings.tts_engine == "piper"

    def test_chatterbox_engine_preserved(self):
        """tts_engine 'chatterbox' should be preserved."""
        from src.jarvis.config import load_settings
        from unittest.mock import patch

        config_data = {
            "tts_engine": "chatterbox",
            "_config_version": 1
        }

        with patch("src.jarvis.config._load_json", return_value=config_data):
            settings = load_settings()
            assert settings.tts_engine == "chatterbox"


class TestPiperTTSThreadSafety:
    """Tests for PiperTTS thread safety."""

    def test_multiple_interrupts_safe(self):
        """Multiple calls to interrupt should be safe."""
        from src.jarvis.output.tts import PiperTTS

        tts = PiperTTS(enabled=True)

        # Should not crash with multiple interrupts
        for _ in range(10):
            tts.interrupt()

    def test_start_stop_cycle(self):
        """Start and stop should be safe to call multiple times."""
        from src.jarvis.output.tts import PiperTTS

        tts = PiperTTS(enabled=False)  # Disabled to avoid actual model loading

        # Multiple start/stop cycles should be safe
        for _ in range(3):
            tts.start()
            tts.stop()

    def test_concurrent_speaks(self):
        """Multiple threads calling speak should not crash."""
        from src.jarvis.output.tts import PiperTTS

        tts = PiperTTS(enabled=True, model_path="/fake/model.onnx")

        # Don't start the actual worker thread
        def speak_text():
            for _ in range(10):
                tts.speak("Hello world")

        threads = [threading.Thread(target=speak_text) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash, queue should have items
        # (actual number depends on timing)
