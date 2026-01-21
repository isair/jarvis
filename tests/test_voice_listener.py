"""
Tests for voice listener module.

These tests verify the Whisper model loading and fallback logic.
"""

from unittest.mock import patch, MagicMock, call
import pytest


class TestWhisperComputeTypeFallback:
    """Tests for Whisper compute type fallback mechanism."""

    def _create_mock_config(self, **kwargs):
        """Create a mock config object with default values."""
        mock_cfg = MagicMock()
        mock_cfg.whisper_model = kwargs.get("whisper_model", "small")
        mock_cfg.whisper_device = kwargs.get("whisper_device", "auto")
        mock_cfg.whisper_compute_type = kwargs.get("whisper_compute_type", "int8")
        mock_cfg.whisper_backend = kwargs.get("whisper_backend", "faster-whisper")
        mock_cfg.sample_rate = kwargs.get("sample_rate", 16000)
        mock_cfg.vad_enabled = kwargs.get("vad_enabled", True)
        mock_cfg.vad_aggressiveness = kwargs.get("vad_aggressiveness", 2)
        mock_cfg.echo_tolerance = kwargs.get("echo_tolerance", 0.3)
        mock_cfg.echo_energy_threshold = kwargs.get("echo_energy_threshold", 2.0)
        mock_cfg.hot_window_seconds = kwargs.get("hot_window_seconds", 6.0)
        mock_cfg.voice_collect_seconds = kwargs.get("voice_collect_seconds", 2.0)
        mock_cfg.voice_max_collect_seconds = kwargs.get("voice_max_collect_seconds", 60.0)
        mock_cfg.voice_device = kwargs.get("voice_device", None)
        mock_cfg.voice_debug = kwargs.get("voice_debug", False)
        mock_cfg.tune_enabled = kwargs.get("tune_enabled", False)
        return mock_cfg

    def test_successful_load_with_int8(self):
        """When int8 is supported, loads successfully without fallback."""
        mock_whisper_model = MagicMock()

        with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
            with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model) as mock_class:
                    with patch("jarvis.listening.listener.sd") as mock_sd:
                        mock_sd.InputStream.side_effect = Exception("Stop test here")

                        from jarvis.listening.listener import VoiceListener

                        mock_db = MagicMock()
                        mock_cfg = self._create_mock_config(whisper_compute_type="int8")
                        mock_tts = MagicMock()
                        mock_dialogue_memory = MagicMock()

                        listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)

                        # Run will attempt to load model then open audio stream
                        listener.run()

                        # Should have been called only once with int8
                        mock_class.assert_called_once_with("small", device="auto", compute_type="int8")
                        assert listener.model == mock_whisper_model

    def test_fallback_from_int8_to_float16(self):
        """When int8 fails with compute type error, falls back to float16."""
        mock_whisper_model = MagicMock()

        def whisper_model_side_effect(model_name, device, compute_type):
            if compute_type == "int8":
                raise RuntimeError("Requested int8 compute type, but the target device or backend do not support efficient int8 computation.")
            return mock_whisper_model

        with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
            with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                with patch("jarvis.listening.listener.WhisperModel", side_effect=whisper_model_side_effect) as mock_class:
                    with patch("jarvis.listening.listener.sd") as mock_sd:
                        mock_sd.InputStream.side_effect = Exception("Stop test here")

                        from jarvis.listening.listener import VoiceListener

                        mock_db = MagicMock()
                        mock_cfg = self._create_mock_config(whisper_compute_type="int8")
                        mock_tts = MagicMock()
                        mock_dialogue_memory = MagicMock()

                        listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                        listener.run()

                        # Should have tried int8 first, then float16
                        assert mock_class.call_count == 2
                        calls = mock_class.call_args_list
                        assert calls[0] == call("small", device="auto", compute_type="int8")
                        assert calls[1] == call("small", device="auto", compute_type="float16")
                        assert listener.model == mock_whisper_model

    def test_fallback_from_int8_to_float32(self):
        """When int8 and float16 both fail, falls back to float32."""
        mock_whisper_model = MagicMock()

        def whisper_model_side_effect(model_name, device, compute_type):
            if compute_type in ("int8", "float16"):
                raise RuntimeError(f"Requested {compute_type} compute type, but not supported.")
            return mock_whisper_model

        with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
            with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                with patch("jarvis.listening.listener.WhisperModel", side_effect=whisper_model_side_effect) as mock_class:
                    with patch("jarvis.listening.listener.sd") as mock_sd:
                        mock_sd.InputStream.side_effect = Exception("Stop test here")

                        from jarvis.listening.listener import VoiceListener

                        mock_db = MagicMock()
                        mock_cfg = self._create_mock_config(whisper_compute_type="int8")
                        mock_tts = MagicMock()
                        mock_dialogue_memory = MagicMock()

                        listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                        listener.run()

                        # Should have tried int8, float16, then float32
                        assert mock_class.call_count == 3
                        calls = mock_class.call_args_list
                        assert calls[0] == call("small", device="auto", compute_type="int8")
                        assert calls[1] == call("small", device="auto", compute_type="float16")
                        assert calls[2] == call("small", device="auto", compute_type="float32")
                        assert listener.model == mock_whisper_model

    def test_no_fallback_for_non_compute_type_errors(self):
        """When error is not about compute type, doesn't try fallback."""
        with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
            with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                with patch("jarvis.listening.listener.WhisperModel") as mock_class:
                    mock_class.side_effect = RuntimeError("Model not found: invalid_model")

                    with patch("jarvis.listening.listener.sd") as mock_sd:
                        from jarvis.listening.listener import VoiceListener

                        mock_db = MagicMock()
                        mock_cfg = self._create_mock_config(whisper_compute_type="int8")
                        mock_tts = MagicMock()
                        mock_dialogue_memory = MagicMock()

                        listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                        listener.run()

                        # Should have only tried once - no fallback for model not found errors
                        mock_class.assert_called_once_with("small", device="auto", compute_type="int8")
                        assert listener.model is None

    def test_all_fallbacks_fail(self):
        """When all compute types fail, model remains None."""
        def whisper_model_side_effect(model_name, device, compute_type):
            raise RuntimeError(f"Requested {compute_type} compute type, but not supported.")

        with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
            with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                with patch("jarvis.listening.listener.WhisperModel", side_effect=whisper_model_side_effect) as mock_class:
                    with patch("jarvis.listening.listener.sd") as mock_sd:
                        from jarvis.listening.listener import VoiceListener

                        mock_db = MagicMock()
                        mock_cfg = self._create_mock_config(whisper_compute_type="int8")
                        mock_tts = MagicMock()
                        mock_dialogue_memory = MagicMock()

                        listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                        listener.run()

                        # Should have tried all three compute types
                        assert mock_class.call_count == 3
                        assert listener.model is None

    def test_float16_config_skips_float16_in_fallback_list(self):
        """When config is float16, fallback list is [float16, float32]."""
        mock_whisper_model = MagicMock()

        def whisper_model_side_effect(model_name, device, compute_type):
            if compute_type == "float16":
                raise RuntimeError("Requested float16 compute type, but not supported.")
            return mock_whisper_model

        with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
            with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                with patch("jarvis.listening.listener.WhisperModel", side_effect=whisper_model_side_effect) as mock_class:
                    with patch("jarvis.listening.listener.sd") as mock_sd:
                        mock_sd.InputStream.side_effect = Exception("Stop test here")

                        from jarvis.listening.listener import VoiceListener

                        mock_db = MagicMock()
                        # Config specifies float16 instead of int8
                        mock_cfg = self._create_mock_config(whisper_compute_type="float16")
                        mock_tts = MagicMock()
                        mock_dialogue_memory = MagicMock()

                        listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                        listener.run()

                        # Should have tried float16, then float32 (no duplicate float16)
                        assert mock_class.call_count == 2
                        calls = mock_class.call_args_list
                        assert calls[0] == call("small", device="auto", compute_type="float16")
                        assert calls[1] == call("small", device="auto", compute_type="float32")
                        assert listener.model == mock_whisper_model

    def test_float32_config_no_fallback_needed(self):
        """When config is float32, there's only float32 to try."""
        with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
            with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                with patch("jarvis.listening.listener.WhisperModel") as mock_class:
                    mock_class.side_effect = RuntimeError("Requested float32 compute type, but not supported.")

                    with patch("jarvis.listening.listener.sd") as mock_sd:
                        from jarvis.listening.listener import VoiceListener

                        mock_db = MagicMock()
                        # Config specifies float32
                        mock_cfg = self._create_mock_config(whisper_compute_type="float32")
                        mock_tts = MagicMock()
                        mock_dialogue_memory = MagicMock()

                        listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                        listener.run()

                        # Should have only tried float32 (no other fallbacks)
                        mock_class.assert_called_once_with("small", device="auto", compute_type="float32")
                        assert listener.model is None
