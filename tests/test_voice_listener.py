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

        # Mock sys.platform to skip Windows CUDA check
        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model) as mock_class:
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            # Mock query_devices to return a fake input device
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
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

        # Mock sys.platform to skip Windows CUDA check
        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", side_effect=whisper_model_side_effect) as mock_class:
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
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

        # Mock sys.platform to skip Windows CUDA check
        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", side_effect=whisper_model_side_effect) as mock_class:
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
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
        # Mock sys.platform to skip Windows CUDA check
        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel") as mock_class:
                        mock_class.side_effect = RuntimeError("Model not found: invalid_model")

                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
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

        # Mock sys.platform to skip Windows CUDA check
        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", side_effect=whisper_model_side_effect) as mock_class:
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                            from jarvis.listening.listener import VoiceListener

                            mock_db = MagicMock()
                            mock_cfg = self._create_mock_config(whisper_compute_type="int8")
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                            listener.run()

                            # Should have tried all configs: 3 compute types x 2 devices (auto + cpu fallback)
                            assert mock_class.call_count == 6
                            assert listener.model is None

    def test_float16_config_skips_float16_in_fallback_list(self):
        """When config is float16, fallback list is [float16, float32]."""
        mock_whisper_model = MagicMock()

        def whisper_model_side_effect(model_name, device, compute_type):
            if compute_type == "float16":
                raise RuntimeError("Requested float16 compute type, but not supported.")
            return mock_whisper_model

        # Mock sys.platform to skip Windows CUDA check
        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", side_effect=whisper_model_side_effect) as mock_class:
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
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
        """When config is float32, tries float32 on auto then cpu."""
        # Mock sys.platform to skip Windows CUDA check
        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel") as mock_class:
                        mock_class.side_effect = RuntimeError("Requested float32 compute type, but not supported.")

                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                            from jarvis.listening.listener import VoiceListener

                            mock_db = MagicMock()
                            # Config specifies float32
                            mock_cfg = self._create_mock_config(whisper_compute_type="float32")
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                            listener.run()

                            # Should have tried float32 on auto, then cpu fallback
                            assert mock_class.call_count == 2
                            calls = mock_class.call_args_list
                            assert calls[0] == call("small", device="auto", compute_type="float32")
                            assert calls[1] == call("small", device="cpu", compute_type="float32")
                            assert listener.model is None


class TestRepetitiveHallucinationDetection:
    """Tests for Whisper hallucination detection."""

    def _create_mock_listener(self):
        """Create a VoiceListener instance for testing."""
        with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
            with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                with patch("jarvis.listening.listener.WhisperModel"):
                    with patch("jarvis.listening.listener.webrtcvad", None):
                        from jarvis.listening.listener import VoiceListener

                        mock_db = MagicMock()
                        mock_cfg = MagicMock()
                        mock_cfg.sample_rate = 16000
                        mock_cfg.vad_enabled = False
                        mock_cfg.echo_tolerance = 0.3
                        mock_cfg.echo_energy_threshold = 2.0
                        mock_cfg.hot_window_seconds = 6.0
                        mock_cfg.voice_collect_seconds = 2.0
                        mock_cfg.voice_max_collect_seconds = 60.0
                        mock_cfg.tune_enabled = False
                        mock_tts = MagicMock()
                        mock_dialogue_memory = MagicMock()

                        return VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)

    def test_detects_repeated_single_word_dont(self):
        """Detects 'don't don't don't...' repetition pattern."""
        listener = self._create_mock_listener()
        text = "don't don't don't don't don't don't don't don't"
        assert listener._is_repetitive_hallucination(text) is True

    def test_detects_repeated_single_word_don(self):
        """Detects 'don don don...' repetition pattern."""
        listener = self._create_mock_listener()
        text = "don don don don don don don don don don"
        assert listener._is_repetitive_hallucination(text) is True

    def test_detects_repeated_stop(self):
        """Detects 'stop stop stop...' repetition pattern."""
        listener = self._create_mock_listener()
        text = "stop stop stop stop stop stop"
        assert listener._is_repetitive_hallucination(text) is True

    def test_detects_consecutive_repetition(self):
        """Detects any word repeated 3+ times consecutively."""
        listener = self._create_mock_listener()
        text = "hello hello hello hello there"
        assert listener._is_repetitive_hallucination(text) is True

    def test_accepts_normal_speech(self):
        """Accepts normal speech with natural repetition."""
        listener = self._create_mock_listener()
        text = "what is the weather today"
        assert listener._is_repetitive_hallucination(text) is False

    def test_accepts_short_text(self):
        """Doesn't flag short text even with repetition."""
        listener = self._create_mock_listener()
        text = "stop stop"
        assert listener._is_repetitive_hallucination(text) is False

    def test_accepts_natural_repetition(self):
        """Accepts text with natural word repetition below threshold."""
        listener = self._create_mock_listener()
        text = "I really really want to go home now"
        assert listener._is_repetitive_hallucination(text) is False

    def test_accepts_empty_text(self):
        """Returns False for empty text."""
        listener = self._create_mock_listener()
        assert listener._is_repetitive_hallucination("") is False
        assert listener._is_repetitive_hallucination("   ") is False

    def test_detects_majority_same_word(self):
        """Detects when a word appears more than 50% of the time."""
        listener = self._create_mock_listener()
        text = "the the the the the hello world"  # 'the' is 5/7 = 71%
        assert listener._is_repetitive_hallucination(text) is True

    def test_accepts_mixed_content(self):
        """Accepts text with varied words even if some repeat."""
        listener = self._create_mock_listener()
        text = "the quick brown fox jumps over the lazy dog"  # 'the' is 2/9 = 22%
        assert listener._is_repetitive_hallucination(text) is False

    def test_detects_japanese_latin_repetition(self):
        """Detects 'Jろ Jろ Jろ...' mixed character repetition."""
        listener = self._create_mock_listener()
        text = "Jろ Jろ Jろ Jろ Jろ Jろ"
        assert listener._is_repetitive_hallucination(text) is True

    def test_detects_no_space_repetition(self):
        """Detects repetition without spaces."""
        listener = self._create_mock_listener()
        text = "JろJろJろJろJろJろ"
        assert listener._is_repetitive_hallucination(text) is True

    def test_detects_single_char_repetition(self):
        """Detects single character repetition."""
        listener = self._create_mock_listener()
        text = "aaaaaaaaaaaaa"
        assert listener._is_repetitive_hallucination(text) is True

    def test_detects_word_with_trailing_punctuation(self):
        """Detects repetition even with trailing punctuation."""
        listener = self._create_mock_listener()
        text = "don don don don don don..."
        assert listener._is_repetitive_hallucination(text) is True

    def test_detects_whisper_thanks_pattern(self):
        """Detects common Whisper hallucination 'Thanks for watching!'."""
        listener = self._create_mock_listener()
        # Whisper sometimes outputs this for silence - consecutive word repetition
        # "thanks" appears 4/8 words = 50% but words repeat consecutively as phrases
        text = "Thanks Thanks Thanks Thanks for watching"
        assert listener._is_repetitive_hallucination(text) is True

    def test_accepts_short_repetition(self):
        """Doesn't flag short character strings even with repetition."""
        listener = self._create_mock_listener()
        text = "aaaa"  # Only 4 chars, too short
        assert listener._is_repetitive_hallucination(text) is False

    def test_accepts_partial_repetition(self):
        """Accepts text where repetition is only partial."""
        listener = self._create_mock_listener()
        text = "hello hello world this is a normal sentence"
        assert listener._is_repetitive_hallucination(text) is False
