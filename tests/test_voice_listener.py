"""
Tests for voice listener module.

These tests verify the Whisper model loading and fallback logic.
"""

from unittest.mock import patch, MagicMock, call
import time
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
        mock_cfg.hot_window_seconds = kwargs.get("hot_window_seconds", 3.0)
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
                            mock_class.assert_called_once()
                            assert mock_class.call_args[1]["device"] == "auto"
                            assert mock_class.call_args[1]["compute_type"] == "int8"
                            assert listener.model == mock_whisper_model

    def test_fallback_from_int8_to_float16(self):
        """When int8 fails with compute type error, falls back to float16."""
        mock_whisper_model = MagicMock()

        def whisper_model_side_effect(model_name, device, compute_type, **kwargs):
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
                            assert calls[0][1]["device"] == "auto"
                            assert calls[0][1]["compute_type"] == "int8"
                            assert calls[1][1]["device"] == "auto"
                            assert calls[1][1]["compute_type"] == "float16"
                            assert listener.model == mock_whisper_model

    def test_fallback_from_int8_to_float32(self):
        """When int8 and float16 both fail, falls back to float32."""
        mock_whisper_model = MagicMock()

        def whisper_model_side_effect(model_name, device, compute_type, **kwargs):
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
                            assert calls[0][1]["device"] == "auto"
                            assert calls[0][1]["compute_type"] == "int8"
                            assert calls[1][1]["device"] == "auto"
                            assert calls[1][1]["compute_type"] == "float16"
                            assert calls[2][1]["device"] == "auto"
                            assert calls[2][1]["compute_type"] == "float32"
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
                            mock_class.assert_called_once()
                            assert mock_class.call_args[1]["device"] == "auto"
                            assert mock_class.call_args[1]["compute_type"] == "int8"
                            assert listener.model is None

    def test_all_fallbacks_fail(self):
        """When all compute types fail, model remains None."""
        def whisper_model_side_effect(model_name, device, compute_type, **kwargs):
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

        def whisper_model_side_effect(model_name, device, compute_type, **kwargs):
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
                            assert calls[0][1]["device"] == "auto"
                            assert calls[0][1]["compute_type"] == "float16"
                            assert calls[1][1]["device"] == "auto"
                            assert calls[1][1]["compute_type"] == "float32"
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
                            assert calls[0][1]["device"] == "auto"
                            assert calls[0][1]["compute_type"] == "float32"
                            assert calls[1][1]["device"] == "cpu"
                            assert calls[1][1]["compute_type"] == "float32"
                            assert listener.model is None


class TestWindowsCudaDetection:
    """Tests for Windows CUDA detection logic."""

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
        mock_cfg.hot_window_seconds = kwargs.get("hot_window_seconds", 3.0)
        mock_cfg.voice_collect_seconds = kwargs.get("voice_collect_seconds", 2.0)
        mock_cfg.voice_max_collect_seconds = kwargs.get("voice_max_collect_seconds", 60.0)
        mock_cfg.voice_device = kwargs.get("voice_device", None)
        mock_cfg.voice_debug = kwargs.get("voice_debug", False)
        mock_cfg.tune_enabled = kwargs.get("tune_enabled", False)
        return mock_cfg

    def test_cuda_detected_when_all_dlls_present(self):
        """When cuBLAS and cuDNN DLLs are found, GPU mode is used."""
        mock_whisper_model = MagicMock()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch("jarvis.listening.listener._setup_nvidia_dll_path"):
                with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                    with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                        with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model) as mock_class:
                            with patch("jarvis.listening.listener.sd") as mock_sd:
                                mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                                mock_sd.InputStream.side_effect = Exception("Stop test here")

                                # All CUDA DLLs load successfully
                                mock_ctypes = MagicMock()
                                mock_ctypes.CDLL.return_value = MagicMock()
                                with patch.dict("sys.modules", {"ctypes": mock_ctypes}):
                                    from jarvis.listening.listener import VoiceListener

                                    mock_cfg = self._create_mock_config()
                                    listener = VoiceListener(MagicMock(), mock_cfg, MagicMock(), MagicMock())
                                    listener.run()

                                    # Should use auto (not forced to cpu)
                                    assert mock_class.call_args_list[0][1]["device"] == "auto"

    def test_cuda_not_available_forces_cpu(self):
        """When CUDA DLLs are missing, falls back to CPU mode."""
        mock_whisper_model = MagicMock()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch("jarvis.listening.listener._setup_nvidia_dll_path"):
                with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                    with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                        with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model) as mock_class:
                            with patch("jarvis.listening.listener.sd") as mock_sd:
                                mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                                mock_sd.InputStream.side_effect = Exception("Stop test here")

                                # All DLL probes fail
                                mock_ctypes = MagicMock()
                                mock_ctypes.CDLL.side_effect = OSError("not found")
                                with patch.dict("sys.modules", {"ctypes": mock_ctypes}):
                                    from jarvis.listening.listener import VoiceListener

                                    mock_cfg = self._create_mock_config()
                                    listener = VoiceListener(MagicMock(), mock_cfg, MagicMock(), MagicMock())
                                    listener.run()

                                    # Should be forced to cpu
                                    assert mock_class.call_args_list[0][1]["device"] == "cpu"

    def test_setup_nvidia_dll_path_adds_pip_package_dirs(self):
        """_setup_nvidia_dll_path adds NVIDIA pip package bin dirs to PATH."""
        import os
        from jarvis.listening.listener import _setup_nvidia_dll_path

        original_path = os.environ.get("PATH", "")

        # Remove any existing nvidia paths so we can detect new additions
        clean_path = os.pathsep.join(
            p for p in original_path.split(os.pathsep)
            if "nvidia" not in p.lower()
        )
        os.environ["PATH"] = clean_path

        try:
            _setup_nvidia_dll_path()
            new_path = os.environ.get("PATH", "")

            # Should have added nvidia DLL dirs (either real pip packages or nothing)
            # If nvidia packages are installed, their bin dirs should be on PATH
            try:
                import nvidia.cublas
                cublas_bin = os.path.join(nvidia.cublas.__path__[0], "bin")
                if os.path.isdir(cublas_bin):
                    assert cublas_bin in new_path
            except ImportError:
                pass  # nvidia packages not installed, nothing to add

            try:
                import nvidia.cudnn
                cudnn_bin = os.path.join(nvidia.cudnn.__path__[0], "bin")
                if os.path.isdir(cudnn_bin):
                    assert cudnn_bin in new_path
            except ImportError:
                pass  # nvidia packages not installed, nothing to add
        finally:
            os.environ["PATH"] = original_path


class TestLargeV3TurboFallback:
    """Tests for large-v3-turbo runtime fallback when faster-whisper is too old."""

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
        mock_cfg.hot_window_seconds = kwargs.get("hot_window_seconds", 3.0)
        mock_cfg.voice_collect_seconds = kwargs.get("voice_collect_seconds", 2.0)
        mock_cfg.voice_max_collect_seconds = kwargs.get("voice_max_collect_seconds", 60.0)
        mock_cfg.voice_device = kwargs.get("voice_device", None)
        mock_cfg.voice_debug = kwargs.get("voice_debug", False)
        mock_cfg.tune_enabled = kwargs.get("tune_enabled", False)
        return mock_cfg

    def test_turbo_falls_back_to_large_v3_when_unsupported(self, capsys):
        """large-v3-turbo config falls back to large-v3 when faster-whisper < 1.1.0."""
        mock_whisper_model = MagicMock()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model) as mock_class:
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            with patch("jarvis.listening.listener._is_faster_whisper_turbo_supported", return_value=False):
                                mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                                mock_sd.InputStream.side_effect = Exception("Stop test here")

                                from jarvis.listening.listener import VoiceListener

                                mock_cfg = self._create_mock_config(whisper_model="large-v3-turbo")
                                listener = VoiceListener(MagicMock(), mock_cfg, MagicMock(), MagicMock())
                                listener.run()

                                # Should load large-v3 instead of large-v3-turbo
                                mock_class.assert_called_once()
                                assert mock_class.call_args[0][0] == "large-v3"

        captured = capsys.readouterr()
        assert "large-v3-turbo is not supported" in captured.out

    def test_turbo_kept_when_faster_whisper_supports_it(self):
        """large-v3-turbo config is kept when faster-whisper >= 1.1.0."""
        mock_whisper_model = MagicMock()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model) as mock_class:
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            with patch("jarvis.listening.listener._is_faster_whisper_turbo_supported", return_value=True):
                                mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                                mock_sd.InputStream.side_effect = Exception("Stop test here")

                                from jarvis.listening.listener import VoiceListener

                                mock_cfg = self._create_mock_config(whisper_model="large-v3-turbo")
                                listener = VoiceListener(MagicMock(), mock_cfg, MagicMock(), MagicMock())
                                listener.run()

                                # Should keep large-v3-turbo
                                mock_class.assert_called_once()
                                assert mock_class.call_args[0][0] == "large-v3-turbo"


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
                        mock_cfg.hot_window_seconds = 3.0
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


class TestCpuOptimisations:
    """Tests for faster-whisper CPU mode optimisations."""

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
        mock_cfg.hot_window_seconds = kwargs.get("hot_window_seconds", 3.0)
        mock_cfg.voice_collect_seconds = kwargs.get("voice_collect_seconds", 2.0)
        mock_cfg.voice_max_collect_seconds = kwargs.get("voice_max_collect_seconds", 60.0)
        mock_cfg.voice_device = kwargs.get("voice_device", None)
        mock_cfg.voice_debug = kwargs.get("voice_debug", False)
        mock_cfg.tune_enabled = kwargs.get("tune_enabled", False)
        return mock_cfg

    def test_cpu_threads_set_when_device_is_cpu(self):
        """CPU cores are passed to WhisperModel when device resolves to cpu."""
        mock_whisper_model = MagicMock()
        # Simulate CTranslate2 model exposing device as string
        mock_whisper_model.model.device = "cpu"

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model) as mock_class:
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                            mock_sd.InputStream.side_effect = Exception("Stop test here")
                            with patch("jarvis.listening.listener.os.cpu_count", return_value=8):
                                from jarvis.listening.listener import VoiceListener

                                mock_cfg = self._create_mock_config(whisper_device="cpu")
                                listener = VoiceListener(MagicMock(), mock_cfg, MagicMock(), MagicMock())
                                listener.run()

                                assert mock_class.call_args[1]["cpu_threads"] == 8

    def test_cpu_threads_set_when_device_is_auto(self):
        """CPU cores are passed to WhisperModel when device is auto (may resolve to CPU)."""
        mock_whisper_model = MagicMock()
        mock_whisper_model.model.device = "cpu"

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model) as mock_class:
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                            mock_sd.InputStream.side_effect = Exception("Stop test here")
                            with patch("jarvis.listening.listener.os.cpu_count", return_value=12):
                                from jarvis.listening.listener import VoiceListener

                                mock_cfg = self._create_mock_config(whisper_device="auto")
                                listener = VoiceListener(MagicMock(), mock_cfg, MagicMock(), MagicMock())
                                listener.run()

                                assert mock_class.call_args[1]["cpu_threads"] == 12

    def test_resolved_device_stored_from_ctranslate2(self):
        """The resolved device from CTranslate2 is stored on the listener."""
        mock_whisper_model = MagicMock()
        mock_whisper_model.model.device = "cpu"

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model):
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                            mock_sd.InputStream.side_effect = Exception("Stop test here")

                            from jarvis.listening.listener import VoiceListener

                            mock_cfg = self._create_mock_config()
                            listener = VoiceListener(MagicMock(), mock_cfg, MagicMock(), MagicMock())
                            listener.run()

                            assert listener._whisper_device == "cpu"

    def test_resolved_device_handles_enum(self):
        """Device resolution works even if CTranslate2 returns an enum-like object."""
        mock_whisper_model = MagicMock()
        # Simulate an enum that str() converts to "cpu"
        mock_device = MagicMock()
        mock_device.__str__ = lambda self: "cpu"
        mock_whisper_model.model.device = mock_device

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model):
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                            mock_sd.InputStream.side_effect = Exception("Stop test here")

                            from jarvis.listening.listener import VoiceListener

                            mock_cfg = self._create_mock_config()
                            listener = VoiceListener(MagicMock(), mock_cfg, MagicMock(), MagicMock())
                            listener.run()

                            assert listener._whisper_device == "cpu"

    def _create_listener_for_transcribe_test(self, whisper_device):
        """Create a VoiceListener wired up for transcription tests."""
        import numpy as np

        mock_whisper_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "hello"
        mock_info = MagicMock()
        mock_whisper_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
            with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                with patch("jarvis.listening.listener.WhisperModel"):
                    from jarvis.listening.listener import VoiceListener

                    mock_cfg = MagicMock()
                    mock_cfg.sample_rate = 16000
                    mock_cfg.vad_enabled = False
                    mock_cfg.echo_tolerance = 0.3
                    mock_cfg.echo_energy_threshold = 2.0
                    mock_cfg.hot_window_seconds = 3.0
                    mock_cfg.voice_collect_seconds = 2.0
                    mock_cfg.voice_max_collect_seconds = 60.0
                    mock_cfg.tune_enabled = False
                    mock_cfg.voice_debug = False
                    mock_cfg.whisper_min_confidence = 0.3
                    mock_cfg.whisper_min_audio_duration = 0.15

                    listener = VoiceListener(MagicMock(), mock_cfg, MagicMock(), MagicMock())
                    listener.model = mock_whisper_model
                    listener._whisper_backend = "faster-whisper"
                    listener._whisper_device = whisper_device
                    listener._samplerate = 16000

                    # Set up state so _finalize_utterance reaches transcription
                    listener._utterance_frames = [np.zeros(16000, dtype=np.float32)]
                    listener.echo_detector._utterance_start_time = time.time() - 1.0
                    listener.is_speech_active = True

                    return listener, mock_whisper_model

    def test_cpu_optimisations_in_transcribe(self):
        """CPU mode passes without_timestamps and disables condition_on_previous_text."""
        listener, mock_model = self._create_listener_for_transcribe_test("cpu")
        listener._finalize_utterance()

        mock_model.transcribe.assert_called_once()
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["without_timestamps"] is True
        assert call_kwargs["condition_on_previous_text"] is False

    def test_gpu_does_not_get_cpu_optimisations(self):
        """CUDA mode does not apply CPU-specific transcribe optimisations."""
        listener, mock_model = self._create_listener_for_transcribe_test("cuda")
        listener._finalize_utterance()

        mock_model.transcribe.assert_called_once()
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["without_timestamps"] is False
        assert call_kwargs["condition_on_previous_text"] is True


class TestRepetitiveHallucinationDetectionExtended:
    """Additional tests for Whisper hallucination detection."""

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
                        mock_cfg.hot_window_seconds = 3.0
                        mock_cfg.voice_collect_seconds = 2.0
                        mock_cfg.voice_max_collect_seconds = 60.0
                        mock_cfg.tune_enabled = False
                        mock_tts = MagicMock()
                        mock_dialogue_memory = MagicMock()

                        return VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)

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

    def test_detects_multi_char_pattern_no_spaces(self):
        """Detects repeating multi-character pattern without spaces."""
        listener = self._create_mock_listener()
        assert listener._is_repetitive_hallucination("abcabcabcabcabc") is True

    def test_accepts_low_coverage_pattern(self):
        """Pattern repeating 4+ times but covering <60% of text is not flagged."""
        listener = self._create_mock_listener()
        assert listener._is_repetitive_hallucination(
            "abababab this is a completely different long sentence") is False

    def test_detects_word_with_varying_punctuation(self):
        """Detects repetition even with varying punctuation across words."""
        listener = self._create_mock_listener()
        assert listener._is_repetitive_hallucination("stop. stop! stop? stop, stop") is True

    def test_accepts_repeated_word_below_50_percent(self):
        """Word appearing 4+ times but <50% of total words is not flagged."""
        listener = self._create_mock_listener()
        # "the" appears 4 times = 4/10 = 40%
        assert listener._is_repetitive_hallucination(
            "the cat and the dog and the bird and the fish") is False

    def test_accepts_two_consecutive_only(self):
        """Only 2 consecutive repetitions — not enough to flag."""
        listener = self._create_mock_listener()
        assert listener._is_repetitive_hallucination(
            "I think think that is fine really") is False


class TestMicPermissionHint:
    """Tests for platform-aware microphone permission hint."""

    def test_windows_hint(self):
        """Returns Windows-specific hint on win32."""
        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "win32"
            from jarvis.listening.listener import _get_mic_permission_hint
            # Re-import won't re-evaluate, so call with patched sys
            # Need to call the function while sys is patched
        # The function reads sys.platform at call time
        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "win32"
            from jarvis.listening.listener import _get_mic_permission_hint
            result = _get_mic_permission_hint()
            assert "Windows Settings" in result

    def test_macos_hint(self):
        """Returns macOS-specific hint on darwin."""
        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "darwin"
            from jarvis.listening.listener import _get_mic_permission_hint
            result = _get_mic_permission_hint()
            assert "System Settings" in result

    def test_linux_hint(self):
        """Returns Linux-specific hint on linux."""
        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            from jarvis.listening.listener import _get_mic_permission_hint
            result = _get_mic_permission_hint()
            assert "pactl" in result


class TestCrossPlatformDeviceLogging:
    """Tests for cross-platform audio device name logging."""

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
        mock_cfg.hot_window_seconds = kwargs.get("hot_window_seconds", 3.0)
        mock_cfg.voice_collect_seconds = kwargs.get("voice_collect_seconds", 2.0)
        mock_cfg.voice_max_collect_seconds = kwargs.get("voice_max_collect_seconds", 60.0)
        mock_cfg.voice_device = kwargs.get("voice_device", None)
        mock_cfg.voice_debug = kwargs.get("voice_debug", False)
        mock_cfg.tune_enabled = kwargs.get("tune_enabled", False)
        return mock_cfg

    def test_device_name_printed_on_linux(self, capsys):
        """Device name is printed on Linux, not just Windows."""
        mock_whisper_model = MagicMock()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model):
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [
                                {"name": "Linux PulseAudio Mic", "max_input_channels": 1}
                            ]
                            mock_default = MagicMock()
                            mock_default.device = (0, 0)
                            mock_sd.default = mock_default
                            # query_devices with index returns specific device
                            mock_sd.query_devices.side_effect = lambda *args: (
                                {"name": "Linux PulseAudio Mic", "max_input_channels": 1}
                                if args else [{"name": "Linux PulseAudio Mic", "max_input_channels": 1}]
                            )
                            mock_sd.InputStream.side_effect = Exception("Stop test here")

                            from jarvis.listening.listener import VoiceListener

                            mock_db = MagicMock()
                            mock_cfg = self._create_mock_config()
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                            listener.run()

                            captured = capsys.readouterr()
                            assert "🎤" in captured.out
                            assert "Linux PulseAudio Mic" in captured.out

    def test_device_name_printed_on_macos(self, capsys):
        """Device name is printed on macOS, not just Windows."""
        mock_whisper_model = MagicMock()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "darwin"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model):
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [
                                {"name": "MacBook Pro Microphone", "max_input_channels": 1}
                            ]
                            mock_default = MagicMock()
                            mock_default.device = (0, 0)
                            mock_sd.default = mock_default
                            mock_sd.query_devices.side_effect = lambda *args: (
                                {"name": "MacBook Pro Microphone", "max_input_channels": 1}
                                if args else [{"name": "MacBook Pro Microphone", "max_input_channels": 1}]
                            )
                            mock_sd.InputStream.side_effect = Exception("Stop test here")

                            from jarvis.listening.listener import VoiceListener

                            mock_db = MagicMock()
                            mock_cfg = self._create_mock_config()
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                            listener.run()

                            captured = capsys.readouterr()
                            assert "🎤" in captured.out
                            assert "MacBook Pro Microphone" in captured.out


class TestCrossPlatformAudioHealthWarning:
    """Tests for cross-platform audio health monitoring."""

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
        mock_cfg.hot_window_seconds = kwargs.get("hot_window_seconds", 3.0)
        mock_cfg.voice_collect_seconds = kwargs.get("voice_collect_seconds", 2.0)
        mock_cfg.voice_max_collect_seconds = kwargs.get("voice_max_collect_seconds", 60.0)
        mock_cfg.voice_device = kwargs.get("voice_device", None)
        mock_cfg.voice_debug = kwargs.get("voice_debug", False)
        mock_cfg.tune_enabled = kwargs.get("tune_enabled", False)
        return mock_cfg

    def test_health_warning_fires_on_linux(self, capsys):
        """Audio health warning fires on Linux when no audio received after 5s."""
        mock_whisper_model = MagicMock()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model):
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [
                                {"name": "Test Mic", "max_input_channels": 1}
                            ]
                            mock_default = MagicMock()
                            mock_default.device = (0, 0)
                            mock_sd.default = mock_default
                            mock_sd.query_devices.side_effect = lambda *args: (
                                {"name": "Test Mic", "max_input_channels": 1}
                                if args else [{"name": "Test Mic", "max_input_channels": 1}]
                            )

                            # Create a mock stream that is active
                            mock_stream = MagicMock()
                            mock_stream.active = True
                            mock_stream.__enter__ = MagicMock(return_value=mock_stream)
                            mock_stream.__exit__ = MagicMock(return_value=False)
                            mock_sd.InputStream.return_value = mock_stream

                            from jarvis.listening.listener import VoiceListener
                            import queue as q

                            mock_db = MagicMock()
                            mock_cfg = self._create_mock_config()
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)

                            # Make _audio_q.get raise Empty then stop the loop
                            get_calls = [0]
                            def fake_get(timeout=0.2):
                                get_calls[0] += 1
                                if get_calls[0] >= 3:
                                    listener._should_stop = True
                                raise q.Empty()

                            listener._audio_q = MagicMock()
                            listener._audio_q.get = fake_get
                            listener._callback_count = 0

                            # time.time() is called first for _audio_start_time (baseline),
                            # then in the loop for the health check (needs to be 6s later)
                            _base = time.time()
                            time_calls = [0]

                            def advancing_time():
                                time_calls[0] += 1
                                # First call sets _audio_start_time baseline
                                if time_calls[0] == 1:
                                    return _base
                                # Subsequent calls return 6s later
                                return _base + 6

                            with patch("jarvis.listening.listener.time") as mock_time:
                                mock_time.time.side_effect = advancing_time
                                mock_time.sleep = time.sleep

                                listener.run()

                            captured = capsys.readouterr()
                            assert "No audio received after 5 seconds" in captured.out
                            assert "pactl" in captured.out


class TestResample:
    """Tests for the _resample helper function."""

    def test_identity_when_rates_match(self):
        """When src and dst rates are the same, returns the same object."""
        import numpy as _np
        from jarvis.listening.listener import _resample

        audio = _np.ones(160, dtype=_np.float32)
        result = _resample(audio, 16000, 16000)
        assert result is audio

    def test_downsample_48k_to_16k(self):
        """Downsampling from 48 kHz to 16 kHz produces correct length and dtype."""
        import numpy as _np
        from jarvis.listening.listener import _resample

        src_rate, dst_rate = 48000, 16000
        duration = 1.0  # 1 second
        audio = _np.random.randn(int(src_rate * duration)).astype(_np.float32)
        result = _resample(audio, src_rate, dst_rate)

        expected_len = int(len(audio) * dst_rate / src_rate)
        assert len(result) == expected_len
        assert result.dtype == _np.float32

    def test_upsample_8k_to_16k(self):
        """Upsampling from 8 kHz to 16 kHz produces correct length."""
        import numpy as _np
        from jarvis.listening.listener import _resample

        src_rate, dst_rate = 8000, 16000
        duration = 0.5
        audio = _np.random.randn(int(src_rate * duration)).astype(_np.float32)
        result = _resample(audio, src_rate, dst_rate)

        expected_len = int(len(audio) * dst_rate / src_rate)
        assert len(result) == expected_len

    def test_preserves_sine_wave_frequency(self):
        """A 440 Hz sine resampled from 48 kHz to 16 kHz keeps its peak near 440 Hz."""
        import numpy as _np
        from jarvis.listening.listener import _resample

        src_rate, dst_rate = 48000, 16000
        freq = 440.0
        duration = 0.5
        t = _np.arange(int(src_rate * duration)) / src_rate
        audio = _np.sin(2 * _np.pi * freq * t).astype(_np.float32)

        resampled = _resample(audio, src_rate, dst_rate)

        # FFT to find dominant frequency
        fft_mag = _np.abs(_np.fft.rfft(resampled))
        freqs = _np.fft.rfftfreq(len(resampled), d=1.0 / dst_rate)
        peak_freq = freqs[_np.argmax(fft_mag)]

        assert abs(peak_freq - freq) <= 2.0, f"Peak frequency {peak_freq} Hz not within 2 Hz of {freq} Hz"


class TestSampleRateFallback:
    """Tests for InputStream sample rate fallback on Linux."""

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
        mock_cfg.hot_window_seconds = kwargs.get("hot_window_seconds", 3.0)
        mock_cfg.voice_collect_seconds = kwargs.get("voice_collect_seconds", 2.0)
        mock_cfg.voice_max_collect_seconds = kwargs.get("voice_max_collect_seconds", 60.0)
        mock_cfg.voice_device = kwargs.get("voice_device", None)
        mock_cfg.voice_debug = kwargs.get("voice_debug", False)
        mock_cfg.tune_enabled = kwargs.get("tune_enabled", False)
        return mock_cfg

    def test_fallback_to_native_rate_on_invalid_sample_rate(self, capsys):
        """Falls back to device native rate when 16 kHz is rejected."""
        mock_whisper_model = MagicMock()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model):
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            import queue as q

                            # query_devices returns native rate info
                            device_info = {
                                "name": "ALSA HDA Intel",
                                "max_input_channels": 2,
                                "default_samplerate": 44100.0,
                            }
                            mock_sd.query_devices.side_effect = lambda *args, **kwargs: (
                                device_info if args or kwargs else [device_info]
                            )

                            # First InputStream call rejects 16 kHz, second succeeds
                            mock_stream = MagicMock()
                            mock_stream.active = False
                            mock_stream.__enter__ = MagicMock(return_value=mock_stream)
                            mock_stream.__exit__ = MagicMock(return_value=False)

                            call_count = [0]
                            def input_stream_side_effect(**kw):
                                call_count[0] += 1
                                if call_count[0] == 1:
                                    raise Exception("Invalid sample rate [PaErrorCode -9987]")
                                return mock_stream

                            mock_sd.InputStream.side_effect = input_stream_side_effect

                            from jarvis.listening.listener import VoiceListener

                            mock_db = MagicMock()
                            mock_cfg = self._create_mock_config()
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)

                            # Make the run loop exit immediately
                            get_calls = [0]
                            def fake_get(timeout=0.2):
                                get_calls[0] += 1
                                if get_calls[0] >= 2:
                                    listener._should_stop = True
                                raise q.Empty()

                            listener._audio_q = MagicMock()
                            listener._audio_q.get = fake_get

                            with patch("jarvis.listening.listener.time") as mock_time:
                                mock_time.time.return_value = 0
                                mock_time.sleep = time.sleep
                                listener.run()

                            # InputStream should have been called twice
                            assert mock_sd.InputStream.call_count == 2
                            # Second call should use native 44100 rate
                            second_call_kwargs = mock_sd.InputStream.call_args_list[1][1]
                            assert second_call_kwargs["samplerate"] == 44100
                            # Listener should store the stream rate
                            assert listener._stream_samplerate == 44100

                            captured = capsys.readouterr()
                            assert "44100" in captured.out
                            assert "resampling" in captured.out.lower()

    def test_no_fallback_for_permission_errors(self):
        """Permission errors do not trigger sample rate fallback."""
        mock_whisper_model = MagicMock()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", return_value=mock_whisper_model):
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [
                                {"name": "Test Mic", "max_input_channels": 1}
                            ]
                            mock_sd.InputStream.side_effect = Exception("Device access denied")

                            from jarvis.listening.listener import VoiceListener

                            mock_db = MagicMock()
                            mock_cfg = self._create_mock_config()
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                            listener.run()

                            # Should only have tried once — no fallback
                            assert mock_sd.InputStream.call_count == 1


class TestCorruptedWhisperCacheRecovery:
    """Tests for automatic recovery from corrupted Whisper model cache."""

    def _create_mock_config(self, **kwargs):
        """Create a mock config object with default values."""
        mock_cfg = MagicMock()
        mock_cfg.whisper_model = kwargs.get("whisper_model", "medium")
        mock_cfg.whisper_device = kwargs.get("whisper_device", "auto")
        mock_cfg.whisper_compute_type = kwargs.get("whisper_compute_type", "int8")
        mock_cfg.whisper_backend = kwargs.get("whisper_backend", "faster-whisper")
        mock_cfg.sample_rate = kwargs.get("sample_rate", 16000)
        mock_cfg.vad_enabled = kwargs.get("vad_enabled", True)
        mock_cfg.vad_aggressiveness = kwargs.get("vad_aggressiveness", 2)
        mock_cfg.echo_tolerance = kwargs.get("echo_tolerance", 0.3)
        mock_cfg.echo_energy_threshold = kwargs.get("echo_energy_threshold", 2.0)
        mock_cfg.hot_window_seconds = kwargs.get("hot_window_seconds", 3.0)
        mock_cfg.voice_collect_seconds = kwargs.get("voice_collect_seconds", 2.0)
        mock_cfg.voice_max_collect_seconds = kwargs.get("voice_max_collect_seconds", 60.0)
        mock_cfg.voice_device = kwargs.get("voice_device", None)
        mock_cfg.voice_debug = kwargs.get("voice_debug", False)
        mock_cfg.tune_enabled = kwargs.get("tune_enabled", False)
        return mock_cfg

    def test_corrupted_cache_detected_and_recovered(self, tmp_path):
        """When model.bin is corrupted, cache is cleared and model reloads."""
        mock_whisper_model = MagicMock()

        # Create a fake cache directory to be deleted
        snapshot_dir = tmp_path / "models--Systran--faster-whisper-medium" / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "model.bin").write_bytes(b"corrupted")

        error_msg = f"Unable to open file 'model.bin' in model '{snapshot_dir}'"
        call_count = 0

        def whisper_model_side_effect(model_name, device, compute_type, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError(error_msg)
            return mock_whisper_model

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
                            mock_cfg = self._create_mock_config()
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                            listener.run()

                            # Should have called WhisperModel twice: first corrupted, then retry
                            assert mock_class.call_count == 2
                            assert listener.model == mock_whisper_model

                            # The corrupted snapshot directory should have been deleted
                            assert not snapshot_dir.exists()

    def test_corrupted_cache_retry_also_fails(self, tmp_path):
        """When retry after cache clear also fails, model remains None."""
        # Create a fake cache directory
        snapshot_dir = tmp_path / "models--Systran--faster-whisper-medium" / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "model.bin").write_bytes(b"corrupted")

        error_msg = f"Unable to open file 'model.bin' in model '{snapshot_dir}'"

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel") as mock_class:
                        mock_class.side_effect = RuntimeError(error_msg)

                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]

                            from jarvis.listening.listener import VoiceListener

                            mock_db = MagicMock()
                            mock_cfg = self._create_mock_config()
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                            listener.run()

                            # First attempt + retry = 2 calls
                            assert mock_class.call_count == 2
                            assert listener.model is None

    def test_corrupted_cache_parent_model_dir_deleted(self, tmp_path):
        """Cache cleanup deletes the parent models-- directory, not just snapshot."""
        mock_whisper_model = MagicMock()

        model_dir = tmp_path / "models--Systran--faster-whisper-medium"
        snapshot_dir = model_dir / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "model.bin").write_bytes(b"corrupted")

        # Also create blobs dir (like real HF cache)
        blobs_dir = model_dir / "blobs"
        blobs_dir.mkdir()
        (blobs_dir / "sha256-fake").write_bytes(b"corrupted blob")

        error_msg = f"Unable to open file 'model.bin' in model '{snapshot_dir}'"
        call_count = 0

        def whisper_model_side_effect(model_name, device, compute_type, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError(error_msg)
            return mock_whisper_model

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", side_effect=whisper_model_side_effect):
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                            mock_sd.InputStream.side_effect = Exception("Stop test here")

                            from jarvis.listening.listener import VoiceListener

                            mock_db = MagicMock()
                            mock_cfg = self._create_mock_config()
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                            listener.run()

                            # The entire models-- directory should have been deleted (including blobs)
                            assert not model_dir.exists()

    def test_unparseable_cache_path_shows_manual_instructions(self, capsys):
        """When error path can't be parsed, shows manual cleanup instructions."""
        error_msg = "Unable to open file 'model.bin' somehow"

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel") as mock_class:
                        mock_class.side_effect = RuntimeError(error_msg)

                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]

                            from jarvis.listening.listener import VoiceListener

                            mock_db = MagicMock()
                            mock_cfg = self._create_mock_config()
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                            listener.run()

                            # Should NOT retry (can't parse path)
                            mock_class.assert_called_once()
                            assert listener.model is None

                            # Should show manual cleanup hint
                            captured = capsys.readouterr()
                            assert "whisper model cache" in captured.out.lower()

    def test_rmtree_oserror_prevents_retry(self, tmp_path):
        """When shutil.rmtree raises OSError, model stays None and no retry occurs."""
        snapshot_dir = tmp_path / "models--Systran--faster-whisper-medium" / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "model.bin").write_bytes(b"corrupted")

        error_msg = f"Unable to open file 'model.bin' in model '{snapshot_dir}'"

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel") as mock_class:
                        mock_class.side_effect = RuntimeError(error_msg)

                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]

                            # Make shutil.rmtree raise OSError
                            with patch("shutil.rmtree", side_effect=OSError("Permission denied")):
                                from jarvis.listening.listener import VoiceListener

                                mock_db = MagicMock()
                                mock_cfg = self._create_mock_config()
                                mock_tts = MagicMock()
                                mock_dialogue_memory = MagicMock()

                                listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                                listener.run()

                                # Only the initial attempt — no retry since cache could not be cleared
                                mock_class.assert_called_once()
                                assert listener.model is None

    def test_no_models_ancestor_prevents_cache_clear(self, tmp_path):
        """When error path has no models-- ancestor, cache is not cleared and model stays None."""
        # Create a path without a models-- segment
        plain_dir = tmp_path / "some" / "random" / "path"
        plain_dir.mkdir(parents=True)
        (plain_dir / "model.bin").write_bytes(b"corrupted")

        error_msg = f"Unable to open file 'model.bin' in model '{plain_dir}'"

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel") as mock_class:
                        mock_class.side_effect = RuntimeError(error_msg)

                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]

                            from jarvis.listening.listener import VoiceListener

                            mock_db = MagicMock()
                            mock_cfg = self._create_mock_config()
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                            listener.run()

                            # No retry — _clear_corrupted_whisper_cache returns False
                            mock_class.assert_called_once()
                            assert listener.model is None


class TestWhisperRateLimitRetry:
    """Tests for retry logic when HuggingFace returns 429 Too Many Requests."""

    def _create_mock_config(self, **kwargs):
        """Create a mock config object with default values."""
        mock_cfg = MagicMock()
        mock_cfg.whisper_model = kwargs.get("whisper_model", "medium")
        mock_cfg.whisper_device = kwargs.get("whisper_device", "auto")
        mock_cfg.whisper_compute_type = kwargs.get("whisper_compute_type", "int8")
        mock_cfg.whisper_backend = kwargs.get("whisper_backend", "faster-whisper")
        mock_cfg.sample_rate = kwargs.get("sample_rate", 16000)
        mock_cfg.vad_enabled = kwargs.get("vad_enabled", True)
        mock_cfg.vad_aggressiveness = kwargs.get("vad_aggressiveness", 2)
        mock_cfg.echo_tolerance = kwargs.get("echo_tolerance", 0.3)
        mock_cfg.echo_energy_threshold = kwargs.get("echo_energy_threshold", 2.0)
        mock_cfg.hot_window_seconds = kwargs.get("hot_window_seconds", 3.0)
        mock_cfg.voice_collect_seconds = kwargs.get("voice_collect_seconds", 2.0)
        mock_cfg.voice_max_collect_seconds = kwargs.get("voice_max_collect_seconds", 60.0)
        mock_cfg.voice_device = kwargs.get("voice_device", None)
        mock_cfg.voice_debug = kwargs.get("voice_debug", False)
        mock_cfg.tune_enabled = kwargs.get("tune_enabled", False)
        return mock_cfg

    def test_429_retried_then_succeeds(self):
        """WhisperModel loading retries on 429 and succeeds."""
        mock_whisper_model = MagicMock()
        call_count = 0

        def whisper_model_side_effect(model_name, device, compute_type, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Got: HfHubHTTPError: 429 Too Many Requests for url: https://huggingface.co/api/models/Systran/faster-whisper-medium")
            return mock_whisper_model

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel", side_effect=whisper_model_side_effect) as mock_class:
                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                            mock_sd.InputStream.side_effect = Exception("Stop test here")

                            with patch("jarvis.listening.listener.time.sleep"):  # Skip actual sleep
                                from jarvis.listening.listener import VoiceListener

                                mock_db = MagicMock()
                                mock_cfg = self._create_mock_config()
                                mock_tts = MagicMock()
                                mock_dialogue_memory = MagicMock()

                                listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                                listener.run()

                                assert mock_class.call_count == 2
                                assert listener.model == mock_whisper_model

    def test_429_gives_up_after_max_retries(self):
        """WhisperModel loading gives up after exhausting 429 retries."""
        error_msg = "429 Too Many Requests for url: https://huggingface.co/api/models/Systran/faster-whisper-medium"

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel") as mock_class:
                        mock_class.side_effect = RuntimeError(error_msg)

                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]

                            with patch("jarvis.listening.listener.time.sleep"):
                                from jarvis.listening.listener import VoiceListener

                                mock_db = MagicMock()
                                mock_cfg = self._create_mock_config()
                                mock_tts = MagicMock()
                                mock_dialogue_memory = MagicMock()

                                listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                                listener.run()

                                # Should have retried multiple times then given up
                                assert mock_class.call_count > 1
                                assert listener.model is None

    def test_non_429_error_not_retried(self):
        """Non-rate-limit errors are not retried."""
        error_msg = "Model not found: invalid_model"

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
                with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                    with patch("jarvis.listening.listener.WhisperModel") as mock_class:
                        mock_class.side_effect = RuntimeError(error_msg)

                        with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]

                            from jarvis.listening.listener import VoiceListener

                            mock_db = MagicMock()
                            mock_cfg = self._create_mock_config()
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                            listener.run()

                            # Should have only tried once — no retry
                            mock_class.assert_called_once()
                            assert listener.model is None
