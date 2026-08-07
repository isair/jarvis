"""
Tests for voice listener module.

These tests verify the SenseVoice (FunASR) engine loading and the
listener's audio/echo behaviour.
"""

from unittest.mock import patch, MagicMock, call
import time
import pytest


def _create_mock_config(**kwargs):
    """Create a mock config object with default values for voice listener tests."""
    mock_cfg = MagicMock()
    mock_cfg.sensevoice_model = kwargs.get("sensevoice_model", "FunAudioLLM/SenseVoiceSmall")
    mock_cfg.sensevoice_device = kwargs.get("sensevoice_device", "auto")
    mock_cfg.sensevoice_min_audio_duration = kwargs.get("sensevoice_min_audio_duration", 0.3)
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

def _sensevoice_engine():
    """Return a stubbed SenseVoiceEngine whose load succeeds and transcribes nothing.

    Lets integration-style tests run the listener's audio loop without a
    real funasr model: warmup is a no-op and any utterance transcribes to
    empty (so no intent-judge dispatch is triggered).
    """
    engine = MagicMock()
    engine.warmup = MagicMock()
    engine.transcribe.return_value = MagicMock(text="", language=None, no_speech=False)
    return engine


def _mock_sensevoice_load():
    """Patches making the listener's SenseVoice load path succeed."""
    return (
        patch("jarvis.listening.listener.is_sensevoice_available", return_value=True),
        patch("jarvis.listening.listener.SenseVoiceEngine.load", return_value=_sensevoice_engine()),
    )


class TestSenseVoiceEngineLoad:
    """Tests for the listener's SenseVoice engine load path."""

    def _run_listener(self, cfg):
        from jarvis.listening.listener import VoiceListener

        # Patch sys.platform to skip the Windows-only mic-permission probe so
        # the mocked InputStream exception below acts as "stop after load".
        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            listener = VoiceListener(MagicMock(), cfg, MagicMock(), MagicMock())
            listener.run()
            return listener

    def test_loads_engine_and_prints_ready(self, capsys):
        avail_patch, load_patch = _mock_sensevoice_load()
        with avail_patch, load_patch:
            with patch("jarvis.listening.listener.sd") as mock_sd:
                mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                mock_sd.InputStream.side_effect = Exception("Stop test here")

                mock_cfg = _create_mock_config()
                mock_cfg.llm_chat_model = ""
                mock_cfg.fast_model = ""
                listener = self._run_listener(mock_cfg)

        assert listener.engine is not None
        assert "SenseVoice" in capsys.readouterr().out

    def test_model_and_device_passed_from_config(self, capsys):
        avail_patch, load_patch = _mock_sensevoice_load()
        with avail_patch, load_patch as mock_load:
            with patch("jarvis.listening.listener.sd") as mock_sd:
                mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                mock_sd.InputStream.side_effect = Exception("Stop test here")

                mock_cfg = _create_mock_config(
                    sensevoice_model="iic/SenseVoiceSmall", sensevoice_device="cpu"
                )
                mock_cfg.llm_chat_model = ""
                mock_cfg.fast_model = ""
                self._run_listener(mock_cfg)

        mock_load.assert_called_once_with(model="iic/SenseVoiceSmall", device="cpu")

    def test_unavailable_prints_install_hint(self, capsys):
        with patch("jarvis.listening.listener.is_sensevoice_available", return_value=False):
            mock_cfg = _create_mock_config()
            listener = self._run_listener(mock_cfg)

        assert listener.engine is None
        out = capsys.readouterr().out
        assert "pip install funasr" in out

    def test_load_error_prints_failure(self, capsys):
        with patch("jarvis.listening.listener.is_sensevoice_available", return_value=True):
            with patch(
                "jarvis.listening.listener.SenseVoiceEngine.load",
                side_effect=RuntimeError("connection refused"),
            ):
                mock_cfg = _create_mock_config()
                listener = self._run_listener(mock_cfg)

        assert listener.engine is None
        out = capsys.readouterr().out
        assert "Failed to load SenseVoice model" in out

    def test_warmup_runs_after_load(self, capsys):
        avail_patch, load_patch = _mock_sensevoice_load()
        with avail_patch, load_patch:
            with patch("jarvis.listening.listener.sd") as mock_sd:
                mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                mock_sd.InputStream.side_effect = Exception("Stop test here")

                mock_cfg = _create_mock_config()
                mock_cfg.llm_chat_model = ""
                mock_cfg.fast_model = ""
                listener = self._run_listener(mock_cfg)

        engine = listener.engine
        engine.warmup.assert_called_once()
        # Warmup receives the listener's sample rate
        assert engine.warmup.call_args[0][0] == 16000


class TestSenseVoiceNoSpeechGate:
    """Tests for the <|nospeech|> rejection in _finalize_utterance."""

    def _make_listener(self, result):
        import numpy as _np
        from jarvis.listening.listener import VoiceListener
        from jarvis.listening.sensevoice import SenseVoiceResult

        listener = object.__new__(VoiceListener)
        listener.cfg = MagicMock()
        listener.cfg.voice_debug = False
        listener.cfg.sensevoice_min_audio_duration = 0.3
        listener.cfg.sample_rate = 16000
        listener._samplerate = 16000
        listener._stream_samplerate = 16000
        listener.engine = MagicMock()
        listener.engine.transcribe.return_value = result
        listener.transcribe_lock = MagicMock()
        listener.state_manager = MagicMock()
        listener.echo_detector = MagicMock()
        listener.echo_detector._utterance_start_time = 100.0
        listener.echo_detector._last_tts_finish_time = 0.0
        listener.echo_detector.echo_tolerance = 0.3
        listener._transcript_buffer = MagicMock()
        listener.tts = MagicMock()
        listener.tts.is_speaking.return_value = False
        listener._first_utterance = True
        listener._utterance_frames = [_np.ones(16000, dtype=_np.float32) * 0.1]
        listener._pre_roll = []
        listener._is_repetitive_hallucination = lambda text: False
        listener._process_transcript = MagicMock()
        listener._calculate_audio_energy = lambda frames: 1.0
        return listener

    def test_nospeech_utterance_rejected(self):
        from jarvis.listening.sensevoice import SenseVoiceResult

        listener = self._make_listener(
            SenseVoiceResult(text="", language=None, no_speech=True)
        )
        listener._finalize_utterance()

        listener._transcript_buffer.add.assert_not_called()
        listener._process_transcript.assert_not_called()
        listener.state_manager.check_hot_window_expiry.assert_called_once()

    def test_speech_utterance_passed_and_language_recorded(self):
        from jarvis.listening.sensevoice import SenseVoiceResult

        listener = self._make_listener(
            SenseVoiceResult(text="what is the weather", language="en", no_speech=False)
        )
        listener._finalize_utterance()

        assert listener._last_detected_language == "en"
        listener._transcript_buffer.add.assert_called_once()
        listener._process_transcript.assert_called_once()
        # The transcribe call went through the shared lock
        listener.engine.transcribe.assert_called_once()






class TestRepetitiveHallucinationDetection:
    """Tests for SenseVoice hallucination detection."""

    def _create_mock_listener(self):
        """Create a VoiceListener instance for testing."""
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




class TestRepetitiveHallucinationDetectionExtended:
    """Additional tests for SenseVoice hallucination detection."""

    def _create_mock_listener(self):
        """Create a VoiceListener instance for testing."""
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

    def test_device_name_printed_on_linux(self, capsys):
        """Device name is printed on Linux, not just Windows."""
        avail_patch, load_patch = _mock_sensevoice_load()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with avail_patch, load_patch:
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
                            mock_cfg = _create_mock_config()
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                            listener.run()

                            captured = capsys.readouterr()
                            assert "🎤" in captured.out
                            assert "Linux PulseAudio Mic" in captured.out

    def test_device_name_printed_on_macos(self, capsys):
        """Device name is printed on macOS, not just Windows."""
        avail_patch, load_patch = _mock_sensevoice_load()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "darwin"
            with avail_patch, load_patch:
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
                            mock_cfg = _create_mock_config()
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                            listener.run()

                            captured = capsys.readouterr()
                            assert "🎤" in captured.out
                            assert "MacBook Pro Microphone" in captured.out


class TestCrossPlatformAudioHealthWarning:
    """Tests for cross-platform audio health monitoring."""

    def test_health_warning_fires_on_linux(self, capsys):
        """Audio health warning fires on Linux when no audio received after 5s."""
        avail_patch, load_patch = _mock_sensevoice_load()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with avail_patch, load_patch:
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
                            mock_cfg = _create_mock_config()
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

                            # time.time() is called for the LLM-warmup baseline,
                            # then for _audio_start_time (both baselines), then in
                            # the loop for the health check (needs to be 6s later)
                            _base = time.time()
                            time_calls = [0]

                            def advancing_time():
                                time_calls[0] += 1
                                # First two calls set baselines (LLM warmup + audio start)
                                if time_calls[0] <= 2:
                                    return _base
                                # Subsequent calls return 6s later
                                return _base + 6

                            with patch("jarvis.listening.listener.time") as mock_time:
                                mock_time.time.side_effect = advancing_time
                                mock_time.sleep = time.sleep

                                # No LLM warmup threads: keeps time.time() call
                                # counting deterministic (the warmup join would
                                # No LLM warmup threads: keeps time.time() call
                                # counting deterministic (the warmup join would
                                # consume mock values racy in a full-suite run).
                                with patch(
                                    "jarvis.listening.listener.VoiceListener._start_llm_warmup",
                                    return_value=[],
                                ):
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

    def test_fallback_to_native_rate_on_invalid_sample_rate(self, capsys):
        """Falls back to device native rate when 16 kHz is rejected."""
        avail_patch, load_patch = _mock_sensevoice_load()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with avail_patch, load_patch:
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
                            mock_cfg = _create_mock_config()
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
        avail_patch, load_patch = _mock_sensevoice_load()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with avail_patch, load_patch:
                with patch("jarvis.listening.listener.sd") as mock_sd:
                            mock_sd.query_devices.return_value = [
                                {"name": "Test Mic", "max_input_channels": 1}
                            ]
                            mock_sd.InputStream.side_effect = Exception("Device access denied")

                            from jarvis.listening.listener import VoiceListener

                            mock_db = MagicMock()
                            mock_cfg = _create_mock_config()
                            mock_tts = MagicMock()
                            mock_dialogue_memory = MagicMock()

                            listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)
                            listener.run()

                            # Should only have tried once — no fallback
                            assert mock_sd.InputStream.call_count == 1






def _make_listener_for_warmup(
    chat_model: str = "llama3.1",
    judge_model: str | None = "gemma4:e2b",
    embed_model: str = "",
    base_url: str = "http://127.0.0.1:11434",
):
    """Construct a VoiceListener with enough stubs to exercise warmup only."""
    with patch("jarvis.listening.listener.sd") as mock_sd:
                mock_sd.query_devices.return_value = [
                    {"name": "Test Mic", "max_input_channels": 1}
                ]

                from jarvis.listening.listener import VoiceListener
                from jarvis.listening.intent_judge import IntentJudge, IntentJudgeConfig

                mock_cfg = _create_mock_config()
                mock_cfg.ollama_chat_model = chat_model
                mock_cfg.llm_chat_model = chat_model
                mock_cfg.embedding_model = embed_model
                mock_cfg.ollama_base_url = base_url
                mock_cfg.llm_tools_timeout_sec = 8.0
                mock_cfg.fast_model = judge_model or ""
                mock_cfg.intent_judge_timeout_sec = 10.0
                mock_cfg.intent_judge_thinking_enabled = False
                mock_cfg.wake_word = "jarvis"
                mock_cfg.wake_aliases = []

                listener = VoiceListener(MagicMock(), mock_cfg, MagicMock(), MagicMock())

                if judge_model is not None:
                    listener._intent_judge = IntentJudge(
                        IntentJudgeConfig(model=judge_model, cfg=mock_cfg)
                    )
                else:
                    listener._intent_judge = None
                return listener


class TestLlmWarmup:
    """Tests for VoiceListener._start_llm_warmup orchestration."""

    def test_spawns_threads_for_chat_and_distinct_judge(self):
        """Two threads when chat and judge point at different models."""
        listener = _make_listener_for_warmup(
            chat_model="llama3.1", judge_model="gemma4:e2b"
        )
        with patch(
            "jarvis.listening.listener.warm_up_chat_model", return_value=True
        ) as chat_warm, patch(
            "jarvis.listening.intent_judge.warm_up_chat_model", return_value=True
        ) as judge_warm:
            threads = listener._start_llm_warmup()
            for t in threads:
                t.join(timeout=2.0)

        assert len(threads) == 2
        assert chat_warm.call_args.args[1] == "llama3.1"
        assert judge_warm.call_args.args[1] == "gemma4:e2b"
        assert listener._llm_warmup_results["chat"] == ("llama3.1", True)
        assert listener._llm_warmup_results["judge"] == ("gemma4:e2b", True)

    def test_deduplicates_when_chat_and_judge_share_model(self):
        """One warmup covers both roles when models match."""
        listener = _make_listener_for_warmup(
            chat_model="llama3.1", judge_model="llama3.1"
        )
        with patch("jarvis.listening.listener.warm_up_chat_model", return_value=True) as warm:
            threads = listener._start_llm_warmup()
            for t in threads:
                t.join(timeout=2.0)

        assert len(threads) == 1
        assert warm.call_count == 1
        assert listener._llm_warmup_results["chat"] == ("llama3.1", True)
        assert listener._llm_warmup_results["judge"] == ("llama3.1", True)

    def test_judge_only_when_no_chat_model(self):
        """Judge still warms when chat model is absent."""
        listener = _make_listener_for_warmup(chat_model="", judge_model="gemma4:e2b")
        with patch(
            "jarvis.listening.intent_judge.warm_up_chat_model", return_value=True
        ) as warm:
            threads = listener._start_llm_warmup()
            for t in threads:
                t.join(timeout=2.0)

        assert len(threads) == 1
        assert warm.call_count == 1
        assert listener._llm_warmup_results["judge"] == ("gemma4:e2b", True)
        assert "chat" not in listener._llm_warmup_results

    def test_empty_when_nothing_to_warm(self):
        """No threads returned when chat and judge are both absent."""
        listener = _make_listener_for_warmup(chat_model="", judge_model=None)
        threads = listener._start_llm_warmup()
        assert threads == []
        assert listener._llm_warmup_results == {}

    def test_records_failure_from_helper(self):
        """A False return from the helper shows up in the results dict."""
        listener = _make_listener_for_warmup(
            chat_model="llama3.1", judge_model="gemma4:e2b"
        )
        with patch(
            "jarvis.listening.listener.warm_up_chat_model", return_value=False
        ), patch(
            "jarvis.listening.intent_judge.warm_up_chat_model", return_value=False
        ):
            threads = listener._start_llm_warmup()
            for t in threads:
                t.join(timeout=2.0)

        assert listener._llm_warmup_results["chat"] == ("llama3.1", False)
        assert listener._llm_warmup_results["judge"] == ("gemma4:e2b", False)

    def test_warms_embed_model_separately(self):
        """Embed model gets its own warmup thread when distinct from chat."""
        listener = _make_listener_for_warmup(
            chat_model="llama3.1", embed_model="nomic-embed-text"
        )
        with patch(
            "jarvis.listening.listener.warm_up_chat_model", return_value=True
        ) as chat_warm, patch(
            "jarvis.listening.intent_judge.warm_up_chat_model", return_value=True
        ) as judge_warm, patch(
            "jarvis.listening.listener.get_embedding_backend"
        ) as mock_get_embed:
            mock_embed_backend = MagicMock()
            mock_embed_backend.embed.return_value = [0.1, 0.2, 0.3]
            mock_get_embed.return_value = mock_embed_backend

            threads = listener._start_llm_warmup()
            for t in threads:
                t.join(timeout=2.0)

        assert len(threads) == 3
        assert chat_warm.call_args.args[1] == "llama3.1"
        assert mock_embed_backend.embed.call_args.args == ("ping", "nomic-embed-text")
        assert listener._llm_warmup_results["embed"] == ("nomic-embed-text", True)

    def test_skips_embed_warmup_when_empty(self):
        """No embed warmup thread when embedding_model is not configured."""
        listener = _make_listener_for_warmup(
            chat_model="llama3.1", embed_model=""
        )
        with patch(
            "jarvis.listening.listener.warm_up_chat_model", return_value=True
        ), patch(
            "jarvis.listening.intent_judge.warm_up_chat_model", return_value=True
        ), patch(
            "jarvis.listening.listener.get_embedding_backend"
        ) as mock_get_embed:
            threads = listener._start_llm_warmup()
            for t in threads:
                t.join(timeout=2.0)

        assert len(threads) == 2
        assert not mock_get_embed.called
        assert "embed" not in listener._llm_warmup_results

    def test_embed_warmup_records_failure(self):
        """None from embed() surfaces in the results dict as False."""
        listener = _make_listener_for_warmup(
            chat_model="llama3.1", embed_model="nomic-embed-text"
        )
        with patch(
            "jarvis.listening.listener.warm_up_chat_model", return_value=True
        ), patch(
            "jarvis.listening.intent_judge.warm_up_chat_model", return_value=True
        ), patch(
            "jarvis.listening.listener.get_embedding_backend"
        ) as mock_get_embed:
            mock_embed_backend = MagicMock()
            mock_embed_backend.embed.return_value = None
            mock_get_embed.return_value = mock_embed_backend

            threads = listener._start_llm_warmup()
            for t in threads:
                t.join(timeout=2.0)

        assert listener._llm_warmup_results["embed"] == ("nomic-embed-text", False)

    def test_embed_warmup_stores_failure_on_backend_init_exception(self):
        """An exception in get_embedding_backend is caught and stored as False."""
        listener = _make_listener_for_warmup(
            chat_model="llama3.1", embed_model="nomic-embed-text"
        )
        with patch(
            "jarvis.listening.listener.warm_up_chat_model", return_value=True
        ), patch(
            "jarvis.listening.intent_judge.warm_up_chat_model", return_value=True
        ), patch(
            "jarvis.listening.listener.get_embedding_backend"
        ) as mock_get_embed:
            mock_get_embed.side_effect = RuntimeError("backend init crashed")

            threads = listener._start_llm_warmup()
            for t in threads:
                t.join(timeout=2.0)

        assert listener._llm_warmup_results["embed"] == ("nomic-embed-text", False)

    def test_embed_warmup_stores_failure_on_embed_exception(self):
        """An exception in embed() is caught and stored as False."""
        listener = _make_listener_for_warmup(
            chat_model="llama3.1", embed_model="nomic-embed-text"
        )
        with patch(
            "jarvis.listening.listener.warm_up_chat_model", return_value=True
        ), patch(
            "jarvis.listening.intent_judge.warm_up_chat_model", return_value=True
        ), patch(
            "jarvis.listening.listener.get_embedding_backend"
        ) as mock_get_embed:
            mock_embed_backend = MagicMock()
            mock_embed_backend.embed.side_effect = ConnectionError("server down")
            mock_get_embed.return_value = mock_embed_backend

            threads = listener._start_llm_warmup()
            for t in threads:
                t.join(timeout=2.0)

        assert listener._llm_warmup_results["embed"] == ("nomic-embed-text", False)








class TestWeatherBannerExample:
    """Tests for the adaptive weather example in the startup banner."""

    def _make_listener(self, **cfg_overrides):
        from unittest.mock import MagicMock
        from jarvis.listening.listener import VoiceListener

        cfg = MagicMock()
        cfg.wake_word = cfg_overrides.get("wake_word", "jarvis")
        cfg.location_enabled = cfg_overrides.get("location_enabled", True)
        cfg.location_auto_detect = cfg_overrides.get("location_auto_detect", True)
        cfg.location_ip_address = cfg_overrides.get("location_ip_address", None)

        listener = object.__new__(VoiceListener)
        listener.cfg = cfg
        return listener

    def test_plain_form_when_auto_detect_enabled(self):
        """Plain 'How's the weather' example when auto-detect is on and database is present."""
        from unittest.mock import patch
        listener = self._make_listener(location_enabled=True, location_auto_detect=True)
        with patch("jarvis.listening.listener.is_location_available", return_value=True):
            result = listener._weather_example("Jarvis")
        assert result == "\"How's the weather, Jarvis?\""

    def test_plain_form_when_manual_ip_configured(self):
        """Plain form when auto-detect is off but a manual IP is set and database is present."""
        from unittest.mock import patch
        listener = self._make_listener(
            location_enabled=True,
            location_auto_detect=False,
            location_ip_address="1.2.3.4",
        )
        with patch("jarvis.listening.listener.is_location_available", return_value=True):
            result = listener._weather_example("Jarvis")
        assert result == "\"How's the weather, Jarvis?\""

    def test_city_placeholder_when_location_disabled(self):
        """City placeholder form when location is explicitly disabled."""
        listener = self._make_listener(location_enabled=False)
        result = listener._weather_example("Jarvis")
        assert result == "\"How's the weather in [your city], Jarvis?\""

    def test_city_placeholder_when_no_location_source(self):
        """City placeholder form when auto-detect is off and no manual IP is set."""
        listener = self._make_listener(
            location_enabled=True,
            location_auto_detect=False,
            location_ip_address=None,
        )
        result = listener._weather_example("Jarvis")
        assert result == "\"How's the weather in [your city], Jarvis?\""

    def test_city_placeholder_when_database_not_available(self):
        """City placeholder form when GeoLite2 database is missing even if config enables location."""
        from unittest.mock import patch
        listener = self._make_listener(location_enabled=True, location_auto_detect=True)
        with patch("jarvis.listening.listener.is_location_available", return_value=False):
            result = listener._weather_example("Jarvis")
        assert result == "\"How's the weather in [your city], Jarvis?\""

    def test_wake_title_reflected_in_example(self):
        """Wake word title is correctly used in the example string."""
        from unittest.mock import patch
        with patch("jarvis.listening.listener.is_location_available", return_value=True):
            listener = self._make_listener(location_enabled=True, location_auto_detect=True)
            assert "Helix?" in listener._weather_example("Helix")

        listener2 = self._make_listener(location_enabled=False)
        assert "Helix?" in listener2._weather_example("Helix")
