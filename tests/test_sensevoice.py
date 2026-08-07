"""
Tests for the SenseVoice (FunASR) speech-recognition engine.

These verify the wrapper behaviour with funasr mocked out — the real
funasr import is lazy so lightweight submodules stay import-light.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def _reset_preloaded_engine():
    """Isolate the module-level engine cache between tests."""
    import jarvis.listening.sensevoice as sv

    sv._preloaded_engine = None
    yield
    sv._preloaded_engine = None

import pytest

from jarvis.listening.sensevoice import (
    SenseVoiceEngine,
    SenseVoiceResult,
    SenseVoiceUnavailableError,
    clean_transcript,
    resolve_device,
    resolve_model_ref,
    is_available,
)


class TestCleanTranscript:
    """Rich-tag stripping, language extraction, and no-speech detection."""

    def test_strips_tags_and_extracts_language(self):
        text, lang, no_speech = clean_transcript(
            "<|zh|><|NEUTRAL|><|Speech|><|woitn|>大家好欢迎使用"
        )
        assert text == "大家好欢迎使用"
        assert lang == "zh"
        assert no_speech is False

    def test_english_with_itn_and_emotion(self):
        text, lang, no_speech = clean_transcript(
            "<|en|><|HAPPY|><|Speech|><|withitn|>Hello world, it is 5 o'clock."
        )
        assert text == "Hello world, it is 5 o'clock."
        assert lang == "en"
        assert no_speech is False

    def test_cantonese_and_japanese(self):
        text, lang, _ = clean_transcript("<|yue|><|NEUTRAL|><|Speech|><|woitn|>你好")
        assert lang == "yue"
        assert text == "你好"

        text, lang, _ = clean_transcript("<|ja|><|NEUTRAL|><|Speech|><|woitn|>こんにちは")
        assert lang == "ja"
        assert text == "こんにちは"

    def test_nospeech_tag_detected_and_language_dropped(self):
        text, lang, no_speech = clean_transcript("<|nospeech|><|Event_UNK|>")
        assert text == ""
        assert no_speech is True
        assert lang is None

    def test_empty_input(self):
        assert clean_transcript("") == ("", None, False)
        assert clean_transcript(None) == ("", None, False)

    def test_text_without_tags_passes_through(self):
        text, lang, no_speech = clean_transcript("just plain text")
        assert text == "just plain text"
        assert lang is None
        assert no_speech is False


class TestResolveDevice:
    def test_explicit_preference_passthrough(self):
        assert resolve_device("cuda") == "cuda"
        assert resolve_device("mps") == "mps"
        assert resolve_device("cpu") == "cpu"

    def test_auto_falls_back_to_cpu_without_accelerators(self):
        with patch("jarvis.listening.sensevoice._is_apple_silicon", return_value=False):
            with patch("jarvis.listening.sensevoice._torch_cuda_available", return_value=False):
                assert resolve_device("auto") == "cpu"

    def test_auto_prefers_cuda_when_available(self):
        with patch("jarvis.listening.sensevoice._is_apple_silicon", return_value=False):
            with patch("jarvis.listening.sensevoice._torch_cuda_available", return_value=True):
                assert resolve_device("auto") == "cuda:0"

    def test_auto_uses_mps_on_apple_silicon(self):
        with patch("jarvis.listening.sensevoice._is_apple_silicon", return_value=True):
            with patch("jarvis.listening.sensevoice._torch_cuda_available", return_value=False):
                with patch("jarvis.listening.sensevoice._torch_mps_available", return_value=True):
                    assert resolve_device("auto") == "mps"


class TestResolveModelRef:
    def test_bundled_model_wins_over_configured(self):
        with patch(
            "jarvis.listening.sensevoice.bundled_model_dir",
            return_value="C:/apps/Jarvis/models/SenseVoiceSmall",
        ):
            model, hub = resolve_model_ref("FunAudioLLM/SenseVoiceSmall")
        assert model == "C:/apps/Jarvis/models/SenseVoiceSmall"
        assert hub == "hf"

    def test_default_id_uses_huggingface_hub(self):
        with patch("jarvis.listening.sensevoice.bundled_model_dir", return_value=None):
            model, hub = resolve_model_ref(None)
        assert model == "FunAudioLLM/SenseVoiceSmall"
        assert hub == "hf"

    def test_iic_prefix_uses_modelscope_hub(self):
        with patch("jarvis.listening.sensevoice.bundled_model_dir", return_value=None):
            model, hub = resolve_model_ref("iic/SenseVoiceSmall")
        assert model == "iic/SenseVoiceSmall"
        assert hub == "ms"

    def test_local_path_passes_through(self):
        with patch("jarvis.listening.sensevoice.bundled_model_dir", return_value=None):
            model, hub = resolve_model_ref("D:/models/SenseVoiceSmall")
        assert model == "D:/models/SenseVoiceSmall"
        assert hub == "hf"


class TestBundledModelDir:
    def test_dev_checkout_uses_repo_models_dir(self, monkeypatch, tmp_path):
        """Dev checkouts resolve models/SenseVoiceSmall at the repo root
        (the same dir scripts/fetch_sensevoice_model.py and CI populate)."""
        import jarvis.listening.sensevoice as sv
        repo_root = tmp_path / "repo"
        model_dir = repo_root / "models" / "SenseVoiceSmall"
        model_dir.mkdir(parents=True)
        monkeypatch.setattr(sv, "_REPO_ROOT", repo_root)
        monkeypatch.setattr(sv.sys, "frozen", False, raising=False)
        monkeypatch.delattr(sv.sys, "_MEIPASS", raising=False)
        assert sv.bundled_model_dir() == str(model_dir)

    def test_dev_checkout_returns_none_without_models_dir(self, monkeypatch, tmp_path):
        import jarvis.listening.sensevoice as sv
        repo_root = tmp_path / "repo"
        monkeypatch.setattr(sv, "_REPO_ROOT", repo_root)
        monkeypatch.setattr(sv.sys, "frozen", False, raising=False)
        monkeypatch.delattr(sv.sys, "_MEIPASS", raising=False)
        assert sv.bundled_model_dir() is None

    def test_frozen_app_resolves_next_to_executable(self, monkeypatch, tmp_path):
        import jarvis.listening.sensevoice as sv
        model_dir = tmp_path / "models" / "SenseVoiceSmall"
        model_dir.mkdir(parents=True)
        exe_dir = tmp_path
        monkeypatch.setattr(sv.sys, "frozen", True, raising=False)
        monkeypatch.setattr(sv.sys, "executable", str(exe_dir / "Jarvis.exe"))
        monkeypatch.setattr(sv.sys, "_MEIPASS", None, raising=False)
        assert sv.bundled_model_dir() == str(model_dir)


class TestDefaultModelDir:
    def test_dev_uses_repo_models_dir(self, monkeypatch, tmp_path):
        import jarvis.listening.sensevoice as sv
        monkeypatch.setattr(sv, "_REPO_ROOT", tmp_path / "repo")
        monkeypatch.setattr(sv.sys, "frozen", False, raising=False)
        assert sv.default_model_dir() == tmp_path / "repo" / "models" / "SenseVoiceSmall"

    def test_frozen_uses_config_dir(self, monkeypatch, tmp_path):
        import jarvis.listening.sensevoice as sv
        monkeypatch.setattr(sv.sys, "frozen", True, raising=False)
        with patch(
            "jarvis.config.default_config_path",
            return_value=tmp_path / "config" / "config.json",
        ):
            assert sv.default_model_dir() == tmp_path / "config" / "models" / "SenseVoiceSmall"


class TestDownloadModel:
    def test_downloads_into_local_dir_without_symlinks(self, tmp_path):
        """The runtime download uses snapshot_download(local_dir=...) so
        files are copied, never symlinked (HF cache symlinks fail on
        Windows without Developer Mode, WinError 1314)."""
        import jarvis.listening.sensevoice as sv
        target = tmp_path / "models" / "SenseVoiceSmall"
        with patch(
            "huggingface_hub.snapshot_download",
            return_value=str(target),
        ) as mock_dl:
            result = sv.download_model("FunAudioLLM/SenseVoiceSmall", target_dir=target)
        assert result == str(target)
        mock_dl.assert_called_once_with(
            repo_id="FunAudioLLM/SenseVoiceSmall", local_dir=str(target)
        )

    def test_download_defaults_to_default_model_dir(self, monkeypatch, tmp_path):
        import jarvis.listening.sensevoice as sv
        monkeypatch.setattr(sv, "_REPO_ROOT", tmp_path / "repo")
        monkeypatch.setattr(sv.sys, "frozen", False, raising=False)
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            result = sv.download_model("FunAudioLLM/SenseVoiceSmall")
        assert result == str(tmp_path / "repo" / "models" / "SenseVoiceSmall")
        mock_dl.assert_called_once()
        assert mock_dl.call_args[1]["local_dir"].startswith(str(tmp_path / "repo"))


class TestAvailability:
    def test_unavailable_when_funasr_missing(self):
        with patch("jarvis.listening.sensevoice._get_auto_model", return_value=None):
            assert is_available() is False

    def test_available_when_funasr_present(self):
        with patch("jarvis.listening.sensevoice._get_auto_model", return_value=object):
            assert is_available() is True


class TestSenseVoiceEngineLoad:
    def test_load_constructs_autmodel_with_resolved_local_dir(self):
        """An id that isn't a local dir is downloaded first (no HF cache
        symlinks), and AutoModel receives the local directory."""
        fake_cls = MagicMock()
        fake_cls.return_value = MagicMock(model_path="C:/models/sv")
        with patch("jarvis.listening.sensevoice._get_auto_model", return_value=fake_cls):
            # No bundled weights: the configured id must be downloaded.
            with patch("jarvis.listening.sensevoice.bundled_model_dir", return_value=None):
                with patch(
                    "jarvis.listening.sensevoice.download_model",
                    return_value="C:/downloaded/SenseVoiceSmall",
                ) as mock_dl:
                    engine = SenseVoiceEngine.load("FunAudioLLM/SenseVoiceSmall", device="cpu")

        assert engine.device == "cpu"
        assert engine.model_path == "C:/models/sv"
        mock_dl.assert_called_once_with("FunAudioLLM/SenseVoiceSmall")
        kwargs = fake_cls.call_args[1]
        assert kwargs["model"] == "C:/downloaded/SenseVoiceSmall"
        assert kwargs["device"] == "cpu"
        assert kwargs["hub"] == "hf"
        assert kwargs["disable_update"] is True

    def test_load_uses_local_dir_without_downloading(self, tmp_path):
        """A configured local model directory is passed straight through."""
        model_dir = tmp_path / "SenseVoiceSmall"
        model_dir.mkdir()
        fake_cls = MagicMock()
        fake_cls.return_value = MagicMock(model_path=str(model_dir))
        with patch("jarvis.listening.sensevoice._get_auto_model", return_value=fake_cls):
            # No bundled weights so the configured local dir is used as-is.
            with patch("jarvis.listening.sensevoice.bundled_model_dir", return_value=None):
                with patch(
                    "jarvis.listening.sensevoice.download_model"
                ) as mock_dl:
                    SenseVoiceEngine.load(str(model_dir), device="cpu")
        mock_dl.assert_not_called()
        assert fake_cls.call_args[1]["model"] == str(model_dir)

    def test_load_raises_when_funasr_unavailable(self):
        with patch("jarvis.listening.sensevoice._get_auto_model", return_value=None):
            with pytest.raises(SenseVoiceUnavailableError):
                SenseVoiceEngine.load()

    def test_load_propagates_model_errors(self):
        fake_cls = MagicMock(side_effect=OSError("connection refused"))
        with patch("jarvis.listening.sensevoice._get_auto_model", return_value=fake_cls):
            with patch(
                "jarvis.listening.sensevoice.download_model",
                return_value="C:/downloaded/SenseVoiceSmall",
            ):
                with pytest.raises(OSError):
                    SenseVoiceEngine.load()

    def test_load_retries_on_cpu_when_device_init_fails(self):
        """A CUDA init failure (cuDNN mismatch, VRAM pressure, flaky
        driver) must fall back to CPU so voice and dictation stay usable,
        mirroring the old faster-whisper device fallback chain."""
        import jarvis.listening.sensevoice as sv

        fake_cls = MagicMock(side_effect=[RuntimeError("cuDNN init failed"), MagicMock(model_path="C:/m")])
        with patch("jarvis.listening.sensevoice._get_auto_model", return_value=fake_cls):
            with patch("jarvis.listening.sensevoice._torch_cuda_available", return_value=True):
                with patch("jarvis.listening.sensevoice._is_apple_silicon", return_value=False):
                    with patch(
                        "jarvis.listening.sensevoice.download_model",
                        return_value="C:/downloaded/SenseVoiceSmall",
                    ):
                        engine = SenseVoiceEngine.load()
        assert engine.device == "cpu"
        devices = [c[1]["device"] for c in fake_cls.call_args_list]
        assert devices == ["cuda:0", "cpu"]

    def test_load_no_cpu_retry_when_cpu_init_fails(self):
        """A genuine CPU init failure propagates — there is no lower device."""
        import jarvis.listening.sensevoice as sv

        fake_cls = MagicMock(side_effect=RuntimeError("cpu boom"))
        with patch("jarvis.listening.sensevoice._get_auto_model", return_value=fake_cls):
            with patch("jarvis.listening.sensevoice._torch_cuda_available", return_value=False):
                with patch("jarvis.listening.sensevoice._is_apple_silicon", return_value=False):
                    with patch(
                        "jarvis.listening.sensevoice.download_model",
                        return_value="C:/downloaded/SenseVoiceSmall",
                    ):
                        with pytest.raises(RuntimeError):
                            SenseVoiceEngine.load()


class TestPreloadEngine:
    def test_preload_constructs_and_caches(self):
        """preload_engine() builds the engine once and later load() calls
        with the same resolved model/device reuse it without rebuilding."""
        import jarvis.listening.sensevoice as sv

        fake_cls = MagicMock(return_value=MagicMock(model_path="C:/m"))
        with patch("jarvis.listening.sensevoice._get_auto_model", return_value=fake_cls):
            with patch("jarvis.listening.sensevoice._torch_cuda_available", return_value=False):
                with patch("jarvis.listening.sensevoice._is_apple_silicon", return_value=False):
                    with patch(
                        "jarvis.listening.sensevoice.download_model",
                        return_value="C:/downloaded/SenseVoiceSmall",
                    ):
                        preloaded = sv.preload_engine()
                        again = sv.SenseVoiceEngine.load()
        assert preloaded is again
        assert fake_cls.call_count == 1

    def test_load_ignores_cache_on_mismatched_device(self):
        """A different device or model still builds a fresh engine."""
        import jarvis.listening.sensevoice as sv

        fake_cls = MagicMock(return_value=MagicMock(model_path="C:/m"))
        with patch("jarvis.listening.sensevoice._get_auto_model", return_value=fake_cls):
            with patch("jarvis.listening.sensevoice._torch_cuda_available", return_value=False):
                with patch("jarvis.listening.sensevoice._is_apple_silicon", return_value=False):
                    with patch(
                        "jarvis.listening.sensevoice.download_model",
                        return_value="C:/downloaded/SenseVoiceSmall",
                    ):
                        sv.preload_engine()
                        sv.SenseVoiceEngine.load(device="cuda:0")
        assert fake_cls.call_count == 2


class TestQuietStartup:
    def test_load_suppresses_funasr_print_chatter(self, capsys):
        """funasr's module-level prints (the misleading ffmpeg notice and
        the version banner) must not reach the daemon console."""
        fake_cls = MagicMock()

        def noisy(*args, **kwargs):
            print("Notice: ffmpeg is not installed. torchaudio is used to load audio")
            print("funasr version: 1.4.1.")
            return MagicMock(model_path="C:/m")

        fake_cls.side_effect = noisy
        with patch("jarvis.listening.sensevoice._get_auto_model", return_value=fake_cls):
            with patch("jarvis.listening.sensevoice._torch_cuda_available", return_value=False):
                with patch("jarvis.listening.sensevoice._is_apple_silicon", return_value=False):
                    with patch(
                        "jarvis.listening.sensevoice.download_model",
                        return_value="C:/downloaded/SenseVoiceSmall",
                    ):
                        SenseVoiceEngine.load(device="cpu")
        out = capsys.readouterr().out
        assert "ffmpeg is not installed" not in out
        assert "funasr version" not in out

    def test_quiet_funasr_loggers_sets_levels(self):
        """funasr (and any child logger) is silenced to CRITICAL; torch and
        friends keep WARNING so real warnings still surface."""
        import jarvis.listening.sensevoice as sv
        import logging

        logging.getLogger("funasr.auto.auto_model").setLevel(logging.DEBUG)
        logging.getLogger("torch").setLevel(logging.DEBUG)
        sv._quiet_funasr_loggers()
        assert logging.getLogger("funasr").level == logging.CRITICAL
        assert logging.getLogger("funasr.auto.auto_model").level == logging.CRITICAL
        assert logging.getLogger("torch").level == logging.WARNING
        assert logging.getLogger("torchaudio").level == logging.WARNING


class TestSenseVoiceEngineTranscribe:
    def _make_engine(self, generate_result=None, generate_error=None):
        auto = MagicMock()
        if generate_error is not None:
            auto.generate.side_effect = generate_error
        else:
            auto.generate.return_value = generate_result or [{"key": "k", "text": ""}]
        engine = SenseVoiceEngine.__new__(SenseVoiceEngine)
        engine._auto = auto
        engine.device = "cpu"
        return engine

    def test_transcribe_parses_result(self):
        engine = self._make_engine(
            [{"key": "k", "text": "<|en|><|NEUTRAL|><|Speech|><|woitn|>what is the weather"}]
        )
        result = engine.transcribe(MagicMock())
        assert isinstance(result, SenseVoiceResult)
        assert result.text == "what is the weather"
        assert result.language == "en"
        assert result.no_speech is False
        # generate receives a dict cache and language auto
        call_kwargs = engine._auto.generate.call_args[1]
        assert call_kwargs["language"] == "auto"
        assert call_kwargs["use_itn"] is True
        assert call_kwargs["cache"] == {}

    def test_transcribe_nospeech(self):
        engine = self._make_engine([{"key": "k", "text": "<|nospeech|><|Event_UNK|>"}])
        result = engine.transcribe(MagicMock())
        assert result.text == ""
        assert result.no_speech is True

    def test_transcribe_error_returns_empty(self):
        engine = self._make_engine(generate_error=RuntimeError("boom"))
        result = engine.transcribe(MagicMock())
        assert result.text == ""
        assert result.no_speech is False

    def test_transcribe_empty_result(self):
        engine = self._make_engine([])
        result = engine.transcribe(MagicMock())
        assert result.text == ""

    def test_warmup_runs_non_silent_audio(self):
        import numpy as np
        engine = self._make_engine(
            [{"key": "k", "text": "<|en|><|NEUTRAL|><|Speech|><|woitn|>warm"}]
        )
        with patch("jarvis.listening.sensevoice.np", np):
            engine.warmup(16000)
        audio = engine._auto.generate.call_args[1]["input"]
        assert audio.shape[0] == 16000
        assert not (audio == 0).all(), "warmup should not use silent audio"
