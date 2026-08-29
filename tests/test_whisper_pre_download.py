"""
Tests for ``_pre_download_whisper_model_safely`` — the subprocess-isolated
Whisper model download that protects the main process from SIGABRT.
"""

from unittest.mock import patch, MagicMock, ANY
from pathlib import Path
import pytest


# ── _pre_download_whisper_model_safely unit tests ─────────────────────────


class TestPreDownloadSubprocess:
    """Direct unit tests for the subprocess-isolated download."""

    @staticmethod
    def _call(model_name: str = "small"):
        from jarvis.listening.listener import _pre_download_whisper_model_safely
        return _pre_download_whisper_model_safely(model_name)

    def test_returns_true_when_model_already_cached(self, tmp_path):
        """When the model is already cached on disk, returns True without
        spawning a subprocess."""
        # Create a fake cache directory tree that mimics a cached model
        cache_root = tmp_path / "models--Systran--faster-whisper-small"
        snapshots = cache_root / "snapshots" / "abc123"
        snapshots.mkdir(parents=True)
        (snapshots / "model.bin").write_bytes(b"fake model data")
        (snapshots / "config.json").write_bytes(b"{}")

        import subprocess as _real_subprocess
        with patch.object(_real_subprocess, "run") as mock_run:
            with patch(
                "jarvis.listening.listener._get_hf_cache_path",
                return_value=tmp_path,
            ):
                result = self._call("small")
                assert result is True
                # No subprocess should be spawned — model is already cached
                mock_run.assert_not_called()

    def test_returns_true_on_successful_subprocess(self):
        """Returns True when the subprocess exits with code 0 (model NOT
        already cached, but download succeeds)."""
        import subprocess as _real_subprocess
        with patch.object(_real_subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0

            # Make sure cache check returns False so subprocess runs
            with patch(
                "jarvis.listening.listener._is_whisper_model_cached",
                return_value=False,
            ):
                result = self._call("small")
                assert result is True

                import sys
                assert mock_run.call_args[0][0][0] == sys.executable
                script = mock_run.call_args[0][0][2]
                assert "download_model" in script
                assert "small" in script
                assert "from faster_whisper.utils import download_model" in script
                assert "ImportError" in script

    def test_returns_false_on_subprocess_failure(self):
        """Returns False when the subprocess exits with non-zero."""
        import subprocess as _real_subprocess
        with patch.object(_real_subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 1

            with patch(
                "jarvis.listening.listener._is_whisper_model_cached",
                return_value=False,
            ):
                result = self._call("base")
                assert result is False

    def test_returns_false_on_sigabrt(self):
        """Returns False when the subprocess is killed by SIGABRT (-6)."""
        import subprocess as _real_subprocess
        with patch.object(_real_subprocess, "run") as mock_run:
            mock_run.return_value.returncode = -6

            with patch(
                "jarvis.listening.listener._is_whisper_model_cached",
                return_value=False,
            ):
                result = self._call("small")
                assert result is False

    def test_returns_false_on_timeout(self):
        """Returns False when the subprocess times out."""
        import subprocess as _real_subprocess
        with patch.object(_real_subprocess, "run") as mock_run:
            mock_run.side_effect = _real_subprocess.TimeoutExpired(
                cmd="test", timeout=300
            )

            with patch(
                "jarvis.listening.listener._is_whisper_model_cached",
                return_value=False,
            ):
                result = self._call("tiny")
                assert result is False

    def test_returns_false_on_subprocess_startup_failure(self):
        """Returns False when subprocess.run raises OSError."""
        import subprocess as _real_subprocess
        with patch.object(_real_subprocess, "run") as mock_run:
            mock_run.side_effect = OSError("sys.executable not found")

            with patch(
                "jarvis.listening.listener._is_whisper_model_cached",
                return_value=False,
            ):
                result = self._call("small")
                assert result is False

    def test_returns_false_when_faster_whisper_unavailable(self):
        """Returns False immediately when FASTER_WHISPER_AVAILABLE is False."""
        with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", False):
            result = self._call("small")
            assert result is False

    def test_returns_false_when_faster_whisper_unavailable_and_model_empty(self):
        """Returns False when FASTER_WHISPER_AVAILABLE is False and model is empty."""
        with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", False):
            result = self._call("")
            assert result is False


# ── Cache check helper tests ─────────────────────────────────────────────


class TestWhisperModelCached:
    """Tests for the filesystem cache check helper."""

    def test_detects_cached_model_by_cache_dir(self, tmp_path):
        """Returns True when the HF cache directory exists with snapshots."""
        cache_root = tmp_path / "models--Systran--faster-whisper-small"
        snapshots = cache_root / "snapshots" / "abc123"
        snapshots.mkdir(parents=True)
        (snapshots / "model.bin").write_bytes(b"fake")

        with patch(
            "jarvis.listening.listener._get_hf_cache_path",
            return_value=tmp_path,
        ):
            from jarvis.listening.listener import _is_whisper_model_cached
            assert _is_whisper_model_cached("small") is True

    def test_returns_false_for_missing_model(self, tmp_path):
        """Returns False when the cache directory doesn't exist."""
        with patch(
            "jarvis.listening.listener._get_hf_cache_path",
            return_value=tmp_path,
        ):
            from jarvis.listening.listener import _is_whisper_model_cached
            assert _is_whisper_model_cached("small") is False

    def test_returns_false_for_empty_cache_dir(self, tmp_path):
        """Returns False when the cache directory exists but has no snapshots."""
        cache_root = tmp_path / "models--Systran--faster-whisper-small"
        cache_root.mkdir(parents=True)
        # No snapshots subdirectory

        with patch(
            "jarvis.listening.listener._get_hf_cache_path",
            return_value=tmp_path,
        ):
            from jarvis.listening.listener import _is_whisper_model_cached
            assert _is_whisper_model_cached("small") is False

    def test_resolves_full_repo_id_directly(self, tmp_path):
        """Models specified as full repo IDs (containing '/') work correctly."""
        cache_root = tmp_path / "models--CustomOrg--custom-whisper"
        snapshots = cache_root / "snapshots" / "abc123"
        snapshots.mkdir(parents=True)
        (snapshots / "model.bin").write_bytes(b"fake")

        with patch(
            "jarvis.listening.listener._get_hf_cache_path",
            return_value=tmp_path,
        ):
            from jarvis.listening.listener import _is_whisper_model_cached
            assert _is_whisper_model_cached("CustomOrg/custom-whisper") is True

    def test_returns_false_for_unknown_model_name(self, tmp_path):
        """Returns False when the model name isn't in the known map and
        doesn't contain '/'."""
        with patch(
            "jarvis.listening.listener._get_hf_cache_path",
            return_value=tmp_path,
        ):
            from jarvis.listening.listener import _is_whisper_model_cached
            assert _is_whisper_model_cached("nonexistent-model") is False


# ── Integration: local_files_only parameter passing ─────────────────────


class TestLocalFilesOnlyParameter:
    """Check that ``local_files_only`` is correctly forwarded based on
    the pre-download result."""

    def test_passes_true_when_pre_download_succeeds(self):
        """``local_files_only=True`` when pre-download returns True."""
        mock_whisper_model = MagicMock()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch(
                "jarvis.listening.listener._pre_download_whisper_model_safely",
                return_value=True,
            ):
                with patch(
                    "jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True
                ):
                    with patch(
                        "jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False
                    ):
                        with patch(
                            "jarvis.listening.listener.WhisperModel",
                            return_value=mock_whisper_model,
                        ) as mock_class:
                            with patch(
                                "jarvis.listening.listener.sd"
                            ) as mock_sd:
                                mock_sd.query_devices.return_value = [
                                    {"name": "Test Mic", "max_input_channels": 1}
                                ]
                                mock_sd.InputStream.side_effect = Exception(
                                    "Stop test here"
                                )

                                from jarvis.listening.listener import VoiceListener

                                listener = VoiceListener(
                                    MagicMock(),
                                    MagicMock(),
                                    MagicMock(),
                                    MagicMock(),
                                )
                                listener.run()

                                mock_class.assert_called_once()
                                assert (
                                    mock_class.call_args[1]["local_files_only"]
                                    is True
                                )

    def test_passes_false_when_pre_download_fails(self):
        """``local_files_only=False`` when pre-download returns False."""
        mock_whisper_model = MagicMock()

        with patch("jarvis.listening.listener.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch(
                "jarvis.listening.listener._pre_download_whisper_model_safely",
                return_value=False,
            ):
                with patch(
                    "jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True
                ):
                    with patch(
                        "jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False
                    ):
                        with patch(
                            "jarvis.listening.listener.WhisperModel",
                            return_value=mock_whisper_model,
                        ) as mock_class:
                            with patch(
                                "jarvis.listening.listener.sd"
                            ) as mock_sd:
                                mock_sd.query_devices.return_value = [
                                    {"name": "Test Mic", "max_input_channels": 1}
                                ]
                                mock_sd.InputStream.side_effect = Exception(
                                    "Stop test here"
                                )

                                from jarvis.listening.listener import VoiceListener

                                listener = VoiceListener(
                                    MagicMock(),
                                    MagicMock(),
                                    MagicMock(),
                                    MagicMock(),
                                )
                                listener.run()

                                mock_class.assert_called_once()
                                assert (
                                    mock_class.call_args[1]["local_files_only"]
                                    is False
                                )
