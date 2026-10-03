"""
Tests for the VRAM detection and model recommendation module.

Covers recommendation logic, warning formatting, and the detection
fallback chain (no real GPU calls — we mock at the subprocess and ctypes
level to avoid dependency on actual hardware).
"""

from unittest.mock import patch, MagicMock
import sys

import pytest

from jarvis.utils.vram import (
    detect_total_vram_mb,
    get_recommended_model_id,
    format_vram_warning,
    required_vram_mb,
    _MODEL_VRAM_TABLE,
    _LOW_VRAM_OPTION,
)


class TestRecommendation:
    """Model recommendation based on VRAM."""

    def test_recommends_low_vram_below_8gb(self):
        """6 GB VRAM should recommend the low-VRAM option (qwen3.5:0.8b)."""
        rec = get_recommended_model_id(6000)
        assert rec == _LOW_VRAM_OPTION[0]

    def test_recommends_low_vram_at_4gb(self):
        """Exactly 4 GB VRAM should still recommend the low-VRAM option."""
        rec = get_recommended_model_id(4096)
        assert rec == _LOW_VRAM_OPTION[0]

    def test_recommends_default_at_8gb(self):
        """8 GB VRAM should recommend gemma4:e2b (the default)."""
        rec = get_recommended_model_id(8192)
        assert rec == "gemma4:e2b"

    def test_recommends_default_above_8gb(self):
        """16 GB VRAM should recommend gemma4:e2b."""
        rec = get_recommended_model_id(16000)
        assert rec == "gemma4:e2b"

    def test_recommends_default_when_unknown(self):
        """Unknown VRAM (None) should recommend the default safely."""
        rec = get_recommended_model_id(None)
        assert rec == "gemma4:e2b"


class TestFormatWarning:
    """Warning string generation."""

    def test_no_warning_when_vram_unknown(self):
        """Unknown VRAM should return None (no warning)."""
        warn = format_vram_warning(None, "gemma4:e2b")
        assert warn is None

    def test_no_warning_when_vram_sufficient(self):
        """Sufficient VRAM (16 GB) should return None."""
        warn = format_vram_warning(16000, "gemma4:e2b")
        assert warn is None

    def test_warning_when_vram_insufficient(self):
        """6 GB VRAM with gemma4:e2b (needs 8 GB) should return a warning string."""
        warn = format_vram_warning(6000, "gemma4:e2b")
        assert warn is not None
        assert "6000 MB" in warn
        assert "gemma4:e2b" in warn
        assert _LOW_VRAM_OPTION[0] in warn

    def test_no_warning_for_low_vram_model_on_low_gpu(self):
        """6 GB VRAM with qwen3.5:0.8b (needs 2 GB) should NOT warn."""
        warn = format_vram_warning(4096, _LOW_VRAM_OPTION[0])
        assert warn is None

    def test_warning_for_unknown_model_returns_none(self):
        """An unknown model ID should return None (can't judge)."""
        warn = format_vram_warning(4000, "nonexistent:model")
        assert warn is None

    def test_warning_mentions_alternative(self):
        """The warning string should mention the low-VRAM alternative by name."""
        warn = format_vram_warning(6000, "gemma4:e2b")
        assert warn is not None
        assert _LOW_VRAM_OPTION[1] in warn  # Low-VRAM display name


class TestRequiredVRAM:
    """VRAM requirement lookup."""

    def test_required_vram_for_known_model(self):
        """gemma4:e2b should require 8192 MB."""
        assert required_vram_mb("gemma4:e2b") == 8192

    def test_required_vram_for_low_vram_model(self):
        """qwen3.5:0.8b should require 2048 MB."""
        assert required_vram_mb(_LOW_VRAM_OPTION[0]) == 2048

    def test_required_vram_for_unknown_model(self):
        """Unknown model should return None."""
        assert required_vram_mb("nonexistent:99b") is None


class TestDetectionMocked:
    """Detection with mocked backends (no real GPU required)."""

    @patch("jarvis.utils.vram._detect_via_dxgi")
    def test_dxgi_takes_precedence_on_windows(self, mock_dxgi):
        """DXGI should be tried first on Windows."""
        mock_dxgi.return_value = 8192
        with patch.object(sys, "platform", "win32"):
            result = detect_total_vram_mb()
        assert result == 8192
        mock_dxgi.assert_called_once()

    @patch("jarvis.utils.vram._detect_via_dxgi")
    @patch("jarvis.utils.vram._detect_via_nvidia_smi")
    def test_falls_back_to_nvidia_smi_when_dxgi_fails(
        self, mock_smi, mock_dxgi
    ):
        """When DXGI returns None, nvidia-smi should be tried."""
        mock_dxgi.return_value = None
        mock_smi.return_value = 4096
        with patch.object(sys, "platform", "win32"):
            result = detect_total_vram_mb()
        assert result == 4096

    @patch("jarvis.utils.vram._detect_via_dxgi")
    @patch("jarvis.utils.vram._detect_via_nvidia_smi")
    def test_returns_none_when_all_fail(
        self, mock_smi, mock_dxgi
    ):
        """When all detectors return None, the result should be None."""
        mock_dxgi.return_value = None
        mock_smi.return_value = None
        with patch.object(sys, "platform", "win32"):
            result = detect_total_vram_mb()
        assert result is None

    @patch("jarvis.utils.vram._detect_via_nvidia_smi")
    def test_uses_nvidia_smi_on_non_windows(self, mock_smi):
        """On non-Windows, nvidia-smi is the primary detector."""
        mock_smi.return_value = 8192
        with patch.object(sys, "platform", "linux"):
            result = detect_total_vram_mb()
        assert result == 8192


class TestNvidiaSmiDetection:
    """nvidia-smi parsing logic (no real subprocess)."""

    @patch("jarvis.utils.vram.subprocess.run")
    def test_parses_single_gpu(self, mock_run):
        """Single GPU should parse correctly."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "12288\n"
        mock_run.return_value = mock_result

        result = _detect_via_nvidia_smi_impl()
        assert result == 12288

    @patch("jarvis.utils.vram.subprocess.run")
    def test_parses_multi_gpu_returns_first(self, mock_run):
        """Multi-GPU should return the first GPU's VRAM."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "8192\n16384\n"
        mock_run.return_value = mock_result

        result = _detect_via_nvidia_smi_impl()
        assert result == 8192

    @patch("jarvis.utils.vram.subprocess.run")
    def test_handles_failure(self, mock_run):
        """Non-zero returncode should return None."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        result = _detect_via_nvidia_smi_impl()
        assert result is None

    @patch("jarvis.utils.vram.subprocess.run")
    def test_handles_timeout(self, mock_run):
        """subprocess.TimeoutExpired should return None."""
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired("nvidia-smi", 10)

        result = _detect_via_nvidia_smi_impl()
        assert result is None

    @patch("jarvis.utils.vram.subprocess.run")
    def test_handles_file_not_found(self, mock_run):
        """FileNotFoundError should return None."""
        mock_run.side_effect = FileNotFoundError()

        result = _detect_via_nvidia_smi_impl()
        assert result is None


def _detect_via_nvidia_smi_impl():
    """Inline copy of _detect_via_nvidia_smi for test isolation."""
    from jarvis.utils.vram import _detect_via_nvidia_smi
    return _detect_via_nvidia_smi()
