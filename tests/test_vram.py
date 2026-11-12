"""
Tests for the VRAM detection and model recommendation module.

Covers recommendation logic, warning formatting, and the detection
fallback chain (no real GPU calls — we mock at the subprocess and ctypes
level to avoid dependency on actual hardware).
"""

from unittest.mock import patch, MagicMock
import sys

import pytest

from jarvis import config
from jarvis.utils import vram
from jarvis.utils.vram import detect_total_vram_mb, _detect_via_nvidia_smi

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.fixture
def catalogue(monkeypatch):
    """A small catalogue exercises tier boundaries independently of model releases."""
    monkeypatch.setattr(config, 'DEFAULT_CHAT_MODEL', 'balanced')
    monkeypatch.setattr(config, 'DEFAULT_FAST_MODEL', 'balanced')
    monkeypatch.setattr(config, 'SUPPORTED_CHAT_MODELS', {
        'large': {'name': 'Large model', 'vram': '16GB+'},
        'compact': {'name': 'Compact model', 'vram': '2GB+'},
        'balanced': {'name': 'Balanced model', 'vram': '8GB+'},
    })
    table = vram._build_vram_table()
    monkeypatch.setattr(vram, '_MODEL_VRAM_TABLE', table)
    monkeypatch.setattr(vram, '_LOW_VRAM_OPTION', table[0])
    return table


def test_catalogue_derives_sorted_requirements_and_low_memory_flags(catalogue):
    assert catalogue == [
        ('compact', 'Compact model', 2 * 1024, True),
        ('balanced', 'Balanced model', 8 * 1024, False),
        ('large', 'Large model', 16 * 1024, False),
    ]


@pytest.mark.parametrize('budget, expected', [
    (None, 'balanced'), (0, 'compact'), (2047, 'compact'),
    (2048, 'compact'), (4096, 'compact'), (8191, 'compact'),
    (8192, 'balanced'), (16000, 'balanced'), (16383, 'balanced'),
    (16384, 'large'), (32768, 'large'),
])
def test_recommendation_selects_the_largest_fitting_tier(catalogue, budget, expected):
    assert vram.get_recommended_model_id(budget) == expected


@pytest.mark.parametrize('budget, model', [
    (None, 'balanced'), (8 * 1024, 'balanced'), (16 * 1024, 'balanced'),
    (4 * 1024, 'compact'), (4 * 1024, 'unknown'),
])
def test_unknown_or_sufficient_memory_has_no_warning(catalogue, budget, model):
    assert vram.format_vram_warning(budget, model) is None


def test_insufficient_memory_warning_describes_requirements_and_alternative(catalogue):
    budget = 8 * 1024 - 1
    warning = vram.format_vram_warning(budget, 'balanced')
    assert warning is not None
    for detail in (f'{budget} MB', 'balanced', '8192 MB',
                   'compact', 'Compact model', '2048 MB'):
        assert detail in warning


@pytest.mark.parametrize('model, expected', [
    ('compact', 2 * 1024), ('balanced', 8 * 1024),
    ('large', 16 * 1024), ('unknown', None),
])
def test_requirement_lookup_uses_the_catalogue(catalogue, model, expected):
    assert vram.required_vram_mb(model) == expected


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

        result = _detect_via_nvidia_smi()
        assert result == 12288

    @patch("jarvis.utils.vram.subprocess.run")
    def test_parses_multi_gpu_returns_first(self, mock_run):
        """Multi-GPU should return the first GPU's VRAM."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "8192\n16384\n"
        mock_run.return_value = mock_result

        result = _detect_via_nvidia_smi()
        assert result == 8192

    @patch("jarvis.utils.vram.subprocess.run")
    def test_handles_failure(self, mock_run):
        """Non-zero returncode should return None."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        result = _detect_via_nvidia_smi()
        assert result is None

    @patch("jarvis.utils.vram.subprocess.run")
    def test_handles_timeout(self, mock_run):
        """subprocess.TimeoutExpired should return None."""
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired("nvidia-smi", 10)

        result = _detect_via_nvidia_smi()
        assert result is None

    @patch("jarvis.utils.vram.subprocess.run")
    def test_handles_file_not_found(self, mock_run):
        """FileNotFoundError should return None."""
        mock_run.side_effect = FileNotFoundError()

        result = _detect_via_nvidia_smi()
        assert result is None
