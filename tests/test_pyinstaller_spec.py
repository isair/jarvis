"""
Tests for the PyInstaller spec file (jarvis_desktop.spec).

Verifies that the bundling configuration includes all required modules
so the packaged app works correctly on all platforms.
"""

import re
from pathlib import Path

import pytest

# Read the spec file once for all tests
SPEC_PATH = Path(__file__).parent.parent / "jarvis_desktop.spec"
SPEC_CONTENT = SPEC_PATH.read_text()


def _code_lines_only(content: str) -> str:
    """Return spec file content with comment lines stripped."""
    return '\n'.join(
        l for l in content.splitlines() if not l.strip().startswith('#')
    )


class TestMLXWhisperBundling:
    """Ensure mlx-whisper is bundled on macOS so Apple Silicon gets the fast backend."""

    def test_spec_does_not_collect_all_mlx_submodules(self):
        """collect_submodules('mlx') must NOT be used — it causes nanobind double-registration."""
        code_only = _code_lines_only(SPEC_CONTENT)
        matches = re.findall(r"collect_submodules\('mlx'\)", code_only)
        assert len(matches) == 0, "collect_submodules('mlx') found — use specific mlx submodules instead"

    def test_spec_lists_specific_mlx_submodules(self):
        """Only the specific mlx submodules used by mlx_whisper should be listed."""
        for mod in ["'mlx'", "'mlx.core'", "'mlx._reprlib_fix'", "'mlx.nn'", "'mlx.utils'"]:
            assert mod in SPEC_CONTENT, f"Missing mlx submodule: {mod}"

    def test_spec_collects_mlx_whisper_submodules_on_darwin(self):
        """The spec should use collect_submodules for mlx_whisper on macOS."""
        assert "collect_submodules('mlx_whisper')" in SPEC_CONTENT

    def test_spec_collects_mlx_whisper_data_files_on_darwin(self):
        """The spec should use collect_data_files for mlx_whisper on macOS."""
        assert "collect_data_files('mlx_whisper')" in SPEC_CONTENT

    def test_mlx_whisper_in_hiddenimports(self):
        """mlx_whisper should appear in hiddenimports as a static entry."""
        assert "'mlx_whisper'" in SPEC_CONTENT

    def test_mlx_collection_guarded_by_darwin(self):
        """MLX collection should only run on macOS (sys.platform == 'darwin')."""
        darwin_block_start = SPEC_CONTENT.index("if sys.platform == 'darwin':")
        mlx_whisper_collect = SPEC_CONTENT.index("collect_submodules('mlx_whisper')")
        else_block = SPEC_CONTENT.index("else:\n    hiddenimports_mlx = []")
        assert darwin_block_start < mlx_whisper_collect < else_block

    def test_mlx_collection_has_fallback(self):
        """If mlx is not installed, the spec should gracefully set an empty list."""
        assert "hiddenimports_mlx = []" in SPEC_CONTENT

    def test_mlx_not_in_excludes(self):
        """mlx should not appear in the excludes list."""
        excludes_start = SPEC_CONTENT.index("_excludes = [")
        excludes_end = SPEC_CONTENT.index("]", excludes_start) + 1
        excludes_block = SPEC_CONTENT[excludes_start:excludes_end]
        assert "mlx" not in excludes_block.lower()

    def test_mlx_not_in_excluded_binary_patterns(self):
        """mlx should not be caught by excluded binary patterns."""
        patterns_start = SPEC_CONTENT.index("excluded_binary_patterns = [")
        patterns_end = SPEC_CONTENT.index("]", patterns_start) + 1
        patterns_block = SPEC_CONTENT[patterns_start:patterns_end]
        assert "mlx" not in patterns_block.lower()

    def test_torch_whisper_excluded_from_mlx_submodules(self):
        """torch_whisper should be filtered out — it needs PyTorch which is excluded."""
        assert "torch_whisper" in SPEC_CONTENT
        assert "'torch_whisper' not in m" in SPEC_CONTENT


class TestMLXWhisperDependencies:
    """Ensure mlx-whisper's runtime deps (scipy, numba, tiktoken) are bundled."""

    def test_scipy_not_excluded_on_darwin(self):
        """scipy should NOT be in the excludes on macOS — mlx_whisper needs it."""
        assert "if sys.platform != 'darwin'" in SPEC_CONTENT
        assert "_excludes.append('scipy')" in SPEC_CONTENT
        # Verify scipy is NOT in the main excludes list (only appended conditionally)
        excludes_start = SPEC_CONTENT.index("_excludes = [")
        excludes_end = SPEC_CONTENT.index("]", excludes_start) + 1
        excludes_block = SPEC_CONTENT[excludes_start:excludes_end]
        assert "'scipy'" not in excludes_block

    def test_tiktoken_collected(self):
        """tiktoken submodules should be collected for mlx_whisper's tokeniser."""
        assert "collect_submodules('tiktoken')" in SPEC_CONTENT
        assert "collect_data_files('tiktoken')" in SPEC_CONTENT

    def test_numba_collected(self):
        """numba submodules should be collected for mlx_whisper's timing module."""
        assert "collect_submodules('numba')" in SPEC_CONTENT

    def test_metallib_bundled(self):
        """The Metal shader library must be bundled for mlx.core to initialise."""
        assert "mlx.metallib" in SPEC_CONTENT
        assert "'mlx/lib'" in SPEC_CONTENT
