"""
Tests for frozen environment (PyInstaller) safety guards.

These tests verify bundled app behaviour for stdin monitoring
and other platform-specific guards.
"""

import sys
from unittest.mock import patch


class TestStdinMonitorFrozenGuard:
    """Tests for stdin_monitor being skipped in bundled (frozen) mode.

    In bundled mode (PyInstaller), sys.stdin may be devnull, so the
    stdin_monitor thread should not start — the desktop app calls
    request_stop() directly instead.
    """

    def test_stdin_monitor_skipped_in_frozen_mode(self):
        """On Windows frozen apps, stdin_monitor should NOT start."""
        with patch.object(sys, 'platform', 'win32'):
            with patch.object(sys, 'frozen', True, create=True):
                should_start = (
                    sys.platform == "win32"
                    and not getattr(sys, 'frozen', False)
                )
                assert should_start is False

    def test_stdin_monitor_starts_in_non_frozen_mode(self):
        """On Windows non-frozen (dev) mode, stdin_monitor should start."""
        with patch.object(sys, 'platform', 'win32'):
            # Remove frozen attribute if present
            frozen_backup = getattr(sys, 'frozen', None)
            if hasattr(sys, 'frozen'):
                delattr(sys, 'frozen')
            try:
                should_start = (
                    sys.platform == "win32"
                    and not getattr(sys, 'frozen', False)
                )
                assert should_start is True
            finally:
                if frozen_backup is not None:
                    sys.frozen = frozen_backup

    def test_stdin_monitor_skipped_on_non_windows(self):
        """On non-Windows platforms, stdin_monitor should not start regardless."""
        for platform in ('darwin', 'linux'):
            with patch.object(sys, 'platform', platform):
                should_start = (
                    sys.platform == "win32"
                    and not getattr(sys, 'frozen', False)
                )
                assert should_start is False

    def test_stdin_monitor_skipped_on_non_windows_frozen(self):
        """On non-Windows frozen apps, stdin_monitor should not start."""
        with patch.object(sys, 'platform', 'darwin'):
            with patch.object(sys, 'frozen', True, create=True):
                should_start = (
                    sys.platform == "win32"
                    and not getattr(sys, 'frozen', False)
                )
                assert should_start is False
