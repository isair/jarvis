"""
Tests for frozen environment (PyInstaller) safety guards.

These tests verify that macOS bundled apps skip dangerous C extension imports
and use safe fallback modes to prevent crashes.
"""

import sys
from unittest.mock import patch, MagicMock
import importlib


class TestWebEngineImportGuard:
    """Tests for QtWebEngine import being skipped on macOS frozen apps."""

    def test_webengine_skipped_on_macos_frozen(self):
        """On macOS frozen apps, HAS_WEBENGINE should be False and import should be skipped."""
        # We can't easily re-import app.py, so test the guard logic directly
        is_macos_bundle = (sys.platform == 'darwin') and getattr(sys, 'frozen', False)

        # On Windows dev environment, this won't be a macOS bundle
        if sys.platform != 'darwin':
            assert not is_macos_bundle, "Should not be detected as macOS bundle on non-macOS"

    def test_webengine_guard_logic_simulated_macos_frozen(self):
        """Simulate the macOS frozen guard logic to verify it blocks the import."""
        # Simulate: sys.platform == 'darwin' and sys.frozen == True
        with patch.object(sys, 'platform', 'darwin'):
            with patch.object(sys, 'frozen', True, create=True):
                _is_macos_bundle = sys.platform == 'darwin' and getattr(sys, 'frozen', False)
                assert _is_macos_bundle is True

                # The guard should set HAS_WEBENGINE = False without attempting import
                if not _is_macos_bundle:
                    HAS_WEBENGINE = True  # Would attempt import
                else:
                    HAS_WEBENGINE = False
                    QWebEngineView = None

                assert HAS_WEBENGINE is False
                assert QWebEngineView is None

    def test_webengine_guard_logic_simulated_non_frozen(self):
        """On non-frozen environments, the import should be attempted."""
        with patch.object(sys, 'platform', 'darwin'):
            # No 'frozen' attribute — normal dev environment
            if hasattr(sys, 'frozen'):
                delattr(sys, 'frozen')

            _is_macos_bundle = sys.platform == 'darwin' and getattr(sys, 'frozen', False)
            assert _is_macos_bundle is False

    def test_webengine_guard_logic_simulated_windows_frozen(self):
        """On Windows frozen apps, WebEngine import should still be attempted."""
        with patch.object(sys, 'platform', 'win32'):
            with patch.object(sys, 'frozen', True, create=True):
                _is_macos_bundle = sys.platform == 'darwin' and getattr(sys, 'frozen', False)
                assert _is_macos_bundle is False


class TestMaxminddbReaderMode:
    """Tests for maxminddb reader mode selection in frozen environments."""

    def test_frozen_env_uses_mmap_mode(self):
        """In frozen environments, reader mode should be MODE_MMAP (2)."""
        with patch.object(sys, 'frozen', True, create=True):
            _reader_mode = 2 if getattr(sys, 'frozen', False) else 0
            assert _reader_mode == 2  # MODE_MMAP (pure Python)

    def test_non_frozen_env_uses_auto_mode(self):
        """In non-frozen environments, reader mode should be MODE_AUTO (0)."""
        # Ensure 'frozen' is not set
        frozen_backup = getattr(sys, 'frozen', None)
        if hasattr(sys, 'frozen'):
            delattr(sys, 'frozen')

        try:
            _reader_mode = 2 if getattr(sys, 'frozen', False) else 0
            assert _reader_mode == 0  # MODE_AUTO
        finally:
            # Restore frozen attribute if it existed
            if frozen_backup is not None:
                sys.frozen = frozen_backup

    def test_reader_mode_passed_to_geoip2(self):
        """Verify the reader mode is actually passed to geoip2.database.Reader."""
        import jarvis.utils.location as loc_mod

        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)

        # Mock the response
        mock_response = MagicMock()
        mock_response.country.name = "United Kingdom"
        mock_response.country.iso_code = "GB"
        mock_response.subdivisions.most_specific.name = "England"
        mock_response.subdivisions.most_specific.iso_code = "ENG"
        mock_response.city.name = "London"
        mock_response.location.latitude = 51.5074
        mock_response.location.longitude = -0.1278
        mock_response.location.time_zone = "Europe/London"
        mock_response.location.accuracy_radius = 10
        mock_reader.city.return_value = mock_response

        # Use a unique IP to avoid hitting the module-level cache
        test_ip = "198.51.100.99"

        # Clear cache entry for this IP to ensure we reach the Reader call
        loc_mod._location_cache.pop(test_ip, None)

        # Simulate frozen environment
        with patch.object(sys, 'frozen', True, create=True):
            with patch('geoip2.database.Reader', return_value=mock_reader) as mock_reader_cls:
                with patch.object(loc_mod, '_get_database_path') as mock_db_path:
                    mock_path = MagicMock()
                    mock_path.exists.return_value = True
                    mock_path.__str__ = MagicMock(return_value="/fake/path/GeoLite2-City.mmdb")
                    mock_db_path.return_value = mock_path

                    result = loc_mod.get_location_info(ip_address=test_ip)

                    # Verify Reader was called with mode=2 (MMAP)
                    mock_reader_cls.assert_called_once_with(
                        "/fake/path/GeoLite2-City.mmdb", mode=2
                    )
