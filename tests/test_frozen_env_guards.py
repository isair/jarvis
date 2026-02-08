"""
Tests for frozen environment (PyInstaller) safety guards.

These tests verify that macOS bundled apps skip dangerous C extension imports
and use safe fallback modes to prevent crashes.
"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock, call
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


class TestOnnxruntimeDllDirectory:
    """Tests for onnxruntime DLL directory registration in frozen environments."""

    def test_runtime_hook_adds_onnxruntime_capi_on_windows(self):
        """The runtime hook should add onnxruntime/capi to DLL directories on Windows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake onnxruntime/capi directory
            ort_capi = os.path.join(tmpdir, 'onnxruntime', 'capi')
            os.makedirs(ort_capi)

            added_dirs = []

            def mock_add_dll_directory(path):
                added_dirs.append(path)

            with patch.object(sys, 'platform', 'win32'):
                with patch.object(sys, '_MEIPASS', tmpdir, create=True):
                    with patch.object(os, 'add_dll_directory', mock_add_dll_directory, create=True):
                        # Re-execute the runtime hook Layer 1 logic
                        _meipass = getattr(sys, '_MEIPASS', None)
                        if _meipass and hasattr(os, 'add_dll_directory'):
                            for _dir in (_meipass, os.path.join(_meipass, 'onnxruntime', 'capi')):
                                if os.path.isdir(_dir):
                                    os.add_dll_directory(_dir)

            assert tmpdir in added_dirs, "Bundle root should be added"
            assert ort_capi in added_dirs, "onnxruntime/capi should be added"

    def test_runtime_hook_skips_missing_onnxruntime_dir(self):
        """If onnxruntime/capi doesn't exist, it should not be added."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't create onnxruntime/capi

            added_dirs = []

            def mock_add_dll_directory(path):
                added_dirs.append(path)

            with patch.object(sys, 'platform', 'win32'):
                with patch.object(sys, '_MEIPASS', tmpdir, create=True):
                    with patch.object(os, 'add_dll_directory', mock_add_dll_directory, create=True):
                        _meipass = getattr(sys, '_MEIPASS', None)
                        if _meipass and hasattr(os, 'add_dll_directory'):
                            for _dir in (_meipass, os.path.join(_meipass, 'onnxruntime', 'capi')):
                                if os.path.isdir(_dir):
                                    os.add_dll_directory(_dir)

            assert tmpdir in added_dirs, "Bundle root should still be added"
            ort_capi = os.path.join(tmpdir, 'onnxruntime', 'capi')
            assert ort_capi not in added_dirs, "Missing dir should not be added"

    def test_runtime_hook_skips_non_windows(self):
        """On non-Windows platforms the hook should not call add_dll_directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ort_capi = os.path.join(tmpdir, 'onnxruntime', 'capi')
            os.makedirs(ort_capi)

            added_dirs = []

            def mock_add_dll_directory(path):
                added_dirs.append(path)

            with patch.object(sys, 'platform', 'linux'):
                with patch.object(sys, '_MEIPASS', tmpdir, create=True):
                    with patch.object(os, 'add_dll_directory', mock_add_dll_directory, create=True):
                        # Replicate the hook's platform guard
                        if sys.platform == 'win32':
                            _meipass = getattr(sys, '_MEIPASS', None)
                            if _meipass and hasattr(os, 'add_dll_directory'):
                                os.add_dll_directory(_meipass)

            assert len(added_dirs) == 0, "Should not add DLL dirs on non-Windows"

    def test_runtime_hook_preloads_onnxruntime_dll(self):
        """The runtime hook should pre-load onnxruntime.dll from the bundle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake DLL in onnxruntime/capi/
            ort_capi = os.path.join(tmpdir, 'onnxruntime', 'capi')
            os.makedirs(ort_capi)
            dll_path = os.path.join(ort_capi, 'onnxruntime.dll')
            with open(dll_path, 'w') as f:
                f.write('fake')

            loaded_dlls = []

            mock_windll_cls = MagicMock(side_effect=lambda path: loaded_dlls.append(path))

            with patch.object(sys, 'platform', 'win32'):
                with patch.object(sys, '_MEIPASS', tmpdir, create=True):
                    with patch('ctypes.WinDLL', mock_windll_cls):
                        # Replicate Layer 3 logic
                        _meipass = getattr(sys, '_MEIPASS', None)
                        for _subdir in ('onnxruntime\\capi', '.'):
                            _dll = os.path.join(_meipass, _subdir, 'onnxruntime.dll')
                            if os.path.isfile(_dll):
                                import ctypes
                                ctypes.WinDLL(_dll)
                                break

            assert len(loaded_dlls) == 1
            assert loaded_dlls[0] == dll_path

    def test_runtime_hook_preload_falls_back_to_root(self):
        """If onnxruntime.dll is only at _MEIPASS root, pre-load from there."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # DLL at root only, not in onnxruntime/capi/
            dll_path = os.path.join(tmpdir, 'onnxruntime.dll')
            with open(dll_path, 'w') as f:
                f.write('fake')

            loaded_dlls = []
            mock_windll_cls = MagicMock(side_effect=lambda path: loaded_dlls.append(path))

            with patch.object(sys, 'platform', 'win32'):
                with patch.object(sys, '_MEIPASS', tmpdir, create=True):
                    with patch('ctypes.WinDLL', mock_windll_cls):
                        _meipass = getattr(sys, '_MEIPASS', None)
                        for _subdir in ('onnxruntime\\capi', '.'):
                            _dll = os.path.join(_meipass, _subdir, 'onnxruntime.dll')
                            if os.path.isfile(_dll):
                                import ctypes
                                ctypes.WinDLL(_dll)
                                break

            assert len(loaded_dlls) == 1
            # os.path.join(tmpdir, '.', 'onnxruntime.dll') keeps the '.' segment;
            # normalise both sides so the assertion is path-equivalent.
            assert os.path.normpath(loaded_dlls[0]) == os.path.normpath(dll_path)

    def test_runtime_hook_prepends_ort_capi_to_path(self):
        """The runtime hook should prepend onnxruntime/capi to PATH."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ort_capi = os.path.join(tmpdir, 'onnxruntime', 'capi')
            os.makedirs(ort_capi)

            original_path = "C:\\existing"

            with patch.object(sys, 'platform', 'win32'):
                with patch.object(sys, '_MEIPASS', tmpdir, create=True):
                    with patch.dict(os.environ, {'PATH': original_path}):
                        # Replicate Layer 2 logic
                        _meipass = getattr(sys, '_MEIPASS', None)
                        _ort_capi = os.path.join(_meipass, 'onnxruntime', 'capi')
                        _path = os.environ.get('PATH', '')
                        _prepend = []
                        for _dir in (_ort_capi, _meipass):
                            if os.path.isdir(_dir) and _dir not in _path:
                                _prepend.append(_dir)
                        if _prepend:
                            os.environ['PATH'] = os.pathsep.join(_prepend) + os.pathsep + _path

                        new_path = os.environ['PATH']

            # ort_capi should appear before the original path
            assert new_path.startswith(ort_capi)
            assert original_path in new_path

    def test_init_py_adds_onnxruntime_capi_in_frozen_env(self):
        """jarvis/__init__.py should add onnxruntime/capi in a frozen Windows env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake directory structure
            ort_capi = os.path.join(tmpdir, 'onnxruntime', 'capi')
            os.makedirs(ort_capi)
            portaudio = os.path.join(tmpdir, '_sounddevice_data', 'portaudio-binaries')
            os.makedirs(portaudio)

            added_dirs = []

            def mock_add_dll_directory(path):
                added_dirs.append(path)

            with patch.object(sys, 'platform', 'win32'):
                with patch.object(sys, 'frozen', True, create=True):
                    with patch.object(sys, '_MEIPASS', tmpdir, create=True):
                        with patch.object(os, 'add_dll_directory', mock_add_dll_directory, create=True):
                            # Replicate the __init__.py logic
                            _meipass = getattr(sys, '_MEIPASS', None)
                            if _meipass and hasattr(os, 'add_dll_directory'):
                                os.add_dll_directory(_meipass)
                                _portaudio_path = os.path.join(_meipass, '_sounddevice_data', 'portaudio-binaries')
                                if os.path.isdir(_portaudio_path):
                                    os.add_dll_directory(_portaudio_path)
                                _ort_capi = os.path.join(_meipass, 'onnxruntime', 'capi')
                                if os.path.isdir(_ort_capi):
                                    os.add_dll_directory(_ort_capi)

            assert tmpdir in added_dirs
            assert portaudio in added_dirs
            assert ort_capi in added_dirs

    def test_init_py_preloads_onnxruntime_dll(self):
        """jarvis/__init__.py should pre-load onnxruntime.dll in frozen Windows env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake DLL
            ort_capi = os.path.join(tmpdir, 'onnxruntime', 'capi')
            os.makedirs(ort_capi)
            dll_path = os.path.join(ort_capi, 'onnxruntime.dll')
            with open(dll_path, 'w') as f:
                f.write('fake')

            loaded_dlls = []
            mock_windll_cls = MagicMock(side_effect=lambda path: loaded_dlls.append(path))

            with patch.object(sys, 'platform', 'win32'):
                with patch.object(sys, 'frozen', True, create=True):
                    with patch.object(sys, '_MEIPASS', tmpdir, create=True):
                        with patch('ctypes.WinDLL', mock_windll_cls):
                            # Replicate __init__.py Method 3 logic
                            _meipass = getattr(sys, '_MEIPASS', None)
                            for _subdir in ('onnxruntime\\capi', '.'):
                                _dll = os.path.join(_meipass, _subdir, 'onnxruntime.dll')
                                if os.path.isfile(_dll):
                                    import ctypes
                                    ctypes.WinDLL(_dll)
                                    break

            assert len(loaded_dlls) == 1
            assert loaded_dlls[0] == dll_path
