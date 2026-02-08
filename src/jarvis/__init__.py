"""
Jarvis Voice Assistant

A modular voice assistant with conversation memory, tool integration,
and natural language processing capabilities.
"""

# =============================================================================
# PyInstaller Windows fix - MUST be at the very top before any audio imports
# =============================================================================
# When bundled with PyInstaller on Windows, sounddevice uses ctypes to locate
# PortAudio. The DLLs are extracted to sys._MEIPASS but won't be found by default.
#
# Python 3.8+ on Windows changed DLL loading behavior - PATH is no longer searched
# for DLLs loaded via ctypes. We must use os.add_dll_directory() instead.
#
# See: https://github.com/pyinstaller/pyinstaller/issues/7065
# See: https://github.com/spatialaudio/python-sounddevice/issues/378
# See: https://docs.python.org/3/whatsnew/3.8.html#ctypes
import os as _os
import sys as _sys

if getattr(_sys, 'frozen', False) and _sys.platform == 'win32':
    _meipass = getattr(_sys, '_MEIPASS', None)
    if _meipass:
        # Method 1: os.add_dll_directory (Python 3.8+, the proper solution)
        # This explicitly adds the directory to the DLL search path for ctypes
        if hasattr(_os, 'add_dll_directory'):
            try:
                _os.add_dll_directory(_meipass)
                # Also add _sounddevice_data/portaudio-binaries if it exists
                _portaudio_path = _os.path.join(_meipass, '_sounddevice_data', 'portaudio-binaries')
                if _os.path.isdir(_portaudio_path):
                    _os.add_dll_directory(_portaudio_path)
                # Add onnxruntime/capi for bundled onnxruntime.dll — prevents
                # C:\Windows\System32\onnxruntime.dll (old version) from being
                # loaded instead of the pip-installed one.
                _ort_capi = _os.path.join(_meipass, 'onnxruntime', 'capi')
                if _os.path.isdir(_ort_capi):
                    _os.add_dll_directory(_ort_capi)
            except Exception:
                pass

        # Method 2: Modify PATH (legacy fallback, helps with subprocess spawning)
        _path = _os.environ.get('PATH', '')
        if _meipass not in _path:
            _os.environ['PATH'] = _meipass + _os.pathsep + _path
        del _path
    del _meipass
del _os, _sys
# =============================================================================

from .config import load_settings


def get_version() -> tuple[str, str]:
    """Get the application version and release channel.

    Returns:
        tuple of (version_string, channel) where channel is 'stable' or 'develop'.
        When running from source without a build, returns ('dev-local', 'develop').
    """
    try:
        from ._version import VERSION, RELEASE_CHANNEL
        return VERSION, RELEASE_CHANNEL
    except ImportError:
        return "dev-local", "develop"


def main() -> None:
    """Lazy entrypoint to avoid importing heavy modules at package import time.

    Importing `jarvis.daemon` here prevents it from being added to sys.modules
    during package import, which avoids runpy warnings when executing
    `python -m jarvis.daemon`.
    """
    from .daemon import main as _main
    _main()

__all__ = ["main", "load_settings", "get_version"]
