"""
PyInstaller runtime hook — runs before any user Python code.

On Windows bundled apps, C:\\Windows\\System32\\onnxruntime.dll (an older,
smaller build) shadows the pip-installed version extracted to _MEIPASS.
We must register the bundled DLL directories *before* onnxruntime is
imported so that the correct library is loaded first.
"""

import os
import sys

if sys.platform == "win32":
    _meipass = getattr(sys, "_MEIPASS", None)
    if _meipass and hasattr(os, "add_dll_directory"):
        # Bundle root (covers sounddevice, etc.)
        try:
            os.add_dll_directory(_meipass)
        except OSError:
            pass

        # onnxruntime native DLL directory
        _ort_capi = os.path.join(_meipass, "onnxruntime", "capi")
        if os.path.isdir(_ort_capi):
            try:
                os.add_dll_directory(_ort_capi)
            except OSError:
                pass
