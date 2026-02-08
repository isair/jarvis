"""
PyInstaller runtime hook — runs before any user Python code.

On Windows bundled apps, C:\\Windows\\System32\\onnxruntime.dll (an older,
smaller build shipped by DirectML/Windows ML) shadows the pip-installed
version extracted to _MEIPASS.  We must ensure the bundled copy is loaded
first.

Three layers of defence:
1. os.add_dll_directory()  — adds to the safe DLL search for LoadLibraryEx
2. PATH prepend            — legacy fallback for bare LoadLibrary calls
3. ctypes.WinDLL pre-load  — loads the correct DLL by absolute path so
                              Windows never searches for it at all
"""

import os
import sys

if sys.platform == "win32":
    _meipass = getattr(sys, "_MEIPASS", None)
    if _meipass:
        _ort_capi = os.path.join(_meipass, "onnxruntime", "capi")

        # ---- Layer 1: Register DLL search directories (Python 3.8+) ----
        if hasattr(os, "add_dll_directory"):
            for _dir in (_meipass, _ort_capi):
                if os.path.isdir(_dir):
                    try:
                        os.add_dll_directory(_dir)
                    except OSError:
                        pass

        # ---- Layer 2: Prepend to PATH ----
        _path = os.environ.get("PATH", "")
        _prepend = []
        for _dir in (_ort_capi, _meipass):
            if os.path.isdir(_dir) and _dir not in _path:
                _prepend.append(_dir)
        if _prepend:
            os.environ["PATH"] = os.pathsep.join(_prepend) + os.pathsep + _path

        # ---- Layer 3: Pre-load bundled onnxruntime.dll ----
        # Once a DLL is loaded, Windows reuses it by module name for all
        # subsequent LoadLibrary("onnxruntime.dll") calls — the System32
        # copy is never reached.  Try onnxruntime/capi/ first (the pip
        # package layout), then the bundle root (flattened layout).
        import ctypes
        for _subdir in ("onnxruntime\\capi", "."):
            _dll = os.path.join(_meipass, _subdir, "onnxruntime.dll")
            if os.path.isfile(_dll):
                try:
                    ctypes.WinDLL(_dll)
                except OSError:
                    pass
                break
