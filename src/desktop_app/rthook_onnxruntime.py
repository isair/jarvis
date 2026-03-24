"""PyInstaller runtime hook: register ONNX Runtime DLL directory on Windows.

When PyInstaller extracts a one-file bundle the native DLLs end up in a
subdirectory (onnxruntime/capi/) of the temporary _MEI* folder.  This hook
adds that directory to the DLL search path so onnxruntime_pybind11_state.pyd
can locate onnxruntime.dll and onnxruntime_providers_shared.dll.
"""

import os
import sys

if sys.platform == "win32" and getattr(sys, "frozen", False):
    _bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    _ort_capi = os.path.join(_bundle_dir, "onnxruntime", "capi")
    if os.path.isdir(_ort_capi):
        try:
            os.add_dll_directory(_ort_capi)
        except (OSError, AttributeError):
            pass
