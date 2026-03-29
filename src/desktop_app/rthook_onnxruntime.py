"""PyInstaller runtime hook: register DLL directories on Windows.

When PyInstaller extracts a one-file bundle the native DLLs end up in
subdirectories of the temporary _MEI* folder.  This hook adds those
directories to the DLL search path so native modules can locate their
dependencies.

Covers:
- ONNX Runtime (onnxruntime/capi/)
- NVIDIA CUDA libraries ({app}/cuda/) — installed optionally by the
  Inno Setup installer for GPU-accelerated speech recognition
"""

import os
import sys

if sys.platform == "win32" and getattr(sys, "frozen", False):
    _bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))

    # ONNX Runtime DLLs
    _ort_capi = os.path.join(_bundle_dir, "onnxruntime", "capi")
    if os.path.isdir(_ort_capi):
        try:
            os.add_dll_directory(_ort_capi)
        except (OSError, AttributeError):
            pass

    # NVIDIA CUDA DLLs (cuBLAS + cuDNN, placed by install_cuda.ps1)
    # Use the app's install directory (not _MEIPASS) since CUDA libs are
    # downloaded post-install, not bundled in the PyInstaller archive.
    _app_dir = os.path.dirname(sys.executable)
    _cuda_dir = os.path.join(_app_dir, "cuda")
    if os.path.isdir(_cuda_dir):
        os.environ["PATH"] = _cuda_dir + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(_cuda_dir)
        except (OSError, AttributeError):
            pass
