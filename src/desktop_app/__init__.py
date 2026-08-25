"""
Jarvis Desktop App - System Tray Application

A cross-platform system tray app for controlling the Jarvis voice assistant.
Supports Windows, Ubuntu (Linux), and macOS.
"""

from __future__ import annotations
import sys
import os

# Fix OpenBLAS threading crash in bundled apps
# Must be set before numpy is imported (via faster-whisper, etc.)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

# When launched as `python -m src.desktop_app` the package is registered in
# sys.modules as 'src.desktop_app', but all internal imports use the bare
# 'desktop_app' name (matching the PYTHONPATH=src invocation used by the run
# scripts).  Alias ourselves and add src/ to sys.path so both styles work
# without triggering a double-import of this __init__.
import importlib
_this_module = sys.modules[__name__]   # 'src.desktop_app' or 'desktop_app'
if __name__ != 'desktop_app':
    sys.modules.setdefault('desktop_app', _this_module)
    _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

# Re-export main for entry point
from desktop_app.app import main

# Re-export commonly used components for backwards compatibility
from desktop_app.app import (
    get_crash_paths,
    check_previous_crash,
    mark_session_started,
    mark_session_clean_exit,
    setup_crash_logging,
    show_crash_report_dialog,
    check_model_support,
    show_unsupported_model_dialog,
    acquire_single_instance_lock,
    JarvisSystemTray,
    LogViewerWindow,
    MemoryViewerWindow,
    LogSignals,
)

__all__ = [
    'main',
    'get_crash_paths',
    'check_previous_crash',
    'mark_session_started',
    'mark_session_clean_exit',
    'setup_crash_logging',
    'show_crash_report_dialog',
    'check_model_support',
    'show_unsupported_model_dialog',
    'acquire_single_instance_lock',
    'JarvisSystemTray',
    'LogViewerWindow',
    'MemoryViewerWindow',
    'LogSignals',
]
