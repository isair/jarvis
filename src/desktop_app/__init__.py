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
