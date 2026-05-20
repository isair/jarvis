"""Thread-safe screen capture helpers (Windows/Qt-safe)."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from typing import Callable, Optional

from .debug import debug_log

# Optional hook: desktop app registers this to run capture on the Qt main thread.
_main_thread_runner: Optional[Callable[[Callable[[], bool]], bool]] = None

_MAX_CAPTURE_SIDE = 1920


def register_main_thread_capture_runner(
    runner: Optional[Callable[[Callable[[], bool]], bool]],
) -> None:
    """Register ``runner(work) -> bool`` that executes ``work`` on the UI thread."""
    global _main_thread_runner
    _main_thread_runner = runner


def _downscale_image(im, max_side: int = _MAX_CAPTURE_SIDE):
    from PIL import Image

    width, height = im.size
    if max(width, height) <= max_side:
        return im
    scale = max_side / float(max(width, height))
    return im.resize(
        (int(width * scale), int(height * scale)),
        Image.Resampling.LANCZOS,
    )


def _save_png(im, png_path: str) -> bool:
    try:
        im.save(png_path, format="PNG")
        return os.path.exists(png_path)
    except Exception as exc:
        debug_log(f"screen_capture: save failed: {exc}", "screenshot")
        return False


def _capture_windows_mss(png_path: str) -> bool:
    try:
        import mss
        from PIL import Image
    except ImportError:
        return False
    try:
        with mss.mss() as sct:
            # Monitor 1 = primary display (0 = virtual desktop spanning all).
            monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            shot = sct.grab(monitor)
            im = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            return _save_png(_downscale_image(im), png_path)
    except Exception as exc:
        debug_log(f"screen_capture: mss failed: {exc}", "screenshot")
        return False


def _capture_windows_imagegrab(png_path: str) -> bool:
    def _work() -> bool:
        try:
            from PIL import ImageGrab

            # Primary display only — ``all_screens=True`` can crash when called
            # off the Qt main thread on Windows (STATUS_STACK_BUFFER_OVERRUN).
            im = ImageGrab.grab(all_screens=False)
            return _save_png(_downscale_image(im), png_path)
        except Exception as exc:
            debug_log(f"screen_capture: ImageGrab failed: {exc}", "screenshot")
            return False

    if _main_thread_runner is not None:
        return bool(_main_thread_runner(_work))
    if threading.current_thread() is threading.main_thread():
        return _work()
    debug_log(
        "screen_capture: ImageGrab off main thread without UI runner — skipped",
        "screenshot",
    )
    return False


def capture_display_png(png_path: str) -> bool:
    """Capture the primary display to ``png_path``. Platform-specific."""
    if sys.platform == "darwin":
        from jarvis.tools.builtin.screenshot import _capture_macos_interactive

        return _capture_macos_interactive(png_path)
    if sys.platform == "win32":
        if _capture_windows_mss(png_path):
            return True
        return _capture_windows_imagegrab(png_path)
    debug_log(f"screen_capture: unsupported platform {sys.platform}", "screenshot")
    return False


def capture_screen_png() -> tuple[str, str]:
    """Capture display to a temp PNG. Returns ``(tmpdir, png_path)`` or ``("", "")``."""
    tmpdir = tempfile.mkdtemp(prefix="jarvis_screen_")
    png_path = os.path.join(tmpdir, "shot.png")
    try:
        if capture_display_png(png_path):
            return tmpdir, png_path
        _cleanup_dir(tmpdir)
        return "", ""
    except Exception as exc:
        debug_log(f"screen_capture: capture failed: {exc}", "screenshot")
        _cleanup_dir(tmpdir)
        return "", ""


def _cleanup_dir(tmpdir: str) -> None:
    if not tmpdir:
        return
    try:
        for name in os.listdir(tmpdir):
            try:
                os.remove(os.path.join(tmpdir, name))
            except OSError:
                pass
        os.rmdir(tmpdir)
    except OSError:
        pass
