"""Screen awareness - expose what the user is currently looking at.

Privacy-first and fully opt-in. When ``screen_awareness_enabled`` is set, the
reply engine can see a short, structured snapshot of the user's screen so it
can answer "what does this button do", "summarise what's on my screen", etc.

Two layers, independently toggleable:

1. **Active window title** (cheap, no image processing) - the name of the
   front-most app and window. This is the default and carries almost no
   privacy surface beyond "the user is in app X".
2. **OCR of visible text** (opt-in via ``screen_awareness_ocr``) - when
   enabled, a screenshot is taken and the visible text is extracted locally
   with Tesseract. No image ever leaves the machine.

Both layers respect ``allowlist_bundles``: if the active app's bundle/process
name is not on the allowlist, capture is skipped and ``None`` is returned. An
empty allowlist means "capture from any app" (the default), so the feature is
useful out of the box but can be locked down per-app.

The provider caches the last snapshot for ``cache_seconds`` so back-to-back
queries within one hot window don't each trigger a fresh capture.
"""

from __future__ import annotations

import platform
import threading
import time
from typing import List, Optional, Tuple

from ..debug import debug_log


def _get_active_window() -> Optional[Tuple[str, str]]:
    """Return ``(app_name, window_title)`` for the front-most window.

    Best-effort across platforms. Returns ``None`` when no window info is
    available (unsupported platform, headless, or missing dependency).
    """
    system = platform.system().lower()
    try:
        if system == "darwin":
            return _get_active_window_macos()
        if system == "windows":
            return _get_active_window_windows()
        return _get_active_window_linux()
    except Exception as exc:
        debug_log(f"screen: active window lookup failed: {exc}", "screen")
        return None


def _get_active_window_macos() -> Optional[Tuple[str, str]]:
    try:
        from AppKit import NSWorkspace  # type: ignore
    except ImportError:
        return None
    try:
        ws = NSWorkspace.sharedWorkspace()
        app = ws.activeApplication()
        app_name = app.get("NSApplicationName", "") if app else ""
        win = ws.frontmostApplication().mainWindow() if ws.frontmostApplication() else None
        title = win.title() if win else ""
        return (app_name or "Unknown", title or "")
    except Exception:
        return None


def _get_active_window_windows() -> Optional[Tuple[str, str]]:
    try:
        import win32gui  # type: ignore
        import win32process  # type: ignore
        import psutil  # type: ignore
    except ImportError:
        # Fallback to pygetwindow if present.
        try:
            import pygetwindow as gw  # type: ignore
            win = gw.getActiveWindow()
            if win:
                return (None, win.title or "")
        except Exception:
            pass
        return None
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None
        title = win32gui.GetWindowText(hwnd) or ""
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            app_name = psutil.Process(pid).name()
        except Exception:
            app_name = "Unknown"
        return (app_name, title)
    except Exception:
        return None


def _get_active_window_linux() -> Optional[Tuple[str, str]]:
    # X11 via xdotool / wmctrl when available; otherwise nothing.
    try:
        import subprocess
        out = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return ("Unknown", out.stdout.strip())
    except Exception:
        pass
    return None


def _ocr_screenshot() -> str:
    """Capture the screen and OCR it locally. Returns extracted text ('' on failure)."""
    import shutil
    import subprocess
    import tempfile
    import os

    system = platform.system().lower()
    tmpdir = tempfile.mkdtemp(prefix="jarvis_screen_")
    png = os.path.join(tmpdir, "screen.png")
    try:
        if system == "darwin":
            sc = shutil.which("screencapture")
            if not sc:
                return ""
            subprocess.run([sc, "-x", png], timeout=5, check=False)
        elif system == "windows":
            # Use Pillow + Windows GDI via ImageGrab.
            try:
                from PIL import ImageGrab  # type: ignore
                img = ImageGrab.grab()
                img.save(png)
            except Exception:
                return ""
        else:
            sc = shutil.which("import")  # ImageMagick
            if not sc:
                return ""
            subprocess.run([sc, png], timeout=5, check=False)

        if not os.path.exists(png):
            return ""

        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
            with Image.open(png) as im:
                text = pytesseract.image_to_string(im)
            return (text or "").strip()
        except Exception as exc:
            debug_log(f"screen: OCR failed: {exc}", "screen")
            return ""
    except Exception as exc:
        debug_log(f"screen: screenshot failed: {exc}", "screen")
        return ""
    finally:
        try:
            if os.path.exists(png):
                os.remove(png)
            os.rmdir(tmpdir)
        except Exception:
            pass


def _app_matches_allowlist(app_name: str, allowlist: List[str]) -> bool:
    """Return True if *app_name* is permitted by *allowlist*.

    An empty allowlist permits everything. Matching is case-insensitive and
    ignores ``.app`` / ``.exe`` suffixes.
    """
    if not allowlist:
        return True
    if not app_name:
        return False
    name = app_name.lower().replace(".app", "").replace(".exe", "")
    for entry in allowlist:
        e = entry.lower().replace(".app", "").replace(".exe", "")
        if e and e in name:
            return True
    return False


class ScreenContextProvider:
    """Thread-safe, cached screen-context provider for the reply engine."""

    def __init__(
        self,
        enabled: bool = False,
        ocr_enabled: bool = False,
        allowlist: Optional[List[str]] = None,
        max_text_chars: int = 500,
        cache_seconds: float = 20.0,
    ) -> None:
        self.enabled = enabled
        self.ocr_enabled = ocr_enabled
        self.allowlist = list(allowlist or [])
        self.max_text_chars = max_text_chars
        self.cache_seconds = cache_seconds
        self._lock = threading.Lock()
        self._cache: Optional[str] = None
        self._cache_time: float = 0.0

    def get_context(self) -> Optional[str]:
        """Return a short screen-context string, or ``None`` if unavailable.

        The result is cached for ``cache_seconds``. Returns ``None`` when the
        feature is disabled, no window info is available, or the active app is
        not on the allowlist.
        """
        if not self.enabled:
            return None

        with self._lock:
            now = time.time()
            if self._cache is not None and (now - self._cache_time) < self.cache_seconds:
                return self._cache

        window = _get_active_window()
        if not window:
            result: Optional[str] = None
        else:
            app_name, title = window
            if not _app_matches_allowlist(app_name, self.allowlist):
                debug_log(f"screen: app '{app_name}' not on allowlist, skipping", "screen")
                result = None
            else:
                parts: List[str] = []
                if app_name and app_name != "Unknown":
                    parts.append(f"App: {app_name}")
                if title:
                    parts.append(f"Window: {title}")
                if self.ocr_enabled:
                    ocr = _ocr_screenshot()
                    if ocr:
                        if len(ocr) > self.max_text_chars:
                            ocr = ocr[:self.max_text_chars].rstrip() + "…"
                        parts.append(f"Visible text (OCR): {ocr}")
                if parts:
                    result = "Screen context (what the user is currently looking at):\n" + "\n".join(parts)
                else:
                    result = None

        with self._lock:
            self._cache = result
            self._cache_time = time.time()
        return result

    def invalidate(self) -> None:
        """Clear the cached snapshot (e.g. on a new conversation)."""
        with self._lock:
            self._cache = None
            self._cache_time = 0.0
