from __future__ import annotations
import os
import tempfile
import subprocess
import shutil
from typing import Optional


def capture_screenshot_and_ocr(interactive: bool = True) -> Optional[str]:
    """
    Captures a screenshot (interactive region by default) and extracts text via OCR.

    Returns the recognized text, or None on failure.
    Requires macOS 'screencapture'. For OCR, prefers system 'tesseract' with pytesseract.
    """
    # Ensure screencapture exists (macOS)
    sc = shutil.which("screencapture")
    if not sc:
        return None

    tmpdir = tempfile.mkdtemp(prefix="jarvis_ocr_")
    png_path = os.path.join(tmpdir, "shot.png")

    try:
        cmd = [sc]
        if interactive:
            cmd.append("-i")
        cmd.append(png_path)
        # Launch screencapture; user selects a region and confirms
        ret = subprocess.run(cmd)
        if ret.returncode != 0 or not os.path.exists(png_path):
            return None
        text = _ocr_tesseract(png_path)
        if text is not None and text.strip():
            return text
        return None
    finally:
        try:
            if os.path.exists(png_path):
                os.remove(png_path)
            os.rmdir(tmpdir)
        except Exception:
            pass


def _ocr_tesseract(image_path: str) -> Optional[str]:
    # Prefer external tesseract binary via pytesseract
    tess = shutil.which("tesseract")
    if not tess:
        return None
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        return None
    try:
        with Image.open(image_path) as im:
            return pytesseract.image_to_string(im)
    except Exception:
        return None


