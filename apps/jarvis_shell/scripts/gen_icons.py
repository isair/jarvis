#!/usr/bin/env python3
"""Generate minimal placeholder icons for the Tauri bundle."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "src-tauri" / "icons"

# HUD cyan on dark (#00e5ff on #0a0f14)
CYAN = (0, 229, 255)
BG = (10, 15, 20)


def _png(size: int) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = bytearray()
    for y in range(size):
        raw.append(0)
        for x in range(size):
            margin = max(2, size // 8)
            inner = margin <= x < size - margin and margin <= y < size - margin
            r, g, b = CYAN if inner else BG
            raw.extend((r, g, b, 255 if inner else 200))
    compressed = zlib.compress(bytes(raw), 9)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    signature = b"\x89PNG\r\n\x1a\n"
    return (
        signature
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


def main() -> None:
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    for name, size in (
        ("32x32.png", 32),
        ("128x128.png", 128),
        ("128x128@2x.png", 256),
        ("icon.png", 256),
    ):
        (ICON_DIR / name).write_bytes(_png(size if size <= 256 else 256))
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(_png(256))).convert("RGBA")
        img.save(
            ICON_DIR / "icon.ico",
            format="ICO",
            sizes=[(16, 16), (32, 32), (48, 48), (256, 256)],
        )
    except ImportError:
        (ICON_DIR / "icon.ico").write_bytes((ICON_DIR / "32x32.png").read_bytes())
    print(f"Generated icons in {ICON_DIR}")


if __name__ == "__main__":
    main()
