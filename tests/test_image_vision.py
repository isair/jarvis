"""Tests for attached image vision helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_is_image_file():
    from jarvis.image_vision import is_image_file

    assert is_image_file("photo.PNG")
    assert not is_image_file("doc.pdf")


@pytest.mark.unit
def test_build_images_context_includes_fence(tmp_path: Path):
    from jarvis.image_vision import build_images_context_for_query

    img = tmp_path / "x.png"
    img.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00"
        b"\x0cIDATx\x9cc\xf8\x0f\x00\x01\x01\x01\x00\x18\xdd\x8d\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    cfg = MagicMock()
    cfg.screen_vision_enabled = False

    with patch("jarvis.image_vision._ocr_image", return_value="label text"):
        block = build_images_context_for_query(cfg, [str(img)], "what is this?")

    assert "UNTRUSTED IMAGE DATA" in block
    assert "label text" in block
