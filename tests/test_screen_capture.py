"""Tests for thread-safe screen capture."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_downscale_large_image():
    from jarvis.screen_capture import _downscale_image

    from PIL import Image

    im = Image.new("RGB", (4000, 2000), color="red")
    out = _downscale_image(im, max_side=1920)
    assert max(out.size) == 1920


@pytest.mark.unit
@patch("jarvis.screen_capture._capture_windows_mss", return_value=True)
def test_windows_prefers_mss(mock_mss, tmp_path):
    from jarvis.screen_capture import capture_display_png

    png = tmp_path / "shot.png"
    assert capture_display_png(str(png)) is True
    mock_mss.assert_called_once()


@pytest.mark.unit
@patch("jarvis.screen_capture._main_thread_runner", None)
@patch("jarvis.screen_capture._capture_windows_mss", return_value=False)
@patch("jarvis.screen_capture._save_png", return_value=True)
@patch("jarvis.screen_capture._downscale_image", side_effect=lambda im, **_: im)
def test_windows_imagegrab_uses_primary_display(mock_downscale, mock_save, _mss):
    import threading

    from jarvis.screen_capture import _capture_windows_imagegrab

    fake_im = MagicMock()
    grab = MagicMock(return_value=fake_im)
    with patch("PIL.ImageGrab.grab", grab):
        with patch.object(threading, "current_thread", return_value=threading.main_thread()):
            assert _capture_windows_imagegrab("/tmp/x.png") is True
    grab.assert_called_once_with(all_screens=False)
    mock_save.assert_called_once()
