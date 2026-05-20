"""Tests for WhatsApp bridge patch and QR rendering."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jarvis.integrations.whatsapp.bridge import (
    PATCH_MARKER,
    apply_bridge_patch,
    render_qr_png,
)


SAMPLE_GO_SNIPPET = '''
		for evt := range qrChan {
			if evt.Event == "code" {
				fmt.Println("\\nScan this QR code with your WhatsApp app:")
				qrterminal.GenerateHalfBlock(evt.Code, qrterminal.L, os.Stdout)
			} else if evt.Event == "success" {
				connected <- true
				break
			}
		}
'''


@pytest.mark.unit
class TestWhatsAppBridgePatch:
    def test_apply_bridge_patch_adds_marker(self, tmp_path: Path) -> None:
        main_go = tmp_path / "main.go"
        main_go.write_text(SAMPLE_GO_SNIPPET, encoding="utf-8")
        apply_bridge_patch(main_go)
        text = main_go.read_text(encoding="utf-8")
        assert PATCH_MARKER in text
        assert 'JARVIS_QR_CODE:%s' in text
        assert "JARVIS_AUTH_OK" in text

    def test_apply_bridge_patch_idempotent(self, tmp_path: Path) -> None:
        main_go = tmp_path / "main.go"
        main_go.write_text(SAMPLE_GO_SNIPPET, encoding="utf-8")
        apply_bridge_patch(main_go)
        first = main_go.read_text(encoding="utf-8")
        apply_bridge_patch(main_go)
        assert main_go.read_text(encoding="utf-8") == first


@pytest.mark.unit
class TestWhatsAppQRRender:
    def test_render_qr_png_returns_bytes(self) -> None:
        png = render_qr_png("test-pairing-payload-12345")
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(png) > 200
