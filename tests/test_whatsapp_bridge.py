"""Tests for WhatsApp bridge helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestWhatsAppBridgeHelpers:
    def test_gcc_hint_mentions_winget(self):
        from jarvis.integrations.whatsapp.bridge import windows_gcc_install_hint

        assert "winget" in windows_gcc_install_hint().lower()

    def test_whatsmeow_context_patch_adds_background(self, tmp_path):
        from jarvis.integrations.whatsapp.bridge import apply_whatsmeow_context_patch

        main_go = tmp_path / "main.go"
        main_go.write_text(
            'x := client.Download(downloader)\n'
            'sqlstore.New("sqlite3", "file:x.db", log)\n'
            'container.GetFirstDevice()\n',
            encoding="utf-8",
        )
        assert apply_whatsmeow_context_patch(main_go) is True
        text = main_go.read_text(encoding="utf-8")
        assert "context.Background()" in text
        assert apply_whatsmeow_context_patch(main_go) is False

    def test_resolve_gcc_finds_winget_winlibs_path(self, tmp_path, monkeypatch):
        from jarvis.integrations.whatsapp import bridge

        pkg = (
            tmp_path
            / "Microsoft"
            / "WinGet"
            / "Packages"
            / "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe"
        )
        gcc_bin = pkg / "mingw64" / "bin"
        gcc_bin.mkdir(parents=True)
        (gcc_bin / "gcc.exe").write_bytes(b"")

        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        monkeypatch.setenv("PATH", "")
        assert bridge.resolve_gcc_command({"PATH": ""}) is not None
        assert bridge.gcc_available() is True

    def test_migrate_legacy_uvx_config(self, tmp_path, monkeypatch):
        from jarvis.integrations import whatsapp as wa_pkg
        from jarvis.integrations.whatsapp import bridge

        install = tmp_path / "whatsapp-mcp"
        server = install / "whatsapp-mcp-main" / "whatsapp-mcp-server"
        server.mkdir(parents=True)
        (server / "main.py").write_text("# stub\n", encoding="utf-8")
        (install / "whatsapp-mcp-main" / "whatsapp-bridge").mkdir(parents=True)
        (install / "whatsapp-mcp-main" / "whatsapp-bridge" / "main.go").write_text(
            "package main\nfunc main() {}\n",
            encoding="utf-8",
        )

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            '{"mcps":{"whatsapp":{"command":"uvx","args":["whatsapp-mcp-server"]}}}',
            encoding="utf-8",
        )

        monkeypatch.setattr(bridge, "whatsapp_install_dir", lambda: install)
        monkeypatch.setattr(bridge, "default_config_path", lambda: cfg_path)

        assert bridge.migrate_legacy_whatsapp_mcp_config() is True
        import json

        saved = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert saved["mcps"]["whatsapp"]["args"][-1] == "main.py"
