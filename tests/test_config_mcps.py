import pytest
from jarvis.config import get_default_config, load_settings


@pytest.mark.unit
def test_default_config_has_empty_mcps():
    cfg = get_default_config()
    assert isinstance(cfg.get("mcps"), dict)
    assert cfg["mcps"] == {}


@pytest.mark.unit
def test_load_settings_normalizes_mcps(monkeypatch, tmp_path):
    # Write a minimal config that overrides mcps with a list of dicts using name field
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        """
        {
          "mcps": [
            {"name": "fs", "transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "~"], "env": {}}
          ],
          "ollama_base_url": "http://localhost",
          "ollama_embed_model": "x",
          "ollama_chat_model": "y"
        }
        """,
        encoding="utf-8",
    )

    # Point loader to our temporary config
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))
    s = load_settings()
    assert isinstance(s.mcps, dict)
    assert "fs" in s.mcps
    assert s.mcps["fs"]["transport"] == "stdio"

