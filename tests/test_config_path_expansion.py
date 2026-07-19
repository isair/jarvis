"""
Regression tests for #467: tilde paths in config.json crash the daemon.

The app's own example config uses "~/.local/share/jarvis/jarvis.db", but
``load_settings`` passed path-like values through verbatim. ``mkdir`` then
created (or failed to create) a literal ``~`` directory — on macOS the
daemon died at boot with ``OSError: [Errno 30] Read-only file system: '~'``.

All user-supplied path-like settings must be tilde-expanded on load.
"""

import json
from pathlib import Path

from jarvis.config import load_settings


def _write_config(tmp_path, monkeypatch, values):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(values))
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))


def test_tilde_db_path_is_expanded(tmp_path, monkeypatch):
    """The example-config db_path value must never produce a literal '~' path."""
    _write_config(tmp_path, monkeypatch, {"db_path": "~/.local/share/jarvis/jarvis.db"})

    cfg = load_settings()

    assert "~" not in cfg.db_path
    assert cfg.db_path == str(Path("~/.local/share/jarvis/jarvis.db").expanduser())


def test_other_path_settings_are_expanded(tmp_path, monkeypatch):
    """All path-like settings honour tilde notation."""
    _write_config(
        tmp_path,
        monkeypatch,
        {
            "sqlite_vss_path": "~/vss/vss0.dylib",
            "tts_piper_model_path": "~/models/voice.onnx",
            "tts_chatterbox_audio_prompt": "~/prompts/me.wav",
        },
    )

    cfg = load_settings()

    assert cfg.sqlite_vss_path == str(Path("~/vss/vss0.dylib").expanduser())
    assert cfg.tts_piper_model_path == str(Path("~/models/voice.onnx").expanduser())
    assert cfg.tts_chatterbox_audio_prompt == str(Path("~/prompts/me.wav").expanduser())


def test_null_db_path_falls_back_to_default(tmp_path, monkeypatch):
    """A 'null' db_path uses the default location instead of a literal path."""
    _write_config(tmp_path, monkeypatch, {"db_path": "null"})

    cfg = load_settings()

    assert "null" not in cfg.db_path
    assert cfg.db_path.endswith("jarvis.db")


def test_tilde_whisper_model_path_is_expanded(tmp_path, monkeypatch):
    """whisper_model accepts a local model directory; tilde must expand, names pass through."""
    _write_config(tmp_path, monkeypatch, {"whisper_model": "~/models/faster-whisper-medium"})
    assert load_settings().whisper_model == str(Path("~/models/faster-whisper-medium").expanduser())

    _write_config(tmp_path, monkeypatch, {"whisper_model": "medium"})
    assert load_settings().whisper_model == "medium"


def test_absolute_and_unset_paths_are_untouched(tmp_path, monkeypatch):
    """Absolute paths pass through unchanged; unset optional paths stay None."""
    db = tmp_path / "jarvis.db"
    _write_config(tmp_path, monkeypatch, {"db_path": str(db)})

    cfg = load_settings()

    assert cfg.db_path == str(db)
    assert cfg.sqlite_vss_path is None
    assert cfg.tts_piper_model_path is None
    assert cfg.tts_chatterbox_audio_prompt is None
