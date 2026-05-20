"""PTT and continuous_listening configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.config import _default_ptt_hotkey, get_default_config, load_settings


@pytest.mark.unit
def test_default_ptt_hotkey() -> None:
    assert _default_ptt_hotkey() == "ctrl+shift+j"


@pytest.mark.unit
def test_default_config_includes_ptt_fields() -> None:
    d = get_default_config()
    assert d["ptt_enabled"] is True
    assert d["ptt_hotkey"] == "ctrl+shift+j"
    assert d["continuous_listening"] is True


@pytest.mark.unit
def test_load_settings_ptt_and_continuous(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "ptt_enabled": True,
                "ptt_hotkey": "ctrl+shift+space",
                "continuous_listening": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("jarvis.config.default_config_path", lambda: cfg)
    s = load_settings()
    assert s.ptt_enabled is True
    assert s.ptt_hotkey == "ctrl+shift+space"
    assert s.continuous_listening is False


@pytest.mark.unit
def test_dictation_engine_jarvis_delivery_submits_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jarvis.dictation.dictation_engine import DictationEngine

    submitted: list[str] = []

    def _fake_submit(text: str, image_paths=None) -> bool:
        submitted.append(text)
        return True

    monkeypatch.setattr(
        "jarvis.text_input.submit_text_query",
        _fake_submit,
    )

    engine = DictationEngine(
        whisper_model_ref=lambda: None,
        whisper_backend_ref=lambda: "faster-whisper",
        mlx_repo_ref=lambda: None,
        delivery_mode="jarvis",
    )
    engine._delivery_mode = "jarvis"
    engine._log_tag = "ptt"
    engine._target_sample_rate = 16000
    monkeypatch.setattr(engine, "_transcribe", lambda _audio: "hello jarvis")
    engine._transcribe_and_paste([__import__("numpy").zeros(8000, dtype="float32")])
    assert submitted == ["hello jarvis"]
