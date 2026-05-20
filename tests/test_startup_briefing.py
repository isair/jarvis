"""Tests for desktop startup briefing (greeting + status speech)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jarvis.config import Settings


def _minimal_settings(**overrides) -> Settings:
    """Build a Settings-like object with only fields the briefing needs."""
    base = {
        "startup_briefing_enabled": True,
        "operator_name": "Mr. Johnson",
        "persona_style": "formal_majordomo",
        "tts_enabled": True,
        "tts_engine": "piper",
        "tts_voice": None,
        "tts_rate": 200,
        "tts_chatterbox_device": "cpu",
        "tts_chatterbox_audio_prompt": None,
        "tts_chatterbox_exaggeration": 0.5,
        "tts_chatterbox_cfg_weight": 0.5,
        "tts_piper_model_path": None,
        "tts_piper_speaker": None,
        "tts_piper_length_scale": None,
        "tts_piper_noise_scale": None,
        "tts_piper_noise_w": None,
        "tts_piper_sentence_silence": None,
    }
    base.update(overrides)
    return MagicMock(**base)


@pytest.mark.unit
def test_build_startup_spoken_brief_addresses_operator_and_includes_sections():
    from desktop_app.startup_briefing import build_startup_spoken_brief

    cfg = _minimal_settings()
    text = build_startup_spoken_brief(
        cfg,
        weather_line="Baldone, 14°C, partly cloudy",
        gmail_lines=["SumUp: daily report", "When I Work: shift swap"],
        whatsapp_lines=["Līna: shift swap request"],
        mcp_line="All integrations ready",
        listening=True,
    )
    assert "Mr. Johnson" in text
    assert "14" in text or "cloudy" in text.lower() or "Baldone" in text or "degrees" in text
    assert "SumUp" in text or "e-past" in text.lower() or "email" in text.lower()
    assert "WhatsApp" in text or "whatsapp" in text.lower() or "Līna" in text
    assert "first" in text.lower() or "?" in text


@pytest.mark.unit
def test_build_startup_spoken_brief_mentions_work_queue(tmp_path, monkeypatch):
    from jarvis import operator

    qpath = tmp_path / "work_queue.json"
    monkeypatch.setattr(operator.work_queue, "_queue_path", lambda: qpath)
    operator.work_queue.create_item(title="Pay invoices")

    from desktop_app.startup_briefing import build_startup_spoken_brief

    cfg = _minimal_settings(work_queue_enabled=True)
    text = build_startup_spoken_brief(
        cfg,
        weather_line="",
        gmail_lines=[],
        whatsapp_lines=[],
        mcp_line="",
        listening=True,
        work_queue_lines=["Pay invoices"],
    )
    assert "Pay invoices" in text
    assert "one at a time" in text.lower() or "pa vienam" in text.lower()


@pytest.mark.unit
def test_gather_startup_brief_context_includes_work_queue(tmp_path, monkeypatch):
    from jarvis import operator

    qpath = tmp_path / "work_queue.json"
    monkeypatch.setattr(operator.work_queue, "_queue_path", lambda: qpath)
    operator.work_queue.create_item(title="Ship report")

    from desktop_app.startup_briefing import _gather_startup_brief_context

    cfg = _minimal_settings(work_queue_enabled=True)
    with patch("desktop_app.pulse_api.load_gmail_preview", return_value={"messages": []}):
        with patch("desktop_app.pulse_api.load_comms_log", return_value={}):
            with patch(
                "desktop_app.sulainis_api.load_calendar_preview",
                return_value={"events": []},
            ):
                ctx = _gather_startup_brief_context(cfg)
    wq = ctx.get("work_queue") or {}
    assert wq.get("total_active", 0) >= 1
    assert any("Ship" in str(t.get("title", "")) for t in wq.get("tasks") or [])


@pytest.mark.unit
def test_build_startup_spoken_brief_latvian_ends_with_question():
    from desktop_app.startup_briefing import build_startup_spoken_brief

    cfg = _minimal_settings(
        operator_name="Jansona kungs",
        latvian_quality_enabled=True,
    )
    text = build_startup_spoken_brief(
        cfg,
        weather_line="Baldone, 14 grādi",
        gmail_lines=[],
        whatsapp_lines=[],
        mcp_line="",
        listening=True,
    )
    assert "?" in text
    assert "vispirms" in text.lower()


@pytest.mark.unit
def test_synthesize_startup_brief_llm_uses_router_model():
    from desktop_app.startup_briefing import synthesize_startup_brief_llm

    cfg = _minimal_settings(
        tool_router_model="",
        intent_judge_model="gemma4:e2b",
        ollama_chat_model="gemma4:e4b",
        ollama_base_url="http://127.0.0.1:11434",
        latvian_quality_enabled=True,
    )
    with patch("desktop_app.startup_briefing._gather_startup_brief_context") as ctx:
        ctx.return_value = {
            "operator": "Jansona kungs",
            "assistant": "Johnny",
            "latvian": True,
            "listening": True,
            "weather": "Baldone, 12 grādi",
            "integrations": "OK",
            "gmail": [],
            "gmail_hint": "",
            "whatsapp": [{"chat": "Līna", "text": "maiņa", "at": ""}],
            "whatsapp_hint": "",
        }
        with patch(
            "jarvis.llm.call_llm_direct",
            return_value=(
                "Labdien, Jansona kungs. Ir aktīva saruna ar Līnu par maiņu. "
                "Ko vēlaties, lai es daru vispirms?"
            ),
        ) as llm:
            out = synthesize_startup_brief_llm(cfg)
    assert out is not None and "Līnu" in out
    assert llm.call_count == 1
    assert llm.call_args[0][1] == "gemma4:e2b"


@pytest.mark.unit
def test_collect_startup_brief_parts_prefers_llm():
    from desktop_app.startup_briefing import collect_startup_brief_parts

    cfg = _minimal_settings()
    with patch(
        "desktop_app.startup_briefing.synthesize_startup_brief_llm",
        return_value="LLM kopsavilkums ar jautājumu?",
    ):
        spoken, _mirror = collect_startup_brief_parts(cfg)
    assert spoken == "LLM kopsavilkums ar jautājumu?"


@pytest.mark.unit
def test_build_startup_spoken_brief_empty_when_disabled():
    from desktop_app.startup_briefing import build_startup_spoken_brief

    cfg = _minimal_settings(startup_briefing_enabled=False)
    assert (
        build_startup_spoken_brief(
            cfg,
            weather_line="x",
            gmail_lines=[],
            whatsapp_lines=[],
            mcp_line="",
            listening=True,
        )
        == ""
    )


@pytest.mark.unit
def test_refresh_startup_data_calls_sync_helpers():
    from desktop_app import startup_briefing

    cfg = _minimal_settings()
    with patch(
        "jarvis.operator.background_sync.maybe_run_background_sync"
    ) as bg:
        with patch("desktop_app.sulainis_sync.sync_all_sulainis_caches") as sul:
            startup_briefing.refresh_startup_data(cfg)
    bg.assert_called_once_with(cfg, force=True)
    sul.assert_called_once_with(cfg, force=True)
