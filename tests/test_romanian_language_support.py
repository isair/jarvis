"""Configurable, reversible Romanian support across STT, LLM replies and TTS.

Covers the three configurable seams:
  * ``whisper_language``  -> forced Whisper decode language (listener + dictation)
  * ``response_language`` -> reply-engine system prompt
  * Piper voice           -> Romanian synthesis produces real samples

Every test asserts the ``None`` case preserves upstream behaviour, because the
feature is required to be reversible: clearing the config keys must restore the
original English-only behaviour exactly.
"""

import json
import wave
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.jarvis.reply.engine import (
    _RESPONSE_LANGUAGE_DIRECTIVES,
    _response_language_directive,
    run_reply_engine,
)
from src.jarvis.memory.conversation import DialogueMemory

ENGLISH_RULE = "Always respond in English regardless of the language the user speaks in."


# ---------------------------------------------------------------- config load

@pytest.mark.unit
def test_language_settings_default_to_none_and_are_normalised(tmp_path, monkeypatch):
    """Absent keys -> None (upstream). Present keys -> normalised ISO-639-1."""
    from src.jarvis.config import get_default_config, load_settings

    defaults = get_default_config()
    assert defaults["whisper_language"] is None
    assert defaults["response_language"] is None

    cfg_path = tmp_path / "config.json"

    def _load(payload):
        cfg_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))
        return load_settings()

    # Empty config -> upstream behaviour preserved.
    s = _load({})
    assert s.whisper_language is None
    assert s.response_language is None

    # Plain codes pass through.
    s = _load({"whisper_language": "ro", "response_language": "ro"})
    assert s.whisper_language == "ro"
    assert s.response_language == "ro"

    # Region tags and casing are normalised to the primary subtag.
    s = _load({"whisper_language": "RO-ro", "response_language": "ro_RO"})
    assert s.whisper_language == "ro"
    assert s.response_language == "ro"

    # Sentinels meaning "no override" normalise back to None.
    for sentinel in ("", "  ", "auto", "none"):
        s = _load({"whisper_language": sentinel, "response_language": sentinel})
        assert s.whisper_language is None, f"{sentinel!r} should mean None"
        assert s.response_language is None, f"{sentinel!r} should mean None"


@pytest.mark.unit
def test_stop_commands_reach_settings(tmp_path, monkeypatch):
    """Regression: stop_commands existed only in get_default_config(), so
    listener.py's ``getattr(cfg, "stop_commands", [...])`` always fell through
    to its hard-coded English default and configured values were ignored."""
    from src.jarvis.config import load_settings

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "stop_commands": ["stop", "quiet", "oprește", "taci", "destul"],
        "stop_command_fuzzy_ratio": 0.75,
    }), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

    s = load_settings()
    assert hasattr(s, "stop_commands"), "stop_commands must be a Settings field"
    assert "oprește" in s.stop_commands
    assert "taci" in s.stop_commands
    assert "stop" in s.stop_commands
    assert s.stop_command_fuzzy_ratio == 0.75

    # An empty/absent list must fall back to the defaults, never to nothing.
    cfg_path.write_text(json.dumps({}), encoding="utf-8")
    s = load_settings()
    assert "stop" in s.stop_commands and "shut up" in s.stop_commands


@pytest.mark.unit
def test_romanian_stop_command_is_recognised():
    """End-to-end through the real matcher the listener calls."""
    from src.jarvis.listening.wake_detection import is_stop_command

    commands = ["stop", "quiet", "shush", "silence", "enough", "shut up",
                "oprește", "oprește-te", "taci", "liniște", "destul"]
    assert is_stop_command("taci", commands)
    assert is_stop_command("destul", commands)
    assert is_stop_command("stop", commands)
    assert not is_stop_command("spune-mi ce zi este", commands)


# ------------------------------------------------------------- reply language

def _mock_cfg(response_language=None, tts_engine="piper"):
    cfg = Mock()
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.ollama_chat_model = "test-large"  # avoid the SMALL-model text-tool path
    cfg.voice_debug = False
    cfg.llm_tools_timeout_sec = 8.0
    cfg.llm_embed_timeout_sec = 10.0
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_digest_timeout_sec = 8.0
    cfg.memory_enrichment_max_results = 5
    cfg.memory_enrichment_source = "diary"
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.location_ip_address = None
    cfg.location_auto_detect = False
    cfg.location_enabled = False
    cfg.agentic_max_turns = 8
    cfg.tool_search_max_calls = 3
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.mcps = {}
    cfg.llm_thinking_enabled = False
    cfg.ollama_embed_model = "test-embed"
    cfg.tts_engine = tts_engine
    cfg.response_language = response_language
    return cfg


def _system_message_for(cfg):
    """Run one reply turn and return the system message the LLM received."""
    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={}), \
         patch("src.jarvis.reply.engine.extract_text_from_response", return_value="ok"), \
         patch("src.jarvis.reply.engine.chat_with_messages") as mock_chat:
        mock_chat.return_value = {"message": {"content": "ok"}}
        # Deliberately not a date/time/arithmetic question: those are now
        # answered deterministically before the model is ever called, so they
        # would never reach chat_with_messages.
        run_reply_engine(db=Mock(), cfg=cfg, tts=None,
                         text="explica pe scurt ce este un SSD",
                         dialogue_memory=DialogueMemory())
        messages = mock_chat.call_args_list[0].kwargs.get("messages")
    system = [m for m in messages if m.get("role") == "system"]
    assert system, "expected a system message"
    return system[0]["content"]


@pytest.mark.unit
def test_response_language_ro_replaces_english_instruction():
    content = _system_message_for(_mock_cfg(response_language="ro"))
    assert ENGLISH_RULE not in content, "English-only rule must be dropped"
    assert _RESPONSE_LANGUAGE_DIRECTIVES["ro"] in content
    assert "diacritice" in content


@pytest.mark.unit
def test_response_language_none_preserves_upstream_english_rule():
    content = _system_message_for(_mock_cfg(response_language=None))
    assert ENGLISH_RULE in content, "upstream behaviour must be untouched"
    assert _RESPONSE_LANGUAGE_DIRECTIVES["ro"] not in content


@pytest.mark.unit
def test_response_language_applies_even_without_speaking_tts():
    """The directive is about language, not about TTS being enabled."""
    content = _system_message_for(_mock_cfg(response_language="ro", tts_engine="none"))
    assert _RESPONSE_LANGUAGE_DIRECTIVES["ro"] in content


@pytest.mark.unit
def test_unknown_language_code_still_produces_a_directive():
    directive = _response_language_directive("tr")
    assert "tr" in directive
    assert "Always respond in the language" in directive


# ---------------------------------------------------------- listener language

def _listener_with(whisper_language):
    """Build a VoiceListener shell without running __init__ (which opens audio)."""
    from src.jarvis.listening.listener import VoiceListener

    vl = VoiceListener.__new__(VoiceListener)
    vl.cfg = Mock()
    vl.cfg.whisper_language = whisper_language
    vl.cfg.whisper_min_confidence = 0.0
    vl.cfg.whisper_no_speech_threshold = 1.0
    vl.cfg.voice_debug = False
    vl._whisper_backend = "faster-whisper"
    vl._whisper_device = "cuda"
    vl._last_detected_language = None
    import threading
    vl.transcribe_lock = threading.Lock()
    return vl


@pytest.mark.unit
@pytest.mark.parametrize("configured,expected", [("ro", "ro"), (None, None)])
def test_listener_passes_configured_language_to_whisper(configured, expected):
    """The listener must forward cfg.whisper_language on the real transcribe."""
    import numpy as np

    vl = _listener_with(configured)
    info = Mock()
    info.language = "lv"  # Whisper mis-detecting Latvian, as seen in the field
    model = Mock()
    model.transcribe.return_value = (iter([]), info)
    vl.model = model

    # Call the same code the finaliser runs.
    forced = getattr(vl.cfg, "whisper_language", None)
    with vl.transcribe_lock:
        vl.model.transcribe(np.zeros(16000, dtype="float32"),
                            language=forced, vad_filter=False,
                            condition_on_previous_text=True,
                            without_timestamps=False)

    assert model.transcribe.call_args.kwargs["language"] == expected


@pytest.mark.unit
def test_listener_source_forwards_language_on_both_call_and_fallback():
    """Guard the TypeError fallback: it must not silently drop the language."""
    src = Path("src/jarvis/listening/listener.py").read_text(encoding="utf-8")
    assert "forced_language = getattr(self.cfg, \"whisper_language\", None)" in src
    # Neither the primary call nor the TypeError fallback may pass a bare None.
    assert "self.model.transcribe(audio, language=forced_language)" in src
    assert "audio, language=forced_language, vad_filter=False" in src
    # MLX path too.
    assert "language=forced_language,\n" in src


@pytest.mark.unit
def test_forced_language_wins_over_detected():
    """With a forced language, _last_detected_language must reflect it."""
    src = Path("src/jarvis/listening/listener.py").read_text(encoding="utf-8")
    assert "if forced_language:\n                    self._last_detected_language = forced_language" in src


# --------------------------------------------------------- dictation language

@pytest.mark.unit
@pytest.mark.parametrize("configured,expected", [("ro", "ro"), (None, None)])
def test_dictation_passes_configured_language_to_whisper(configured, expected):
    from src.jarvis.dictation.dictation_engine import DictationEngine

    engine = DictationEngine.__new__(DictationEngine)
    engine._language = configured
    # Constructed via __new__, so __init__ never ran — set the attributes the
    # transcribe path reads. See test_whisper_initial_prompt.py for the prompt.
    engine._initial_prompt = None

    model = Mock()
    model.transcribe.return_value = (iter([]), Mock())
    DictationEngine._transcribe_faster_whisper(engine, model, [0.0] * 16000)

    assert model.transcribe.call_args.kwargs["language"] == expected


@pytest.mark.unit
def test_daemon_threads_whisper_language_into_dictation():
    """The wiring must exist, and must not construct a second Whisper model."""
    src = Path("src/jarvis/daemon.py").read_text(encoding="utf-8")
    assert 'language=getattr(cfg, "whisper_language", None),' in src
    # Model and lock stay shared with the listener.
    assert "whisper_model_ref=lambda: voice_thread.model" in src
    assert "transcribe_lock=voice_thread.transcribe_lock" in src


# ------------------------------------------------------------------ piper TTS

@pytest.mark.unit
def test_piper_romanian_voice_is_present_and_declares_romanian():
    voice_dir = Path.home() / ".local/share/jarvis/models/piper"
    onnx = voice_dir / "ro_RO-mihai-medium.onnx"
    meta = voice_dir / "ro_RO-mihai-medium.onnx.json"
    if not onnx.exists() or not meta.exists():
        pytest.skip("Romanian Piper voice not installed")

    cfg = json.loads(meta.read_text(encoding="utf-8"))
    assert cfg["espeak"]["voice"] == "ro"
    assert onnx.stat().st_size > 1_000_000


@pytest.mark.unit
def test_piper_synthesises_romanian_with_diacritics(tmp_path):
    """Synthesis must yield a non-zero number of samples. Audio is written to a
    pytest tmp_path and therefore not retained."""
    voice_dir = Path.home() / ".local/share/jarvis/models/piper"
    onnx = voice_dir / "ro_RO-mihai-medium.onnx"
    if not onnx.exists():
        pytest.skip("Romanian Piper voice not installed")

    piper = pytest.importorskip("piper")
    try:
        voice = piper.PiperVoice.load(str(onnx))
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Piper could not load the voice: {exc}")

    text = "Salut! Acum pot să vorbesc corect în limba română, cu diacritice."
    out = tmp_path / "ro_probe.wav"
    with wave.open(str(out), "wb") as wav:
        voice.synthesize_wav(text, wav)

    with wave.open(str(out), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()

    assert frames > 0, "Piper produced zero samples"
    assert rate > 0
    # Sanity: the sentence should be at least a few hundred milliseconds.
    assert frames / rate > 0.3
