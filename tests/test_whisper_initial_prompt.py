"""Configurable Whisper `initial_prompt`.

The prompt biases Whisper's decoder toward a vocabulary; it never substitutes
words after the fact. With the key unset every transcribe call must be
byte-identical to upstream, so the feature is a true no-op by default.
"""

import json
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

PROMPT = "Limba română, cu diacritice. Forme uzuale: explică-mi, propoziții."


# ---------------------------------------------------------------- config load

def _load(tmp_path, monkeypatch, payload):
    from src.jarvis.config import load_settings
    p = tmp_path / "config.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(p))
    return load_settings()


@pytest.mark.unit
def test_default_is_none(tmp_path, monkeypatch):
    from src.jarvis.config import get_default_config
    assert get_default_config()["whisper_initial_prompt"] is None
    assert _load(tmp_path, monkeypatch, {}).whisper_initial_prompt is None


@pytest.mark.unit
def test_value_is_loaded_and_stripped(tmp_path, monkeypatch):
    s = _load(tmp_path, monkeypatch, {"whisper_initial_prompt": f"   {PROMPT}   "})
    assert s.whisper_initial_prompt == PROMPT


@pytest.mark.unit
@pytest.mark.parametrize("blank", [None, "", "   ", "\n\t "])
def test_blank_normalises_to_none(tmp_path, monkeypatch, blank):
    s = _load(tmp_path, monkeypatch, {"whisper_initial_prompt": blank})
    assert s.whisper_initial_prompt is None, f"{blank!r} must disable the prompt"


@pytest.mark.unit
def test_length_is_capped_at_300_chars(tmp_path, monkeypatch):
    s = _load(tmp_path, monkeypatch, {"whisper_initial_prompt": "ă" * 500})
    assert s.whisper_initial_prompt is not None
    assert len(s.whisper_initial_prompt) <= 300


@pytest.mark.unit
def test_configured_prompt_stays_within_cap():
    """The real config must not silently lose content to the cap."""
    real = Path.home() / ".config/jarvis/config.json"
    if not real.exists():
        pytest.skip("no local config")
    val = json.loads(real.read_text(encoding="utf-8")).get("whisper_initial_prompt")
    if val:
        assert len(val) <= 300, "configured prompt would be truncated"


# ------------------------------------------------------------------- listener

def _listener(language="ro", prompt=None):
    from src.jarvis.listening.listener import VoiceListener
    vl = VoiceListener.__new__(VoiceListener)
    vl.cfg = Mock()
    vl.cfg.whisper_language = language
    vl.cfg.whisper_initial_prompt = prompt
    vl.cfg.voice_debug = False
    vl._whisper_device = "cuda"
    vl._last_detected_language = None
    vl.transcribe_lock = threading.Lock()
    return vl


def _fw_call(vl, model):
    """Replay listener.py's faster-whisper call exactly."""
    cpu_mode = vl._whisper_device == "cpu"
    forced_language = getattr(vl.cfg, "whisper_language", None)
    initial_prompt = getattr(vl.cfg, "whisper_initial_prompt", None)
    prompt_kw = {"initial_prompt": initial_prompt} if initial_prompt else {}
    with vl.transcribe_lock:
        try:
            return model.transcribe(None, language=forced_language, vad_filter=False,
                                    condition_on_previous_text=not cpu_mode,
                                    without_timestamps=cpu_mode, **prompt_kw)
        except TypeError:
            return model.transcribe(None, language=forced_language)


@pytest.mark.unit
def test_listener_sends_language_and_prompt():
    model = Mock()
    model.transcribe.return_value = (iter([]), Mock())
    _fw_call(_listener("ro", PROMPT), model)
    kw = model.transcribe.call_args.kwargs
    assert kw["language"] == "ro"
    assert kw["initial_prompt"] == PROMPT


@pytest.mark.unit
def test_listener_omits_prompt_when_unset():
    """Unset must produce the upstream call shape, not initial_prompt=None."""
    model = Mock()
    model.transcribe.return_value = (iter([]), Mock())
    _fw_call(_listener("ro", None), model)
    assert "initial_prompt" not in model.transcribe.call_args.kwargs


@pytest.mark.unit
def test_listener_typeerror_fallback_keeps_language_drops_prompt():
    """An older backend rejecting initial_prompt must not lose language='ro'."""
    model = Mock()
    calls = []

    def transcribe(_a, **kw):
        calls.append(kw)
        if "initial_prompt" in kw:
            raise TypeError("unexpected keyword argument 'initial_prompt'")
        return (iter([]), Mock())

    model.transcribe.side_effect = transcribe
    _fw_call(_listener("ro", PROMPT), model)

    assert len(calls) == 2, "must retry once"
    assert "initial_prompt" in calls[0]
    assert "initial_prompt" not in calls[1], "retry must drop the prompt"
    assert calls[1]["language"] == "ro", "retry must keep the language"


@pytest.mark.unit
def test_listener_source_wires_both_backends():
    src = Path("src/jarvis/listening/listener.py").read_text(encoding="utf-8")
    assert src.count('getattr(self.cfg, "whisper_initial_prompt", None)') == 2, \
        "faster-whisper and MLX paths must both read the config"
    assert src.count('{"initial_prompt": initial_prompt} if initial_prompt else {}') == 2
    # Fallback must be explicit, not silent.
    assert "retrying" in src and "language kept" in src


# ------------------------------------------------------------------ dictation

def _dict_engine(language="ro", prompt=None):
    from src.jarvis.dictation.dictation_engine import DictationEngine
    e = DictationEngine.__new__(DictationEngine)
    e._language = language
    e._initial_prompt = prompt
    return e


@pytest.mark.unit
def test_dictation_sends_language_and_prompt():
    from src.jarvis.dictation.dictation_engine import DictationEngine
    e = _dict_engine("ro", PROMPT)
    model = Mock()
    model.transcribe.return_value = (iter([]), Mock())
    DictationEngine._transcribe_faster_whisper(e, model, [0.0] * 16000)
    kw = model.transcribe.call_args.kwargs
    assert kw["language"] == "ro"
    assert kw["initial_prompt"] == PROMPT


@pytest.mark.unit
def test_dictation_omits_prompt_when_unset():
    from src.jarvis.dictation.dictation_engine import DictationEngine
    e = _dict_engine("ro", None)
    model = Mock()
    model.transcribe.return_value = (iter([]), Mock())
    DictationEngine._transcribe_faster_whisper(e, model, [0.0] * 16000)
    assert "initial_prompt" not in model.transcribe.call_args.kwargs


@pytest.mark.unit
def test_dictation_typeerror_fallback_keeps_language():
    from src.jarvis.dictation.dictation_engine import DictationEngine
    e = _dict_engine("ro", PROMPT)
    calls = []

    def transcribe(_a, **kw):
        calls.append(kw)
        if "initial_prompt" in kw:
            raise TypeError("unexpected keyword argument 'initial_prompt'")
        return (iter([]), Mock())

    model = Mock()
    model.transcribe.side_effect = transcribe
    DictationEngine._transcribe_faster_whisper(e, model, [0.0] * 16000)

    assert len(calls) == 2
    assert calls[1]["language"] == "ro"
    assert "initial_prompt" not in calls[1]


@pytest.mark.unit
def test_daemon_threads_initial_prompt_into_dictation():
    src = Path("src/jarvis/daemon.py").read_text(encoding="utf-8")
    assert 'initial_prompt=getattr(cfg, "whisper_initial_prompt", None),' in src
    # Model and lock must stay shared — no second Whisper load.
    assert "whisper_model_ref=lambda: voice_thread.model" in src
    assert "transcribe_lock=voice_thread.transcribe_lock" in src


# ------------------------------------------------------- untouched behaviour

@pytest.mark.unit
def test_audio_pipeline_settings_untouched():
    """This change must not move any audio knob."""
    from src.jarvis.config import load_settings
    s = load_settings()
    assert s.sample_rate == 16000
    assert s.voice_device == "12"
    assert s.voice_min_energy == 0.005
    assert s.vad_enabled is False
    assert s.whisper_model == "large-v3"
    assert s.whisper_language == "ro"
