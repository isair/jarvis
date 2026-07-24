"""Phase 4 Module D — identity / capability registry.

Two guarantees under test:

* ``build_capability_registry`` reports the truth about the running process —
  Supertonic active ⇒ CONFIGURED, Piper ⇒ fallback AVAILABLE, Whisper and the
  chat model ⇒ AVAILABLE, n8n / MCP ⇒ NOT_CONNECTED on an empty cfg, and every
  Phase-4 feature ⇒ DISABLED by default. No status is ever invented and no
  filesystem path or secret leaks into a label or detail.

* ``answer_identity_question`` answers reflexive, second-person questions in
  Romanian, before the LLM — and does so with narrow matchers that never
  swallow "spune-mi despre X" or a "memorează ..." command, return None on no
  match, and stay inert while ``identity_registry_enabled`` is off.
"""

from types import SimpleNamespace

import pytest

from src.jarvis.reply.identity_registry import (
    AVAILABLE,
    BLOCKED,
    CONFIGURED,
    DISABLED,
    NOT_CONNECTED,
    STATUSES,
    Capability,
    answer_identity_question,
    build_capability_registry,
)


# A fake cfg shaped like the live Settings object (Supertonic F5 active, all
# Phase-4 gates off). identity_registry_enabled defaults True so the answer
# helpers can be exercised; individual tests override what they need.
def _cfg(**overrides):
    base = dict(
        tts_engine="supertonic",
        tts_supertonic_voice="F5",
        tts_supertonic_language="ro",
        whisper_model="large-v3",
        ollama_chat_model="gemma3:4b",
        mcps={},
        conversation_learning_enabled=False,
        legacy_knowledge_auto_write_enabled=False,
        owner_profile_enabled=False,
        state_memory_enabled=False,
        memory_require_confirmation=False,
        internet_learning_enabled=False,
        self_eval_enabled=False,
        audit_panel_enabled=False,
        owner_triggered_development_enabled=False,
        development_agent_provider="disabled",
        openai_realtime_enabled=False,
        identity_registry_enabled=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _by_key(cfg):
    return {c.key: c for c in build_capability_registry(cfg)}


# ------------------------------------------------------------------ registry

@pytest.mark.unit
def test_supertonic_configured_when_active():
    cap = _by_key(_cfg())["tts_supertonic"]
    # CONFIGURED is the strongest positive state (present AND active) — it
    # implies available, satisfying "AVAILABLE + CONFIGURED".
    assert cap.status == CONFIGURED
    assert "F5" in cap.detail and "ro" in cap.detail


@pytest.mark.unit
def test_supertonic_available_but_not_configured_when_piper_active():
    cap = _by_key(_cfg(tts_engine="piper"))["tts_supertonic"]
    assert cap.status == AVAILABLE


@pytest.mark.unit
def test_piper_is_available_fallback_when_supertonic_active():
    reg = _by_key(_cfg())
    assert reg["tts_piper"].status == AVAILABLE
    # And Piper becomes the active engine on the default config.
    assert _by_key(_cfg(tts_engine="piper"))["tts_piper"].status == CONFIGURED


@pytest.mark.unit
def test_whisper_and_chat_model_available():
    reg = _by_key(_cfg())
    assert reg["asr_whisper"].status == AVAILABLE
    assert "large-v3" in reg["asr_whisper"].detail
    assert reg["chat_model"].status == AVAILABLE
    assert "gemma3:4b" in reg["chat_model"].detail


@pytest.mark.unit
def test_n8n_and_mcp_not_connected_when_mcps_empty():
    reg = _by_key(_cfg())
    assert reg["n8n"].status == NOT_CONNECTED
    assert reg["mcp"].status == NOT_CONNECTED


@pytest.mark.unit
def test_mcp_and_n8n_configured_when_present():
    reg = _by_key(_cfg(mcps={"n8n": {"url": "x"}, "other": {}}))
    assert reg["mcp"].status == CONFIGURED
    assert reg["n8n"].status == CONFIGURED
    assert "2" in reg["mcp"].detail  # count surfaced, deterministically


@pytest.mark.unit
def test_conversation_learning_disabled_by_default():
    assert _by_key(_cfg())["conversation_learning"].status == DISABLED
    assert _by_key(_cfg(conversation_learning_enabled=True))[
        "conversation_learning"].status == AVAILABLE


@pytest.mark.unit
def test_legacy_auto_write_disabled_by_default():
    assert _by_key(_cfg())["legacy_knowledge_auto_write"].status == DISABLED
    assert _by_key(_cfg(legacy_knowledge_auto_write_enabled=True))[
        "legacy_knowledge_auto_write"].status == AVAILABLE


@pytest.mark.unit
def test_owner_triggered_development_disabled_unless_enabled_and_provider():
    # both off → DISABLED
    assert _by_key(_cfg())["owner_triggered_development"].status == DISABLED
    # switch on but provider still disabled → still DISABLED
    assert _by_key(_cfg(owner_triggered_development_enabled=True))[
        "owner_triggered_development"].status == DISABLED
    # both satisfied → CONFIGURED
    cap = _by_key(_cfg(owner_triggered_development_enabled=True,
                       development_agent_provider="claude_cli"))[
        "owner_triggered_development"]
    assert cap.status == CONFIGURED
    assert "claude_cli" in cap.detail


@pytest.mark.unit
def test_premium_voice_blocked_when_enabled_disabled_otherwise():
    assert _by_key(_cfg())["premium_voice"].status == DISABLED
    assert _by_key(_cfg(openai_realtime_enabled=True))[
        "premium_voice"].status == BLOCKED


@pytest.mark.unit
def test_every_status_is_in_the_declared_enum():
    for cap in build_capability_registry(_cfg()):
        assert isinstance(cap, Capability)
        assert cap.status in STATUSES


@pytest.mark.unit
def test_registry_leaks_no_secret_or_absolute_path():
    """Detail fields must never carry a runtime path or a secret-shaped token."""
    cfg = _cfg(
        tts_supertonic_runtime_path=r"C:\Users\Administrator\Downloads\cora-labs",
        db_path=r"C:\Users\Administrator\.local\share\jarvis\jarvis.db",
    )
    for cap in build_capability_registry(cfg):
        blob = f"{cap.label} {cap.detail}"
        assert "C:\\" not in blob and ":/" not in blob
        assert "cora-labs" not in blob
        assert ".db" not in blob
        assert "[REDACTED" not in blob  # nothing secret-shaped got that far


# ---------------------------------------------------------- identity answers

@pytest.mark.unit
def test_who_answer_is_romanian_and_names_cora():
    a = answer_identity_question("cine ești", _cfg())
    assert a is not None
    assert "Cora" in a
    assert "local" in a.lower()
    for english in ("I am", "assistant that", "Hello", "you are"):
        assert english not in a


@pytest.mark.unit
@pytest.mark.parametrize("q", [
    "cine ești",
    "cine esti tu",       # diacritics dropped by ASR
    "tu cine ești",
    "ce ești tu",
])
def test_who_variants_matched(q):
    assert answer_identity_question(q, _cfg()) is not None


@pytest.mark.unit
def test_capabilities_answer_deterministic_and_romanian():
    a = answer_identity_question("ce poți face", _cfg())
    assert a is not None
    assert "Pot să" in a
    # active systems are enumerated from the live registry, never invented
    assert "Whisper" in a and "Supertonic" in a


@pytest.mark.unit
def test_voice_answer_reflects_active_supertonic():
    a = answer_identity_question("ce voce folosești", _cfg())
    assert a is not None
    assert "Supertonic" in a and "F5" in a and "Piper" in a


@pytest.mark.unit
def test_voice_answer_reflects_piper_when_active():
    a = answer_identity_question("cu ce voce vorbești", _cfg(tts_engine="piper"))
    assert a is not None and "Piper" in a


@pytest.mark.unit
def test_model_answer_names_the_configured_model():
    a = answer_identity_question("ce model folosești", _cfg())
    assert a is not None
    assert "gemma3:4b" in a and "Ollama" in a


@pytest.mark.unit
def test_memory_answer_is_romanian_and_reflects_flags():
    off = answer_identity_question("cum funcționează memoria ta", _cfg())
    assert off is not None
    assert "dezactivată" in off and "local" in off.lower()
    on = answer_identity_question(
        "cum funcționează memoria ta",
        _cfg(state_memory_enabled=True, memory_require_confirmation=True),
    )
    assert "durată" in on and "confirmarea" in on


@pytest.mark.unit
def test_connected_answer_reports_n8n_not_connected():
    a = answer_identity_question("la ce sisteme ești conectată", _cfg())
    assert a is not None
    assert "n8n" in a and "nu sunt conectate" in a
    assert "local" in a.lower()


# --------------------------------------------------- narrowness / no-hijack

@pytest.mark.unit
@pytest.mark.parametrize("q", [
    "spune-mi despre pisici",
    "memorează că îmi place cafeaua",
    "memorează ce poți face",          # command wins over the capability stem
    "notează că ești conectată la n8n",
    "explică pe scurt ce este un model de limbaj",
    "cum funcționează un motor cu ardere internă",
    "povestește-mi despre vocea umană",
    "spune-mi ora",
    "ce zi este astăzi",
    "",
    "   ",
])
def test_non_identity_questions_fall_through_to_the_model(q):
    assert answer_identity_question(q, _cfg()) is None


@pytest.mark.unit
def test_inert_when_flag_disabled():
    """With the gate off the module does nothing, even for a clear match."""
    assert answer_identity_question("cine ești", _cfg(identity_registry_enabled=False)) is None
    assert answer_identity_question("ce poți face", _cfg(identity_registry_enabled=False)) is None


@pytest.mark.unit
def test_answer_never_leaks_a_secret_shaped_model_name():
    """A secret-shaped chat model name is redacted out of the answer."""
    secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    a = answer_identity_question("ce model folosești", _cfg(ollama_chat_model=secret))
    assert a is not None
    assert secret not in a
    assert "[REDACTED" in a  # the scrub gate fired


@pytest.mark.unit
def test_answers_are_pure_no_mutation_of_cfg():
    cfg = _cfg()
    before = dict(cfg.__dict__)
    for q in ("cine ești", "ce poți face", "ce voce folosești",
              "ce model folosești", "cum funcționează memoria ta",
              "la ce ești conectată"):
        answer_identity_question(q, cfg)
    assert cfg.__dict__ == before
