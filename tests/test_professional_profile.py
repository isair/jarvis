"""Professional assistant profile + model-registry additions.

The professional persona is opt-in and fully reversible: with
``assistant_style`` unset (or set to anything unrecognised) the upstream
British-butler persona must come back byte-identical.
"""

import json

import pytest

from src.jarvis.system_prompt import (
    _PROFESSIONAL_PROMPT_TEMPLATE,
    _SYSTEM_PROMPT_TEMPLATE,
    build_system_prompt,
)
from src.jarvis.reply.prompts.model_variants import ModelSize, detect_model_size


# ------------------------------------------------------------ persona switch

@pytest.mark.unit
def test_default_and_unknown_styles_return_upstream_butler():
    upstream = _SYSTEM_PROMPT_TEMPLATE.format(name="Jarvis")
    assert build_system_prompt("Jarvis") == upstream
    assert build_system_prompt("Jarvis", "butler") == upstream
    # A typo must degrade to upstream, never to something undefined.
    assert build_system_prompt("Jarvis", "profesional") == upstream
    assert build_system_prompt("Jarvis", "") == upstream


@pytest.mark.unit
def test_professional_style_returns_the_professional_template():
    p = build_system_prompt("Jarvis", "professional")
    assert p == _PROFESSIONAL_PROMPT_TEMPLATE.format(name="Jarvis")
    assert p != _SYSTEM_PROMPT_TEMPLATE.format(name="Jarvis")


@pytest.mark.unit
def test_professional_drops_the_butler_character():
    p = build_system_prompt("Jarvis", "professional").lower()
    for trait in ("british butler", "sarcastic", "deadpan", "quip", "witty"):
        assert trait not in p, f"professional persona must not be {trait}"


@pytest.mark.unit
def test_professional_bans_filler_openers():
    p = build_system_prompt("Jarvis", "professional")
    for phrase in ("Voi proceda", "Cu siguranță", "Sunt aici să te ajut"):
        assert phrase in p, f"{phrase!r} must be listed as banned"


@pytest.mark.unit
@pytest.mark.parametrize("requirement", [
    "answer first",          # information-first ordering
    "already given you",     # don't re-ask answered questions
    "assumption",            # separate fact from assumption
    "do not know",           # admit ignorance
    "confirmed it",          # no unconfirmed success claims
    "order asked",           # multi-step ordering
    "read aloud",            # voice-appropriate output
])
def test_professional_covers_each_required_behaviour(requirement):
    assert requirement in build_system_prompt("Jarvis", "professional").lower()


@pytest.mark.unit
def test_professional_does_not_suppress_tool_calls():
    """Regression: the first draft's "no JSON / no code fences" rule made both
    gemma4:e2b and qwen3.5:9b stop emitting tool calls entirely — gemma
    answered "vremea este prezentă acum" with no tool invoked at all."""
    p = build_system_prompt("Jarvis", "professional").lower()
    assert "never suppress or alter a tool call" in p
    assert "never call a tool with empty arguments" in p
    # The formatting rules must be explicitly scoped to spoken output.
    assert "apply" in p and "spoken to the user" in p


@pytest.mark.unit
def test_persona_uses_configured_assistant_name():
    assert "Friday" in build_system_prompt("Friday", "professional")


# ------------------------------------------------------------- config wiring

@pytest.mark.unit
def test_assistant_style_and_thinking_reach_settings(tmp_path, monkeypatch):
    """assistant_style, llm_thinking_enabled and intent_judge_thinking_enabled
    must be real Settings fields — the thinking flags previously existed only in
    get_default_config(), so a configured `true` was silently ignored."""
    from src.jarvis.config import load_settings

    cfg_path = tmp_path / "config.json"

    def _load(payload):
        cfg_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))
        return load_settings()

    s = _load({})
    assert s.assistant_style == "butler"
    assert s.llm_thinking_enabled is False
    assert s.intent_judge_thinking_enabled is False

    s = _load({"assistant_style": "professional",
               "llm_thinking_enabled": True,
               "intent_judge_thinking_enabled": True})
    assert s.assistant_style == "professional"
    assert s.llm_thinking_enabled is True, "configured value must not be ignored"
    assert s.intent_judge_thinking_enabled is True

    # Unrecognised style falls back to upstream.
    s = _load({"assistant_style": "pirate"})
    assert s.assistant_style == "butler"


@pytest.mark.unit
def test_engine_passes_assistant_style_to_persona_builder():
    from pathlib import Path
    src = Path("src/jarvis/reply/engine.py").read_text(encoding="utf-8")
    assert 'style=getattr(cfg, "assistant_style", "butler")' in src


# ----------------------------------------------------------- model registry

@pytest.mark.unit
def test_qwen35_registered_as_supported_chat_model():
    from src.jarvis.config import SUPPORTED_CHAT_MODELS, get_supported_model_ids

    assert "qwen3.5:9b" in SUPPORTED_CHAT_MODELS
    assert "qwen3.5:9b" in get_supported_model_ids()
    entry = SUPPORTED_CHAT_MODELS["qwen3.5:9b"]
    for field in ("name", "description", "size", "vram"):
        assert entry.get(field), f"{field} must be populated"
    # The measured thinking-latency trap must stay documented.
    assert "thinking" in entry["description"].lower()


@pytest.mark.unit
@pytest.mark.parametrize("model,expected", [
    ("qwen3.5:9b", ModelSize.LARGE),
    ("gemma4:e2b", ModelSize.SMALL),
    ("gpt-oss:20b", ModelSize.LARGE),
    ("llama3:7b", ModelSize.SMALL),
    (None, ModelSize.LARGE),
])
def test_model_size_classification(model, expected):
    assert detect_model_size(model) == expected


@pytest.mark.unit
def test_every_supported_model_classifies_without_error():
    from src.jarvis.config import SUPPORTED_CHAT_MODELS

    for model_id in SUPPORTED_CHAT_MODELS:
        assert detect_model_size(model_id) in (ModelSize.SMALL, ModelSize.LARGE)
