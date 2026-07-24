"""Phase 4 · Section D wiring — identity handler reachable via try_local_answer.

Confirms the local_answers integration: identity questions are answered when
identity_registry_enabled is set, inert otherwise, and existing local answers
(arithmetic) are unaffected.
"""

from __future__ import annotations

from types import SimpleNamespace

from jarvis.reply.local_answers import try_local_answer


def _cfg(enabled: bool):
    # build_capability_registry uses getattr with safe defaults, so a minimal
    # namespace suffices; provide a couple of real fields for a realistic answer.
    return SimpleNamespace(
        identity_registry_enabled=enabled,
        tts_engine="supertonic",
        tts_supertonic_voice="F5",
        ollama_chat_model="gemma4:e2b",
        whisper_model="large-v3",
        mcps={},
    )


def test_identity_answered_when_enabled():
    ans = try_local_answer("cine ești?", cfg=_cfg(True))
    assert ans is not None and isinstance(ans, str) and ans.strip()


def test_identity_inert_when_disabled():
    assert try_local_answer("cine ești?", cfg=_cfg(False)) is None


def test_identity_inert_without_cfg_backward_compat():
    # Old call sites that pass no cfg must behave exactly as before.
    assert try_local_answer("cine ești?") is None


def test_arithmetic_still_works_with_cfg():
    # Regression: threading cfg must not break the existing local answers.
    ans = try_local_answer("cât fac 2 plus 2", cfg=_cfg(True))
    assert ans is not None and "4" in ans


def test_command_prefix_not_hijacked_as_identity():
    # "spune-mi despre..." and "memorează..." must fall through to the pipeline.
    assert try_local_answer("spune-mi despre pisici", cfg=_cfg(True)) is None
    assert try_local_answer("memorează că prefer cafea", cfg=_cfg(True)) is None
