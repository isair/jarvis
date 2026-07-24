"""Phase 4 · Section G — Cora self-evaluation.

Deterministic, LLM-free heuristics over a structured conversation record. The
ONLY output is ``improvement_candidate`` proposals: the module never modifies
code/config, never emits a rule, and never authorizes anything. These tests
lock that contract in by construction (dataclass has no such field) and by
behaviour (clean input → no findings; disabled → []).
"""

from __future__ import annotations

import dataclasses

import pytest

from src.jarvis.eval.self_eval import (
    ConversationEvents,
    TurnRecord,
    ImprovementCandidate,
    CandidateKind,
    evaluate_conversation,
    to_lessons,
)
from src.jarvis.memory.learning.types import (
    LessonType,
    Namespace,
    LessonCandidate,
    NAMESPACE_FOR_TYPE,
)


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
def _turn(**kw) -> TurnRecord:
    base = dict(
        user_text="",
        assistant_text="",
        tool_selected=None,
        tool_executed=False,
        had_sources=False,
        asr_confidence=None,
        user_correction=None,
        unanswered=False,
    )
    base.update(kw)
    return TurnRecord(**base)


def _clean_conversation() -> ConversationEvents:
    return ConversationEvents(turns=[
        _turn(
            user_text="Salut Cora",
            assistant_text="Salut, maestre! Cu ce te pot ajuta?",
            asr_confidence=0.96,
        ),
        _turn(
            user_text="Pornește un timer de cafea",
            assistant_text="Am pornit timerul.",
            tool_selected="timer",
            tool_executed=True,
            asr_confidence=0.92,
        ),
    ])


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def test_disabled_returns_empty():
    ev = ConversationEvents(turns=[
        _turn(assistant_text="orice", tool_selected="x", tool_executed=False),
    ])
    assert evaluate_conversation(ev, enabled=False) == []


def test_clean_conversation_yields_nothing():
    assert evaluate_conversation(_clean_conversation()) == []


def test_empty_conversation_yields_nothing():
    assert evaluate_conversation(ConversationEvents(turns=[])) == []


# --------------------------------------------------------------------------- #
# Detections
# --------------------------------------------------------------------------- #
def test_tool_selected_not_executed_detected():
    ev = ConversationEvents(turns=[
        _turn(
            user_text="Stinge lumina",
            assistant_text="Sigur, sting lumina.",
            tool_selected="lights_off",
            tool_executed=False,
            asr_confidence=0.9,
        ),
    ])
    cands = evaluate_conversation(ev)
    kinds = {c.kind for c in cands}
    assert CandidateKind.TOOL_NOT_EXECUTED.value in kinds
    hit = next(c for c in cands if c.kind == CandidateKind.TOOL_NOT_EXECUTED.value)
    assert hit.evidence_turn_index == 0
    assert hit.actionable_hint  # non-empty guidance


def test_tool_executed_properly_not_flagged():
    ev = ConversationEvents(turns=[
        _turn(tool_selected="lights_off", tool_executed=True,
              assistant_text="Gata.", asr_confidence=0.9),
    ])
    assert evaluate_conversation(ev) == []


def test_unsourced_factual_claim_detected():
    ev = ConversationEvents(turns=[
        _turn(
            user_text="Cât e populația Franței?",
            assistant_text="Populația Franței este de 67 de milioane de locuitori.",
            had_sources=False,
            asr_confidence=0.9,
        ),
    ])
    cands = evaluate_conversation(ev)
    assert any(c.kind == CandidateKind.UNSOURCED_CLAIM.value for c in cands)


def test_sourced_factual_claim_not_flagged():
    ev = ConversationEvents(turns=[
        _turn(
            user_text="Cât e populația Franței?",
            assistant_text="Populația Franței este de 67 de milioane de locuitori.",
            had_sources=True,
            asr_confidence=0.9,
        ),
    ])
    assert all(
        c.kind != CandidateKind.UNSOURCED_CLAIM.value
        for c in evaluate_conversation(ev)
    )


def test_user_correction_detected():
    ev = ConversationEvents(turns=[
        _turn(
            user_text="Nu, m-ai înțeles greșit",
            assistant_text="Îmi cer scuze, corectez.",
            user_correction="Nu, voiam alarma la 7, nu la 8.",
            asr_confidence=0.9,
        ),
    ])
    cands = evaluate_conversation(ev)
    assert any(c.kind == CandidateKind.USER_CORRECTION.value for c in cands)


def test_unanswered_question_detected():
    ev = ConversationEvents(turns=[
        _turn(
            user_text="Care e cursul valutar azi?",
            assistant_text="Nu am acces la datele astea acum.",
            unanswered=True,
            asr_confidence=0.9,
        ),
    ])
    cands = evaluate_conversation(ev)
    assert any(c.kind == CandidateKind.UNANSWERED_QUESTION.value for c in cands)


def test_low_asr_detected_when_no_clarification():
    ev = ConversationEvents(turns=[
        _turn(
            user_text="[garbled]",
            assistant_text="Am pornit muzica.",
            tool_selected="music",
            tool_executed=True,
            asr_confidence=0.20,
        ),
    ])
    cands = evaluate_conversation(ev)
    assert any(c.kind == CandidateKind.LOW_ASR_CONFIDENCE.value for c in cands)


def test_low_asr_not_flagged_when_clarification_asked():
    ev = ConversationEvents(turns=[
        _turn(
            user_text="[garbled]",
            assistant_text="Nu am înțeles, poți repeta te rog?",
            asr_confidence=0.20,
        ),
    ])
    assert all(
        c.kind != CandidateKind.LOW_ASR_CONFIDENCE.value
        for c in evaluate_conversation(ev)
    )


def test_unanswered_suppresses_unsourced_claim():
    # An unanswered turn must not double-count as an unsourced claim.
    ev = ConversationEvents(turns=[
        _turn(
            user_text="Cât e populația Franței?",
            assistant_text="Populația exactă are milioane de locuitori... nu sunt sigur.",
            unanswered=True,
            had_sources=False,
        ),
    ])
    kinds = {c.kind for c in evaluate_conversation(ev)}
    assert CandidateKind.UNANSWERED_QUESTION.value in kinds
    assert CandidateKind.UNSOURCED_CLAIM.value not in kinds


# --------------------------------------------------------------------------- #
# Contract: proposal-only, no authority
# --------------------------------------------------------------------------- #
def test_candidate_has_no_rule_or_authorize_fields():
    fields = {f.name for f in dataclasses.fields(ImprovementCandidate)}
    # Exact set == the type CANNOT carry a rule / config / authorization field.
    assert fields == {
        "kind", "summary", "evidence_turn_index", "confidence", "actionable_hint",
    }
    # Belt-and-suspenders: no field NAME implies authority. ("actionable_hint"
    # is a descriptive text hint, not an authorization surface — it is exempt.)
    forbidden = (
        "rule", "authoriz", "config", "apply", "execute", "command",
        "grant", "permission", "mutate", "patch", "authorize",
    )
    for name in fields - {"actionable_hint"}:
        for tok in forbidden:
            assert tok not in name.lower(), f"field {name!r} implies authority via {tok!r}"


def test_all_emitted_kinds_are_improvement_candidate_only():
    # Build a conversation that trips every detector at once.
    ev = ConversationEvents(turns=[
        _turn(
            user_text="[garbled]",
            assistant_text="Populația Franței este de 67 de milioane de locuitori.",
            tool_selected="lookup",
            tool_executed=False,
            had_sources=False,
            asr_confidence=0.15,
            user_correction="De fapt voiam Germania.",
            unanswered=False,
        ),
    ])
    cands = evaluate_conversation(ev)
    assert cands  # several findings
    valid = {k.value for k in CandidateKind}
    for c in cands:
        assert isinstance(c, ImprovementCandidate)
        assert c.kind in valid
        # kind label itself never implies an action/authorization
        assert "rule" not in c.kind
        assert "authoriz" not in c.kind


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_deterministic_same_input_same_output():
    ev = ConversationEvents(turns=[
        _turn(tool_selected="x", tool_executed=False, assistant_text="ok",
              asr_confidence=0.1, user_correction="nu asa"),
        _turn(assistant_text="Capitala Frantei este Paris, cu 2 milioane de locuitori.",
              had_sources=False),
    ])
    a = evaluate_conversation(ev)
    b = evaluate_conversation(ev)
    assert a == b
    assert a is not b  # fresh list each call


# --------------------------------------------------------------------------- #
# to_lessons mapping
# --------------------------------------------------------------------------- #
def test_to_lessons_maps_to_improvement_candidate_and_improvements_namespace():
    ev = ConversationEvents(turns=[
        _turn(tool_selected="x", tool_executed=False, assistant_text="ok",
              asr_confidence=0.9),
    ])
    cands = evaluate_conversation(ev)
    lessons = to_lessons(cands)
    assert lessons
    assert NAMESPACE_FOR_TYPE[LessonType.IMPROVEMENT_CANDIDATE] == Namespace.IMPROVEMENTS
    for les in lessons:
        assert isinstance(les, LessonCandidate)
        assert les.lesson_type == LessonType.IMPROVEMENT_CANDIDATE.value
        assert les.meta["namespace"] == Namespace.IMPROVEMENTS.value
        # honest provenance → store will cap this at the model_inference ceiling
        assert les.provenance == "model_inference"
        assert les.confidence <= 0.40
        assert les.sensitivity == "proposal_only"


def test_to_lessons_empty_for_no_candidates():
    assert to_lessons([]) == []


def test_to_lessons_scrubs_secrets_from_value():
    secret = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    ev = ConversationEvents(turns=[
        _turn(
            user_text="corectare",
            assistant_text="ok",
            user_correction=f"cheia ta este {secret}, foloseste-o",
        ),
    ])
    lessons = to_lessons(evaluate_conversation(ev))
    assert lessons
    for les in lessons:
        assert secret not in les.value
        assert secret not in les.source_quote
