"""Deterministic self-evaluation heuristics for finished conversations.

Phase 4 · Section G. After a conversation ends, Cora can look back over a
structured record of what happened and propose ways it could have done better.
This is done with **plain, deterministic heuristics — no LLM call** — so the
same input always yields the same findings and nothing here depends on network
or model state.

Hard safety contract (enforced by construction *and* tested):

  * The ONLY thing this module produces is a list of
    :class:`ImprovementCandidate` — proposal-only findings.
  * :class:`ImprovementCandidate` deliberately has **no** field that could
    encode a rule, a config change, a command, or an authorization. There is
    nothing to "apply". A downstream reviewer decides what, if anything, to do.
  * :func:`evaluate_conversation` returns ``[]`` when ``enabled`` is False; the
    intended call site is additionally gated by ``self_eval_enabled`` (default
    OFF), so the module is inert at runtime until the owner opts in.
  * User/assistant text that leaks into a finding's summary or a mapped lesson
    is scrubbed for secrets and length-capped.

The optional :func:`to_lessons` maps findings onto
:class:`~jarvis.memory.learning.types.LessonCandidate` records typed as
``IMPROVEMENT_CANDIDATE`` in the ``IMPROVEMENTS`` namespace, for a *later*
persistence step. This module never touches the database itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

from ..memory.learning.types import (
    CONFIDENCE_CAPS,
    LessonCandidate,
    LessonType,
    NAMESPACE_FOR_TYPE,
    Namespace,
)

try:  # reuse the repo's secret scrubber when available
    from ..utils.redact import scrub_secrets as _scrub
except Exception:  # pragma: no cover - defensive
    def _scrub(text: str) -> str:  # type: ignore
        return text


__all__ = [
    "CandidateKind",
    "ConversationEvents",
    "ImprovementCandidate",
    "TurnRecord",
    "evaluate_conversation",
    "to_lessons",
]

# A turn whose ASR confidence is below this and where Cora did NOT ask for a
# clarification is a candidate for "confirm before acting".
LOW_ASR_THRESHOLD = 0.55

# How much raw text may echo into a finding / lesson (after scrubbing).
_SNIPPET_LIMIT = 100


class CandidateKind(str, Enum):
    """The fixed vocabulary of self-eval findings.

    Every value is a *description of a shortcoming* — none of them is or implies
    a rule, a config change, or an authorization.
    """

    TOOL_NOT_EXECUTED = "tool_selected_not_executed"
    UNSOURCED_CLAIM = "unsourced_claim"
    USER_CORRECTION = "user_correction"
    UNANSWERED_QUESTION = "unanswered_question"
    LOW_ASR_CONFIDENCE = "low_asr_confidence"


# Deterministic per-kind confidence for the *proposal*. These describe how sure
# the heuristic is that a shortcoming occurred — never an authority to act.
_KIND_CONFIDENCE = {
    CandidateKind.TOOL_NOT_EXECUTED: 0.80,
    CandidateKind.USER_CORRECTION: 0.90,
    CandidateKind.UNANSWERED_QUESTION: 0.70,
    CandidateKind.UNSOURCED_CLAIM: 0.55,
    CandidateKind.LOW_ASR_CONFIDENCE: 0.55,
}

_KIND_HINT = {
    CandidateKind.TOOL_NOT_EXECUTED: (
        "A tool was chosen but never ran — check tool-call parsing / execution "
        "wiring so a selected tool is always dispatched or an error is surfaced."
    ),
    CandidateKind.UNSOURCED_CLAIM: (
        "A factual assertion was made without any source — prefer a lookup tool "
        "or retrieved context, or qualify the answer as uncertain."
    ),
    CandidateKind.USER_CORRECTION: (
        "The user had to correct Cora — capture the correction as a lesson and "
        "review why the first answer diverged."
    ),
    CandidateKind.UNANSWERED_QUESTION: (
        "The question went unanswered / data was missing — route to a knowledge "
        "or tool lookup, or ask a focused clarifying question."
    ),
    CandidateKind.LOW_ASR_CONFIDENCE: (
        "Speech was recognised with low confidence yet Cora acted anyway — "
        "confirm the transcription before taking an action."
    ),
}


# --- claim / clarification detection ------------------------------------- #
# A conservative "this looks like an external, verifiable factual claim" cue
# set (RO + EN). Kept selective on purpose so ordinary chit-chat, arithmetic,
# and tool acknowledgements ("Am pornit timerul") do NOT trip it.
_FACTUAL_CLAIM = re.compile(
    r"(?i)("
    r"capital[aă]|popula[țt]i|locuitori|suprafa[țt]|"
    r"conform|potrivit|studiile\s+arat[aă]|cercet[aă]ril|statistic|"
    r"rata\s|la\s+sut[aă]|procent|"
    r"[îi]n\s+anul\s+\d|anul\s+\d{3,4}|s-a\s+n[aă]scut\s+[îi]n|"
    r"a\s+fost\s+(fondat|[îi]nfiin[țt]at|construit)|"
    r"the\s+capital|population|according\s+to|studies\s+show|research\s+shows|"
    r"in\s+the\s+year|founded\s+in|was\s+born\s+in|percent"
    r"|\d+\s*(%|km|km/h|kg|°c|grade|milioane|miliarde|metri|tone|locuitori)"
    r")"
)

_CLARIFY = re.compile(
    r"(?i)("
    r"nu\s+am\s+[îi]n[țt]eles|po[țt]i\s+repeta|ai\s+putea\s+repeta|"
    r"repet[aă]|spune\s+din\s+nou|mai\s+spune\s+o\s+dat[aă]|clarif|"
    r"didn'?t\s+catch|could\s+you\s+repeat|say\s+again|repeat\s+that"
    r")"
)


@dataclass
class TurnRecord:
    """One assistant turn's worth of structured, post-hoc signals.

    All fields are facts already known by the time the conversation ends; none
    require an LLM to compute. Everything is optional-friendly so a caller can
    populate only what it tracks.
    """

    user_text: str = ""
    assistant_text: str = ""
    tool_selected: Optional[str] = None
    tool_executed: bool = False
    had_sources: bool = False
    asr_confidence: Optional[float] = None
    user_correction: Optional[str] = None
    unanswered: bool = False


@dataclass
class ConversationEvents:
    """A finished conversation as an ordered list of :class:`TurnRecord`."""

    turns: List[TurnRecord] = field(default_factory=list)


@dataclass
class ImprovementCandidate:
    """A proposal-only self-eval finding.

    There is deliberately no ``rule`` / ``action`` / ``authorize`` / ``config``
    field: the type cannot carry authority. ``confidence`` is how sure the
    heuristic is that the shortcoming occurred, not permission to act.
    """

    kind: str
    summary: str
    evidence_turn_index: int
    confidence: float
    actionable_hint: str


def _snippet(text: Optional[str], limit: int = _SNIPPET_LIMIT) -> str:
    """Scrub secrets, collapse whitespace, and cap length for safe echoing."""
    cleaned = " ".join(_scrub(text or "").split())
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def _looks_like_factual_claim(text: str) -> bool:
    """Conservative, deterministic 'external factual claim' detector."""
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith("?"):  # a question is not an assertion
        return False
    return bool(_FACTUAL_CLAIM.search(t))


def _asked_for_clarification(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if "?" in t:
        return True
    return bool(_CLARIFY.search(t))


def _make(kind: CandidateKind, idx: int, summary: str) -> ImprovementCandidate:
    return ImprovementCandidate(
        kind=kind.value,
        summary=summary,
        evidence_turn_index=idx,
        confidence=_KIND_CONFIDENCE[kind],
        actionable_hint=_KIND_HINT[kind],
    )


def evaluate_conversation(
    events: ConversationEvents,
    *,
    enabled: bool = True,
) -> List[ImprovementCandidate]:
    """Return proposal-only improvement findings for a finished conversation.

    Deterministic and LLM-free. Returns ``[]`` when ``enabled`` is False (the
    module is inert unless the caller opts in). Detections, per turn, in a fixed
    order:

      1. a tool was *selected but never executed* (problem #6);
      2. an *unsourced factual claim* (``had_sources`` False on a claim turn);
      3. a *user correction* was issued;
      4. an *unanswered / missing-data* question;
      5. a *low-ASR* turn where Cora acted without confirming.

    The output never contains a rule, config change, or authorization — the
    return type simply cannot express one.
    """
    if not enabled:
        return []

    turns = getattr(events, "turns", None)
    if turns is None:
        turns = list(events)  # tolerate a bare iterable of TurnRecord

    out: List[ImprovementCandidate] = []
    for i, t in enumerate(turns):
        # 1) tool chosen but not dispatched
        if t.tool_selected and not t.tool_executed:
            out.append(_make(
                CandidateKind.TOOL_NOT_EXECUTED, i,
                f"Tool '{_snippet(str(t.tool_selected), 40)}' was selected but "
                f"never executed.",
            ))

        # 2) factual claim without any source (only if it actually answered)
        if (
            not t.unanswered
            and not t.had_sources
            and _looks_like_factual_claim(t.assistant_text)
        ):
            out.append(_make(
                CandidateKind.UNSOURCED_CLAIM, i,
                f"Unsourced factual claim: \"{_snippet(t.assistant_text)}\".",
            ))

        # 3) explicit user correction
        if t.user_correction and str(t.user_correction).strip():
            out.append(_make(
                CandidateKind.USER_CORRECTION, i,
                f"User corrected Cora: \"{_snippet(t.user_correction)}\".",
            ))

        # 4) unanswered / missing-data question
        if t.unanswered:
            out.append(_make(
                CandidateKind.UNANSWERED_QUESTION, i,
                f"Question left unanswered / data missing for: "
                f"\"{_snippet(t.user_text)}\".",
            ))

        # 5) low ASR confidence, acted without confirming
        if (
            t.asr_confidence is not None
            and float(t.asr_confidence) < LOW_ASR_THRESHOLD
            and not _asked_for_clarification(t.assistant_text)
        ):
            out.append(_make(
                CandidateKind.LOW_ASR_CONFIDENCE, i,
                f"Acted on low-confidence speech "
                f"(ASR={float(t.asr_confidence):.2f}) without confirming.",
            ))

    return out


def to_lessons(cands: Sequence[ImprovementCandidate]) -> List[LessonCandidate]:
    """Map findings to ``IMPROVEMENT_CANDIDATE`` lesson candidates (no persist).

    Provenance is ``model_inference`` on purpose: these are automated heuristic
    inferences, not user-confirmed facts, so the learning store will cap their
    confidence at the model-inference ceiling. Namespace resolves to
    ``IMPROVEMENTS`` via :data:`NAMESPACE_FOR_TYPE` and is echoed into ``meta``
    for callers that need it before a :class:`Lesson` is constructed.
    """
    ns = NAMESPACE_FOR_TYPE.get(
        LessonType.IMPROVEMENT_CANDIDATE, Namespace.IMPROVEMENTS
    ).value
    cap = CONFIDENCE_CAPS.get("model_inference", 0.40)

    lessons: List[LessonCandidate] = []
    for c in cands:
        lessons.append(LessonCandidate(
            lesson_type=LessonType.IMPROVEMENT_CANDIDATE.value,
            subject_key=c.kind,
            value=_snippet(f"{c.summary} {c.actionable_hint}", 480),
            source_quote=f"self_eval:turn={c.evidence_turn_index}",
            confidence=min(float(c.confidence), float(cap)),
            provenance="model_inference",
            turn_id=str(c.evidence_turn_index),
            sensitivity="proposal_only",
            meta={
                "namespace": ns,
                "kind": c.kind,
                "evidence_turn_index": c.evidence_turn_index,
                "actionable_hint": c.actionable_hint,
            },
        ))
    return lessons
