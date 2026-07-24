"""Cora self-evaluation (Phase 4 · Section G).

Deterministic, LLM-free heuristics that read a *finished* conversation record
and propose ``improvement_candidate`` findings. The module has no authority: it
never modifies code, config, prompts, or models, never emits a rule, and never
authorizes a tool. Its output is proposal-only and is meant to be surfaced /
persisted for the owner to review, nothing else.

Inert unless the caller opts in: ``evaluate_conversation`` returns ``[]`` when
``enabled`` is False, and the intended call site is gated by the
``self_eval_enabled`` config flag (default OFF).
"""

from .self_eval import (
    CandidateKind,
    ConversationEvents,
    ImprovementCandidate,
    TurnRecord,
    evaluate_conversation,
    to_lessons,
)

__all__ = [
    "CandidateKind",
    "ConversationEvents",
    "ImprovementCandidate",
    "TurnRecord",
    "evaluate_conversation",
    "to_lessons",
]
