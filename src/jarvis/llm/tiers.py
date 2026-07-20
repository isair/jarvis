"""The two-tier model system.

Jarvis runs every LLM context on one of two models:

- ``Tier.FAST`` — the small, warm, low-latency model behind real-time
  classification passes. These contexts take a few thousand tokens in and
  emit tiny strict-JSON answers, so latency dominates and a ~2B model is
  ideal.
- ``Tier.CHAT`` — the capable model that writes replies, plans,
  summarises, and extracts knowledge. Long-form output; quality dominates.

The Model tiers table in ``llm.spec.md`` is the authoritative list of
which context runs on which tier; docstrings elsewhere point here rather
than re-enumerating it.

Both fields are fully resolved at config load (``fast_model`` /
``llm_chat_model`` always hold a provider-valid model name), so resolution
here is a plain field read. Keeping it behind one function means every
context states its tier instead of inventing a fallback chain, and any
future routing logic lands in exactly one place.
"""

from __future__ import annotations

from enum import Enum


class Tier(Enum):
    """Which of the two models a context runs on."""

    FAST = "fast"
    CHAT = "chat"


def resolve_model(cfg, tier: Tier) -> str:
    """Return the model name for ``tier`` under the active settings.

    ``Tier.FAST`` reads ``cfg.fast_model``; ``Tier.CHAT`` reads
    ``cfg.llm_chat_model``. An empty fast model (possible only on
    hand-built cfg objects — config load always resolves it) falls back
    to the chat model so a context never receives an empty name.
    """
    chat = str(getattr(cfg, "llm_chat_model", "") or "").strip()
    if tier is Tier.FAST:
        return str(getattr(cfg, "fast_model", "") or "").strip() or chat
    return chat
