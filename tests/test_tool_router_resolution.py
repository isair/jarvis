"""Tool routing runs on the fast tier.

The reply engine and the listener warmup path both need to pick the model
used for LLM-based tool selection, and they MUST pick the same one — if they
diverge, warmup loads the wrong model and the first real routing call eats a
cold-start stall. Both resolve through the single tier helper
(``resolve_model(cfg, Tier.FAST)``), which these tests exercise directly.

The key property: routing rides the small, fast, already-warm model
(``fast_model``), and only falls back to the chat model when no fast model
is set on a hand-built cfg — config load always resolves ``fast_model``.
"""

import pytest

from jarvis.llm import resolve_model, Tier


class _Cfg:
    """Minimal cfg stand-in with only the attributes the resolver reads."""

    def __init__(self, fast="", chat=""):
        self.fast_model = fast
        self.llm_chat_model = chat


class TestFastTierResolution:

    @pytest.mark.unit
    def test_routing_rides_the_fast_model(self):
        """The whole point of the tier: routing picks the fast model, not the
        chat model. This keeps the routing call on the small, warm model
        instead of reloading the large chat model every turn."""
        cfg = _Cfg(fast="judge-m", chat="chat-m")
        assert resolve_model(cfg, Tier.FAST) == "judge-m"

    @pytest.mark.unit
    def test_falls_through_to_chat_when_fast_unset(self):
        cfg = _Cfg(fast="", chat="chat-m")
        assert resolve_model(cfg, Tier.FAST) == "chat-m"

    @pytest.mark.unit
    def test_returns_empty_when_nothing_configured(self):
        """The caller handles an empty model name by falling back to the
        all-tools path — the helper itself should not invent a default."""
        cfg = _Cfg(fast="", chat="")
        assert resolve_model(cfg, Tier.FAST) == ""

    @pytest.mark.unit
    def test_robust_to_missing_attributes(self):
        """When a cfg-like object is missing an attribute entirely (as can
        happen for partial mocks), the resolver must not raise."""
        class Partial:
            llm_chat_model = "only-chat"
        assert resolve_model(Partial(), Tier.FAST) == "only-chat"

    @pytest.mark.unit
    def test_warmup_and_engine_pick_the_same_model(self):
        """Warmup (listener) and routing (engine) must target the same model.
        Both call resolve_model(cfg, Tier.FAST); this pins that there is no
        second resolution path to drift."""
        cfg = _Cfg(fast="small-m", chat="big-m")
        warmup_choice = resolve_model(cfg, Tier.FAST)
        engine_choice = resolve_model(cfg, Tier.FAST)
        assert warmup_choice == engine_choice == "small-m"
