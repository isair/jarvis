"""Tests for tool-router model resolution order.

The reply engine and the listener warmup path both need to pick the model
used for LLM-based tool selection, and they MUST pick the same one — if they
diverge, warmup loads the wrong model and the first real routing call eats a
cold-start stall. The resolution order is enforced by a single helper
(``resolve_tool_router_model``), which this test exercises directly.

Order: `tool_router_model` → `intent_judge_model` → `ollama_chat_model` →
empty string. The key property is that an explicit `tool_router_model` wins
over everything, and that an empty `tool_router_model` falls through to the
(small, fast, already-warm) judge model BEFORE the (large, slow) chat model.
"""

import pytest

from jarvis.reply.engine import resolve_tool_router_model


class _Cfg:
    """Minimal cfg stand-in with only the attributes the resolver reads."""

    def __init__(self, router="", judge="", chat=""):
        self.tool_router_model = router
        self.intent_judge_model = judge
        self.ollama_chat_model = chat


class TestToolRouterModelResolution:

    @pytest.mark.unit
    def test_explicit_router_wins(self):
        cfg = _Cfg(router="custom-router", judge="judge-m", chat="chat-m")
        assert resolve_tool_router_model(cfg) == "custom-router"

    @pytest.mark.unit
    def test_empty_router_falls_through_to_judge(self):
        """The whole point of the helper: an unset tool_router_model must
        pick the judge model, not the chat model. This is what keeps the
        routing call on the small, warm model instead of reloading the
        large chat model every turn."""
        cfg = _Cfg(router="", judge="judge-m", chat="chat-m")
        assert resolve_tool_router_model(cfg) == "judge-m"

    @pytest.mark.unit
    def test_falls_through_to_chat_when_no_router_or_judge(self):
        cfg = _Cfg(router="", judge="", chat="chat-m")
        assert resolve_tool_router_model(cfg) == "chat-m"

    @pytest.mark.unit
    def test_returns_empty_when_nothing_configured(self):
        """The caller handles an empty model name by falling back to the
        all-tools path — the helper itself should not invent a default."""
        cfg = _Cfg(router="", judge="", chat="")
        assert resolve_tool_router_model(cfg) == ""

    @pytest.mark.unit
    def test_robust_to_missing_attributes(self):
        """When a cfg-like object is missing an attribute entirely (as can
        happen for partial mocks), the resolver must not raise."""
        class Partial:
            ollama_chat_model = "only-chat"
        assert resolve_tool_router_model(Partial()) == "only-chat"
