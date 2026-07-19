"""Regression tests for the ``llm_embedding_timeout_sec`` config wiring.

Production previously read the attribute as ``llm_embed_timeout_sec``
(missing two letters). Because every read site used
``getattr(cfg, "llm_embed_timeout_sec", 10.0)`` with a default, the
typo never raised; instead the user's configured value was silently
ignored and the 10.0 fallback was used everywhere.

These tests pin two behaviours that together prevent the typo from
re-creeping in:

1. Setting ``cfg.llm_embedding_timeout_sec`` actually flows through to
   the embedding call. The toolSearchTool path is the cleanest place
   to verify this end-to-end because ``select_tools`` is one
   monkey-patch away and the rest of its inputs are simple to set up.
2. The misspelt attribute name is absent from every production module
   that consumes the timeout. A ``getattr`` with a default is
   indistinguishable from "field not yet read" in tests, so we assert
   directly against the source text.
"""

from __future__ import annotations

import inspect

import pytest


def test_tool_search_passes_configured_embedding_timeout_to_select_tools(
    mock_config, monkeypatch
):
    """Setting ``llm_embedding_timeout_sec`` on the config must reach
    ``select_tools`` as ``embed_timeout_sec``. Before the rename, this
    test would have asserted 10.0 (the masked fallback) rather than the
    sentinel value 42.0."""
    captured: dict = {}

    def fake_select_tools(**kwargs):
        captured.update(kwargs)
        return ["stop"]

    monkeypatch.setattr(
        "jarvis.tools.builtin.tool_search.select_tools", fake_select_tools
    )

    mock_config.llm_embedding_timeout_sec = 42.0

    from jarvis.tools.base import ToolContext
    from jarvis.tools.builtin.tool_search import ToolSearchTool

    tool = ToolSearchTool()
    ctx = ToolContext(
        db=None,
        cfg=mock_config,
        system_prompt="",
        original_prompt="anything",
        redacted_text="anything",
        max_retries=0,
        user_print=lambda *_: None,
    )
    result = tool.run({"query": "weather"}, ctx)

    assert result.success is True or result.success is False  # call completed
    assert captured.get("embed_timeout_sec") == 42.0


@pytest.mark.parametrize(
    "module_path",
    [
        "jarvis.reply.engine",
        "jarvis.tools.builtin.tool_search",
    ],
)
def test_no_module_references_misspelt_attribute(module_path):
    """Source-level guard: the typo'd attribute name must not reappear
    in any of the modules that read the embedding timeout. A future
    edit that re-introduces ``llm_embed_timeout_sec`` would silently
    fall back to the 10.0 default again — fail loudly here instead."""
    import importlib

    module = importlib.import_module(module_path)
    src = inspect.getsource(module)

    assert "llm_embed_timeout_sec" not in src, (
        f"{module_path} still references the misspelt "
        f"`llm_embed_timeout_sec`; rename to `llm_embedding_timeout_sec`."
    )
    assert "llm_embedding_timeout_sec" in src, (
        f"{module_path} no longer reads `llm_embedding_timeout_sec`; "
        f"the embedding-timeout config has lost its consumer."
    )
