"""
Regression eval: DuckDuckGo bot-challenge rescued by the fallback chain.

Prior to the fallback chain, a DDG rate-limit produced either a phantom
"Found 1 result" line over an empty payload or a confabulation from the
reply LLM's priors. The fix was threefold: structural challenge detection
(HTTP 400 + `anomaly-modal`/`anomaly.js` markers), a Brave → Wikipedia
fallback, and an honest-block envelope when every provider fails.

This file is behavioural, not judge-driven: it exercises the real
`WebSearchTool.run` against a mocked network and asserts the observable
outcome — the rescued content lands in the untrusted-extract fence and no
anti-confabulation / block envelope fires when a rescue succeeded.

Run: .venv/bin/python -m pytest evals/test_web_search_fallback.py -v
"""

from unittest.mock import Mock, patch

import pytest

from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.web_search import WebSearchTool


def _make_ctx(cfg_overrides=None):
    cfg = Mock()
    cfg.web_search_enabled = True
    cfg.voice_debug = False
    cfg.brave_search_api_key = ""
    cfg.wikipedia_fallback_enabled = True
    for k, v in (cfg_overrides or {}).items():
        setattr(cfg, k, v)
    ctx = Mock(spec=ToolContext)
    ctx.user_print = Mock()
    ctx.cfg = cfg
    ctx.language = "en"
    return ctx


@pytest.mark.eval
class TestFallbackChainRescuesBotChallenge:
    """DDG bot-challenge + Wikipedia fallback = honest rescue, not confabulation."""

    @patch("jarvis.tools.builtin.web_search._wikipedia_summary")
    @patch("jarvis.tools.builtin.web_search.requests.get")
    def test_wikipedia_rescues_when_ddg_blocks(self, mock_get, mock_wiki):
        # DDG instant API empty, /lite/ returns the bot-challenge structural markers.
        instant = Mock(status_code=200)
        instant.json.return_value = {}
        instant.raise_for_status = Mock()
        challenge = Mock(status_code=400)
        challenge.content = (
            b'<html><body><div class="anomaly-modal"></div>'
            b'<form action="//duckduckgo.com/anomaly.js"></form></body></html>'
        )
        mock_get.side_effect = [instant, challenge]
        mock_wiki.return_value = (
            "Possessor",
            "https://en.wikipedia.org/wiki/Possessor",
            "Possessor is a 2020 psychological body-horror film.",
        )

        result = WebSearchTool().run({"search_query": "possessor movie"}, _make_ctx())

        assert result.success is True
        # Rescued content must be inside the untrusted fence.
        assert "<<<BEGIN UNTRUSTED WEB EXTRACT>>>" in result.reply_text
        assert "psychological body-horror" in result.reply_text
        # The block envelope must NOT fire — the chain rescued the query.
        lowered = result.reply_text.lower()
        assert "blocked by duckduckgo" not in lowered
        assert "you have failed" not in lowered
        # Provenance line list matches the rescue source.
        assert "Possessor" in result.reply_text
        assert "en.wikipedia.org" in result.reply_text

    @patch("jarvis.tools.builtin.web_search._wikipedia_summary")
    @patch("jarvis.tools.builtin.web_search.requests.get")
    def test_honest_block_when_all_providers_fail(self, mock_get, mock_wiki):
        """No Brave key, Wikipedia miss → honest-block envelope, no confabulation."""
        instant = Mock(status_code=200)
        instant.json.return_value = {}
        instant.raise_for_status = Mock()
        challenge = Mock(status_code=400)
        challenge.content = b'<div class="anomaly-modal"></div>'
        mock_get.side_effect = [instant, challenge]
        mock_wiki.return_value = None

        result = WebSearchTool().run({"search_query": "obscure thing"}, _make_ctx())

        assert result.success is True
        lowered = result.reply_text.lower()
        # Honest-block markers from the rate-limited envelope.
        assert "blocked by duckduckgo" in lowered
        assert "you have failed" in lowered
        assert "two short sentences" in lowered
        # Must not pretend there were results.
        assert "<<<BEGIN UNTRUSTED WEB EXTRACT>>>" not in result.reply_text
