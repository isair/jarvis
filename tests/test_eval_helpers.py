"""Unit tests for shared eval helpers.

These helpers shape what the eval suite actually measures — specifically
the fallback-reply detection that turns the malformed-output guard from
a silent shield into a loud failure. Pinning the helpers at unit level
means a typo or drift in the canned fallback strings in
``src/jarvis/reply/engine.py`` is caught without needing to run a live
LLM eval.
"""

from pathlib import Path
import sys

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_EVALS = _ROOT / "evals"
if str(_EVALS) not in sys.path:
    sys.path.insert(0, str(_EVALS))

from helpers import (  # noqa: E402
    FALLBACK_REPLY_PHRASES,
    assert_not_fallback_reply,
    is_fallback_reply,
)


class TestIsFallbackReply:
    """The helper must recognise every canned fallback string the reply
    engine might emit on malformed model output."""

    def test_empty_and_none_are_not_fallback(self):
        assert is_fallback_reply(None) is False
        assert is_fallback_reply("") is False

    @pytest.mark.parametrize(
        "reply",
        [
            "I had trouble understanding that request. Could you try rephrasing it?",
            "I had trouble understanding that request.",
            "Sorry, I had trouble processing that. Could you try again?",
            "sorry, i had trouble performing the web search.",
            # Case-insensitive match.
            "I HAD TROUBLE UNDERSTANDING THAT REQUEST.",
        ],
    )
    def test_canned_fallbacks_are_flagged(self, reply):
        assert is_fallback_reply(reply), (
            f"Helper should flag {reply!r} as the engine's canned "
            "malformed-guard fallback."
        )

    @pytest.mark.parametrize(
        "reply",
        [
            "The weather in Hackney is 14°C and partly cloudy.",
            "I found three results: Annie Lennox, Lulu, and Shirley Manson.",
            "Sure — I opened YouTube for you.",
            "I don't have that information, but I can search for it.",
        ],
    )
    def test_real_replies_are_not_flagged(self, reply):
        assert not is_fallback_reply(reply), (
            f"Helper must NOT flag genuine replies: {reply!r}"
        )


class TestFallbackPhrasesAgainstEngineSource:
    """Pin the helper's phrase list against the actual canned strings in
    the reply engine. If someone changes a fallback string in
    ``engine.py`` without updating the helper, this test fails and the
    eval suite doesn't silently revert to "fallback looks like success".
    """

    def test_every_phrase_appears_in_engine_source(self):
        engine_src = (_ROOT / "src" / "jarvis" / "reply" / "engine.py").read_text()
        engine_src_lower = engine_src.lower()
        for phrase in FALLBACK_REPLY_PHRASES:
            assert phrase in engine_src_lower, (
                f"Fallback phrase {phrase!r} no longer appears in "
                f"engine.py. Either the engine's canned reply changed "
                f"(update FALLBACK_REPLY_PHRASES in evals/helpers.py) "
                f"or the phrase list has drifted."
            )


class TestAssertNotFallbackReply:
    def test_passes_on_real_reply(self):
        # Should not raise.
        assert_not_fallback_reply("Today is sunny in Hackney.", context="weather")

    def test_fails_on_canned_fallback(self):
        # pytest.fail raises _pytest.outcomes.Failed, which inherits from
        # BaseException (not Exception), so catch the broader type.
        with pytest.raises(BaseException) as exc_info:
            assert_not_fallback_reply(
                "I had trouble understanding that request. Could you try rephrasing it?",
                context="weather-warm-memory",
            )
        # Context tag should show up in the message so failing evals point
        # at the specific parametrised variant.
        assert "weather-warm-memory" in str(exc_info.value)

    def test_passes_on_empty(self):
        # Empty response is a separate failure mode (no text at all),
        # not the malformed-guard fallback — don't conflate them.
        assert_not_fallback_reply("", context="x")
        assert_not_fallback_reply(None, context="x")
