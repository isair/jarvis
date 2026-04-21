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
    MAX_TURNS_DIGEST_PHRASES,
    assert_not_fallback_reply,
    assert_not_max_turns_digest,
    is_fallback_reply,
    is_max_turns_digest,
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


class TestIsMaxTurnsDigest:
    """The helper must recognise the canonical caveat shapes the
    ``digest_loop_for_max_turns`` summariser produces."""

    def test_empty_and_none_are_not_digest(self):
        assert is_max_turns_digest(None) is False
        assert is_max_turns_digest("") is False

    @pytest.mark.parametrize(
        "reply",
        [
            "I could not fully finish your request. I found the weather is 8°C.",
            "I couldn't fully finish this. I found the London forecast looks cloudy today.",
            "I was unable to fully finish the request, but I got the forecast.",
            "I wasn't able to fully finish that, but here's what I found.",
            # Case-insensitive match.
            "I COULD NOT FULLY FINISH YOUR REQUEST.",
        ],
    )
    def test_caveats_are_flagged(self, reply):
        assert is_max_turns_digest(reply), (
            f"Helper should flag {reply!r} as the max-turns digest caveat."
        )

    @pytest.mark.parametrize(
        "reply",
        [
            "The weather in Hackney is 14°C and partly cloudy.",
            "I found three results: Annie Lennox, Lulu, and Shirley Manson.",
            "Sure — I opened YouTube for you.",
            # "Finish" appearing in a non-caveat sentence must not trigger.
            "You can finish the task by pressing enter.",
        ],
    )
    def test_real_replies_are_not_flagged(self, reply):
        assert not is_max_turns_digest(reply), (
            f"Helper must NOT flag genuine replies: {reply!r}"
        )


class TestMaxTurnsPhrasesAgainstEnrichmentSource:
    """Drift pin: every phrase in ``MAX_TURNS_DIGEST_PHRASES`` must
    correspond to the caveat instruction in the digest prompt source.
    If the prompt's caveat wording is changed, the phrase list must be
    updated in lockstep or the eval silently stops catching the leak.
    """

    def test_digest_prompt_mentions_fully_finish(self):
        src = (_ROOT / "src" / "jarvis" / "reply" / "enrichment.py").read_text()
        # The digest prompt instructs the LLM to open with a caveat about
        # not being able to fully finish; the anchor phrase here is
        # ``fully finish``, which is the semantic core every canonical
        # phrase in MAX_TURNS_DIGEST_PHRASES shares.
        assert "fully finish" in src.lower(), (
            "Digest prompt in enrichment.py no longer contains the "
            "'fully finish' caveat anchor — either the prompt wording "
            "changed (update MAX_TURNS_DIGEST_PHRASES in evals/helpers.py) "
            "or the anchor drifted."
        )
        # Every phrase we flag must contain the shared anchor; this keeps
        # the helper honest about what it claims to detect.
        for phrase in MAX_TURNS_DIGEST_PHRASES:
            assert "fully finish" in phrase, (
                f"MAX_TURNS_DIGEST_PHRASES entry {phrase!r} does not "
                f"contain the 'fully finish' anchor — the helper would "
                f"flag unrelated replies."
            )


class TestAssertNotMaxTurnsDigest:
    def test_passes_on_real_reply(self):
        assert_not_max_turns_digest(
            "The weather in Paris is 14°C and partly cloudy.",
            context="weather",
        )

    def test_fails_on_digest_caveat(self):
        with pytest.raises(BaseException) as exc_info:
            assert_not_max_turns_digest(
                "I could not fully finish your request. I found the weather is 8°C.",
                context="single-weather-terminal",
            )
        assert "single-weather-terminal" in str(exc_info.value)

    def test_passes_on_empty(self):
        assert_not_max_turns_digest("", context="x")
        assert_not_max_turns_digest(None, context="x")
