"""
Unit tests for the deterministic diary-deflection scrub.

The summariser prompt (rule 6, added in #232) tells the LLM not to narrate
assistant failures. On the smallest supported model (gemma4:e2b) the prompt
rule reduces but does not eliminate the leak — field measurement on the
author's diary showed ~40% of post-rule writes still containing banned
phrasing. The scrub is a deterministic safety net that runs after the LLM
returns and before the row lands in ``conversation_summaries``, plus a bulk
sweep that walks existing rows so historical poisoning can be cleaned in
one pass.

Mirrors the two-layer defence the knowledge graph already has (#291
extractor BANNED FACT FORMS, #298/#305 deterministic merge-time rewrite).
The summariser's prompt is the first layer; this scrub is the second.
"""

from __future__ import annotations

import pytest

from jarvis.memory.conversation import (
    scrub_deflection_sentences,
    DEFLECTION_PATTERNS,
)


class TestScrubDropsBannedSentences:
    """Each banned-phrase variant must drop the sentence that contains it."""

    @pytest.mark.parametrize(
        "summary,must_disappear",
        [
            (
                "The user asked about Possessor. The assistant did not have specific information about it.",
                "did not have specific information",
            ),
            (
                "The user asked about Piranesi. The assistant was unable to provide details.",
                "was unable",
            ),
            (
                "The user asked to open YouTube. The assistant explained it could not open applications.",
                "could not open",
            ),
            (
                "The user asked about tube strikes. The assistant offered to search the web.",
                "offered to search",
            ),
            (
                "The user asked about a recipe. The assistant suggested checking an online cookbook.",
                "suggested checking",
            ),
            (
                "The user asked for legal advice. The assistant recommended consulting a professional.",
                "recommended consulting",
            ),
            (
                "The user asked about an obscure topic. The assistant clarified that it could not answer.",
                "clarified that it could not",
            ),
            (
                "The user wanted help with code. The assistant lacks access to the repository.",
                "lacks access",
            ),
            (
                "The user asked about cooking times. The assistant cannot access live kitchen sensors.",
                "cannot access",
            ),
            (
                "The user asked about an old photo. The assistant does not have information about it.",
                "does not have information",
            ),
            (
                "The user asked about an obscure topic. The assistant stated it could not find anything.",
                "stated it could not",
            ),
            (
                "The user mentioned X. The assistant indicated it did not have details on the matter.",
                "indicated it did not have",
            ),
            (
                "The user asked about a film. The assistant explained that it does not access streaming catalogues.",
                "explained that it does not",
            ),
            # ── Reporting verb "said" + denial (the user-reported leak) ─────
            (
                "The user asked about a tube delay. The assistant said it couldn't check live travel data.",
                "couldn't check",
            ),
            (
                "The user asked to open a website. The assistant said it could not open browsers.",
                "could not open",
            ),
            (
                "The user asked about a recipe. The assistant said it didn't know that one.",
                "didn't know",
            ),
            (
                "The user asked about a venue. The assistant said it does not have access to that information.",
                "does not have access",
            ),
            (
                "The user asked about a fact. The assistant said it has no information on that topic.",
                "has no information",
            ),
            # ── Direct "failed to" shape ────────────────────────────────────
            (
                "The user asked for a search. The assistant failed to find any matching results.",
                "failed to find",
            ),
            # ── Direct contraction shapes ──────────────────────────────────
            (
                "The user asked about a place. The assistant didn't have any details about it.",
                "didn't have",
            ),
            (
                "The user asked a question. The assistant doesn't have access to that.",
                "doesn't have",
            ),
            (
                "The user asked about an obscure topic. The assistant couldn't answer that.",
                "couldn't answer",
            ),
            # ── Other reporting verbs + "it" + denial ──────────────────────
            (
                "The user asked about a setting. The assistant noted that it could not change system preferences.",
                "could not change",
            ),
            (
                "The user asked about an event. The assistant acknowledged it had no information about it.",
                "had no information",
            ),
            (
                "The user asked about a capability. The assistant admitted it was not able to perform that.",
                "was not able",
            ),
        ],
    )
    def test_banned_sentence_is_dropped_in_full(self, summary, must_disappear):
        """Drop the entire offending sentence, not just the phrase — leaving
        half-sentences corrupts the record worse than the original leak.
        """
        cleaned, removed = scrub_deflection_sentences(summary)
        assert must_disappear.lower() not in cleaned.lower(), (
            f"Banned phrase '{must_disappear}' survived scrub: {cleaned!r}"
        )
        assert removed >= 1, "scrub must report at least one removed sentence"


class TestScrubPreservesLegitimateContent:
    """The scrub must not destroy real content. False positives poison the
    diary in a different way — by erasing facts the user actually shared.
    """

    def test_user_preferences_survive(self):
        summary = (
            "The user prefers Celsius for temperatures. "
            "The user lives in Hackney."
        )
        cleaned, removed = scrub_deflection_sentences(summary)
        assert "celsius" in cleaned.lower()
        assert "hackney" in cleaned.lower()
        assert removed == 0

    def test_tool_grounded_facts_survive(self):
        summary = (
            "The weather in London was 12°C and partly cloudy. "
            "The user asked about the weekend forecast."
        )
        cleaned, removed = scrub_deflection_sentences(summary)
        assert "12" in cleaned and "london" in cleaned.lower()
        assert removed == 0

    def test_attributed_assistant_answer_survives(self):
        """An attributed assistant answer ("the assistant said X") is the
        opposite of a deflection — the prompt explicitly wants these
        preserved (rule 7 attribution). The scrub must not strip them.
        """
        summary = (
            "The user asked about Possessor. "
            "The assistant said it is a 2020 science-fiction horror film by Brandon Cronenberg."
        )
        cleaned, removed = scrub_deflection_sentences(summary)
        assert "cronenberg" in cleaned.lower()
        assert "2020" in cleaned
        assert removed == 0

    def test_user_corrections_survive(self):
        summary = (
            "The assistant said Possessor is a 2006 film. "
            "The user corrected that it is from 2020."
        )
        cleaned, _ = scrub_deflection_sentences(summary)
        assert "corrected" in cleaned.lower()
        assert "2020" in cleaned

    def test_legitimate_user_imperatives_survive(self):
        """Genuine user-issued imperatives are not deflections and must
        survive — same distinction the graph merge prompt makes between
        meta-narrative and directives.
        """
        summary = (
            "The user told the assistant to always reply in British English. "
            "The user said they prefer concise answers."
        )
        cleaned, removed = scrub_deflection_sentences(summary)
        assert "british english" in cleaned.lower()
        assert "concise" in cleaned.lower()
        assert removed == 0


class TestScrubEdgeCases:
    def test_empty_input_returns_empty(self):
        cleaned, removed = scrub_deflection_sentences("")
        assert cleaned == ""
        assert removed == 0

    def test_none_input_returns_empty(self):
        cleaned, removed = scrub_deflection_sentences(None)
        assert cleaned == ""
        assert removed == 0

    def test_summary_made_entirely_of_deflection_is_kept_intact(self):
        """If scrubbing would empty the summary, keep the original. An
        empty diary entry is worse than a slightly-leaky one — downstream
        retrieval treats absence as "no record" and the user loses the
        topic of the conversation entirely.
        """
        summary = "The assistant did not have information. The assistant was unable to help."
        cleaned, removed = scrub_deflection_sentences(summary)
        assert cleaned == summary, (
            "When scrubbing would empty the summary, the original must be returned unchanged"
        )
        # removed count should still report what *would* have been dropped, so the
        # caller can log "row would have been emptied — kept original".
        assert removed >= 1

    def test_idempotent(self):
        """Running scrub twice on the same input produces the same output —
        critical for the bulk sweep, which must be safe to re-run."""
        summary = (
            "The user asked about Piranesi. "
            "The assistant did not have specific information. "
            "The user lives in Hackney."
        )
        once, _ = scrub_deflection_sentences(summary)
        twice, removed_second = scrub_deflection_sentences(once)
        assert once == twice
        assert removed_second == 0

    def test_multiple_deflections_one_pass(self):
        summary = (
            "The user asked about A. "
            "The assistant did not have information. "
            "The user asked about B. "
            "The assistant offered to search the web. "
            "The user lives in Hackney."
        )
        cleaned, removed = scrub_deflection_sentences(summary)
        assert removed == 2
        assert "hackney" in cleaned.lower()
        assert "asked about a" in cleaned.lower()
        assert "asked about b" in cleaned.lower()
        assert "did not have" not in cleaned.lower()
        assert "offered to search" not in cleaned.lower()

    def test_multiline_summary_with_embedded_newlines(self):
        """Sentence splitter must handle ``\\n`` whitespace between
        sentences. Real summaries can land with newlines instead of
        spaces depending on how the model formats its output.
        """
        summary = (
            "The user asked about X.\n"
            "The assistant did not have information.\n"
            "The user lives in Hackney."
        )
        cleaned, removed = scrub_deflection_sentences(summary)
        assert removed == 1
        assert "hackney" in cleaned.lower()
        assert "did not have" not in cleaned.lower()

    def test_summary_without_terminal_punctuation_is_treated_as_one_sentence(self):
        """If the LLM forgets the final period, the entire summary becomes
        one sentence. The scrub must still behave coherently — drop the
        whole thing if it matches a banned pattern, otherwise leave it.
        """
        summary = "The user asked about X and the assistant did not have information"
        cleaned, removed = scrub_deflection_sentences(summary)
        # One sentence containing a banned phrase → would empty → keep
        # original (per the empty-row guard).
        assert removed == 1
        assert cleaned == summary

    def test_field_observed_post_rule_shape(self):
        """The exact shape that gemma4:e2b produced on 2026-04-27 in the
        author's diary (paraphrased — no real content in tests).

        Two sentences: a legitimate topic record + a banned deflection
        narration. The scrub must drop only the second.
        """
        summary = (
            "The user enquired about an obscure restaurant in Hackney. "
            "The assistant did not have specific information about that establishment."
        )
        cleaned, removed = scrub_deflection_sentences(summary)
        assert removed == 1
        assert "hackney" in cleaned.lower()
        assert "obscure restaurant" in cleaned.lower()
        assert "did not have" not in cleaned.lower()


class TestWritePathIntegration:
    """``update_daily_conversation_summary`` must run the scrub before the
    row lands in ``conversation_summaries``. This is the second layer
    behind the prompt rule — the prompt aims to prevent the leak, the
    scrub catches what slips through.
    """

    def test_scrub_fires_before_db_write(self, tmp_path, monkeypatch):
        from jarvis.memory.db import Database
        from jarvis.memory import conversation as cmod

        db = Database(tmp_path / "jarvis.db")

        leaky_llm_output = (
            "SUMMARY: The user asked about a restaurant in Hackney. "
            "The assistant did not have specific information about it.\n"
            "TOPICS: hackney, restaurant"
        )

        def fake_call(base_url, model, system_prompt, user_prompt, **kwargs):
            return leaky_llm_output

        monkeypatch.setattr("jarvis.llm.call_llm_direct", fake_call)
        monkeypatch.setattr(cmod, "call_llm_direct", fake_call)
        monkeypatch.setattr(cmod, "get_embedding", lambda *a, **k: None)

        cmod.update_daily_conversation_summary(
            db=db,
            new_chunks=["User: any good spot in Hackney?", "Assistant: ..."],
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="gemma4:e2b",
            ollama_embed_model="nomic-embed-text",
        )

        rows = db.get_all_conversation_summaries()
        assert len(rows) == 1
        persisted = rows[0]["summary"].lower()
        assert "hackney" in persisted, "legitimate content must survive scrub"
        assert "did not have" not in persisted, (
            "scrub must run before persistence — banned phrase reached the DB"
        )


class TestPatternsExposed:
    """The pattern set is exposed as a module-level constant so the bulk
    sweep, the unit tests, and any future enrichment-side filter can share
    one source of truth.
    """

    def test_patterns_are_compiled(self):
        import re
        assert DEFLECTION_PATTERNS, "must expose at least one pattern"
        for pat in DEFLECTION_PATTERNS:
            assert isinstance(pat, re.Pattern)

    def test_patterns_are_case_insensitive(self):
        cleaned, removed = scrub_deflection_sentences(
            "The User Asked About X. THE ASSISTANT DID NOT HAVE INFORMATION."
        )
        assert removed == 1
        assert "did not have" not in cleaned.lower()
