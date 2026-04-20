"""
Memory Digest — Identity-Query Fact Surfacing (Live)

Guards that the memory digest distiller (``enrichment.digest_memory_for_query``)
surfaces user-stated facts about the user (location, interests, ongoing
plans, biography) when the current query asks who the user is or what the
assistant knows about them, rather than surfacing past Q&A topics the user
merely asked about.

Motivating field incident:
  The user asked "what do you know about me?". The diary contained a
  user-stated fact ("goes boxing near E3 2WS") alongside a past Q&A where
  the user asked for the area of a rectangle. The digest surfaced the
  rectangle question, which is not a fact about the user at all — leading
  the reply model to miss the actual identity signal entirely.

General principle (encoded in the digest prompt): for identity queries,
user-stated facts dominate over past Q&A topics, and multiple such facts
should be surfaced when present.

Run: EVAL_JUDGE_MODEL=gemma4:e2b pytest evals/test_memory_digest_identity.py -v
"""

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_BASE_URL, JUDGE_MODEL


@pytest.mark.eval
@requires_judge_llm
class TestMemoryDigestSurfacesIdentityFacts:
    """Live tests that the digest prefers user-stated facts for identity queries."""

    def _digest(self, query: str, diary_entries: list[str]) -> str:
        from jarvis.reply.enrichment import digest_memory_for_query
        return digest_memory_for_query(
            query=query,
            diary_entries=diary_entries,
            graph_parts=[],
            ollama_base_url=JUDGE_BASE_URL,
            ollama_chat_model=JUDGE_MODEL,
            timeout_sec=60.0,
        )

    def test_identity_query_surfaces_user_stated_fact_over_past_qa(self):
        """Reproduces the field incident directly at the digest layer.

        Padding filler ensures the raw block exceeds ``_DIGEST_MIN_CHARS``
        (400) so the distil LLM actually runs — below that threshold the
        raw text is passed through unchanged and this test would be a
        no-op.
        """
        diary = [
            "[2026-04-10] The user said they go boxing near E3 2WS.",
            "[2026-04-12] The user asked for the area of a rectangle 7 by 9; "
            "the assistant said 63.",
            "[2026-04-11] The user asked what the capital of Peru is; the "
            "assistant said Lima. They also asked about the population and "
            "the assistant said it is roughly 10 million in the metro area.",
            "[2026-04-09] The user asked the assistant to convert 200 USD to "
            "GBP; the assistant said approximately 158 GBP at the current rate.",
            "[2026-04-08] The user asked the assistant for the boiling point "
            "of water at sea level; the assistant said 100 degrees Celsius.",
        ]
        digest = self._digest("what do you know about me?", diary)
        print(f"\n  Digest: {digest!r}")

        if not digest:
            pytest.xfail(
                f"Small judge model {JUDGE_MODEL} returned NONE for an "
                f"identity query despite user-stated facts being present."
            )

        lowered = digest.lower()
        surfaced_fact = "boxing" in lowered or "e3" in lowered
        # Past Q&A topics that must stay out of an identity digest. The
        # field-incident topic (rectangle area) is the primary guard;
        # currency and boiling-point are included because they are
        # numeric/factoid Q&As with no user-preference character — the
        # exact failure class the identity rule targets.
        surfaced_past_qa = any(
            kw in lowered
            for kw in (
                "rectangle",
                "7 by 9",
                "area of",
                "usd",
                "gbp",
                "boiling",
            )
        )
        assert surfaced_fact, (
            f"Digest did not surface the user-stated boxing/location fact "
            f"for an identity query. Got: {digest!r}"
        )
        assert not surfaced_past_qa, (
            f"Digest surfaced past Q&A topics as if they were facts "
            f"about the user. Got: {digest!r}"
        )

    def test_identity_query_surfaces_multiple_user_facts_when_present(self):
        """When several user-stated facts exist, the digest should combine
        them rather than pick just one."""
        diary = [
            "[2026-04-10] The user said they live in East London.",
            "[2026-04-11] The user said they are vegetarian.",
            "[2026-04-12] The user said they are learning Japanese.",
            "[2026-04-13] The user asked about the capital of Peru; the "
            "assistant said Lima.",
            "[2026-04-09] The user asked the assistant to convert 200 USD to "
            "GBP; the assistant said approximately 158 GBP at the current rate.",
            "[2026-04-08] The user asked the boiling point of water at sea "
            "level; the assistant said 100 degrees Celsius.",
        ]
        digest = self._digest("tell me about myself", diary)
        print(f"\n  Digest: {digest!r}")

        if not digest:
            pytest.xfail(
                f"Small judge model {JUDGE_MODEL} returned NONE for an "
                f"identity query despite multiple user-stated facts."
            )

        lowered = digest.lower()
        facts_hit = sum(
            kw in lowered
            for kw in ("east london", "vegetarian", "japanese")
        )
        assert facts_hit >= 2, (
            f"Digest surfaced fewer than 2 of the 3 user-stated facts for "
            f"an identity query. Got: {digest!r}"
        )
        past_qa_leak = any(
            kw in lowered for kw in ("usd", "gbp", "boiling")
        )
        assert not past_qa_leak, (
            f"Digest leaked a past Q&A topic into an identity-query "
            f"digest. Got: {digest!r}"
        )

    def test_identity_query_with_only_past_qa_returns_none_or_no_false_facts(self):
        """Regression guard: if NO user-stated facts exist, the digest must
        not fabricate a user fact from past Q&A topics."""
        diary = [
            "[2026-04-12] The user asked for the area of a rectangle 7 by 9; "
            "the assistant said 63.",
            "[2026-04-13] The user asked about the capital of Peru; the "
            "assistant said Lima.",
            "[2026-04-11] The user asked the assistant to convert 200 USD to "
            "GBP; the assistant said approximately 158 GBP at the current rate.",
            "[2026-04-10] The user asked the boiling point of water at sea "
            "level; the assistant said 100 degrees Celsius.",
            "[2026-04-09] The user asked for the capital of Australia; the "
            "assistant said Canberra.",
        ]
        digest = self._digest("what do you know about me?", diary)
        print(f"\n  Digest: {digest!r}")

        lowered = digest.lower()
        fabricated_user_fact = any(
            phrase in lowered
            for phrase in (
                "user likes math",
                "user is interested in math",
                "user likes geography",
                "user is interested in peru",
            )
        )
        assert not fabricated_user_fact, (
            f"Digest fabricated a user-preference claim from past Q&A "
            f"topics. Got: {digest!r}"
        )

    def test_identity_query_does_not_trigger_recommendation_engagement_rule(self):
        """Cross-rule guard: the recommendation-engagement rule says past
        interactions count as preference signals for 'what should I watch'.
        An IDENTITY query with the same film-engagement diary must not
        mistakenly treat the films as facts about the user — the identity
        rule still applies and past Q&A topics stay out unless the snippet
        explicitly says the user is into that topic."""
        diary = [
            "[2026-04-20] The user asked about the movie Titanic; the "
            "assistant summarised its plot and noted it is a 1997 film "
            "directed by James Cameron.",
            "[2026-04-19] The conversation focused on the film Possessor; "
            "the assistant said it is a 2020 sci-fi horror by Brandon "
            "Cronenberg.",
            "[2026-04-10] The user said they live in East London and work "
            "as a software engineer.",
        ]
        digest = self._digest("what do you know about me?", diary)
        print(f"\n  Digest: {digest!r}")

        if not digest:
            pytest.xfail(
                f"Small judge model {JUDGE_MODEL} returned NONE for an "
                f"identity query despite user-stated facts present."
            )

        lowered = digest.lower()
        user_fact_surfaced = any(
            kw in lowered
            for kw in ("east london", "software engineer", "engineer")
        )
        assert user_fact_surfaced, (
            f"Digest did not surface the user-stated location/occupation "
            f"fact for an identity query. Got: {digest!r}"
        )
        # The film Q&As must NOT be presented as user facts. The identity
        # rule's "not a fact unless the snippet says the user is into it"
        # clause must override the recommendation-engagement rule here.
        film_presented_as_user_fact = any(
            phrase in lowered
            for phrase in (
                "the user likes",
                "the user enjoys",
                "the user is a fan",
                "the user is into",
                "taste signal",
                "already covered",
            )
        )
        assert not film_presented_as_user_fact, (
            f"Digest applied the recommendation-engagement rule to an "
            f"identity query: films framed as user taste/preference. "
            f"Got: {digest!r}"
        )

    def test_recommendation_query_still_surfaces_engagement_when_user_facts_present(self):
        """Reverse cross-rule guard: a recommendation query alongside
        user-stated facts must still surface engagement-as-preference.
        The identity rule's 'prefer user-stated facts' must not suppress
        the recommendation rule's engagement signals."""
        diary = [
            "[2026-04-20] The user asked about the movie Titanic; the "
            "assistant summarised its plot and noted it is a 1997 film "
            "directed by James Cameron.",
            "[2026-04-19] The conversation focused on the film Possessor; "
            "the assistant said it is a 2020 sci-fi horror by Brandon "
            "Cronenberg.",
            "[2026-04-10] The user said they live in East London.",
        ]
        digest = self._digest("what should I watch tonight?", diary)
        print(f"\n  Digest: {digest!r}")

        if not digest:
            pytest.xfail(
                f"Small judge model {JUDGE_MODEL} returned NONE for a "
                f"recommendation query despite engagement signals present."
            )

        lowered = digest.lower()
        engagement_surfaced = any(
            kw in lowered for kw in ("titanic", "possessor")
        )
        assert engagement_surfaced, (
            f"Digest suppressed engagement-as-preference signals on a "
            f"recommendation query, likely because the identity rule "
            f"dominated. Got: {digest!r}"
        )
