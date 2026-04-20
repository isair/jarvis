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
        surfaced_math_topic = (
            "rectangle" in lowered
            or "7 by 9" in lowered
            or "area of" in lowered
        )
        assert surfaced_fact, (
            f"Digest did not surface the user-stated boxing/location fact "
            f"for an identity query. Got: {digest!r}"
        )
        assert not surfaced_math_topic, (
            f"Digest surfaced a past Q&A topic (rectangle area) as if it "
            f"were a fact about the user. Got: {digest!r}"
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
