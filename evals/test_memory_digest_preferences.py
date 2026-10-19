"""Memory-digest evals surface same-domain engagement for recommendations."""

import pytest

from evals.tool_routing import requires_judge_llm
from evals.memory_digest import digest_for_eval


@pytest.mark.eval
@requires_judge_llm
class TestMemoryDigestSurfacesPreferenceSignals:
    """Live tests that the digest surfaces engagement-as-preference signals."""

    def _digest(self, query: str, diary_entries: list[str]) -> str:
        return digest_for_eval(query, diary_entries)

    def test_watch_recommendation_surfaces_recently_discussed_films(self):
        """Reproduces the 2026-04-20 incident directly at the digest layer."""
        diary = [
            "[2026-04-20] The user asked about the movie Titanic; the assistant "
            "summarised its plot and noted it is a 1997 film directed by James Cameron.",
            "[2026-04-19] The conversation focused on the film Possessor; the "
            "assistant said it is a 2020 sci-fi horror by Brandon Cronenberg.",
            "[2026-04-15] The user discussed their weekend plans and mentioned "
            "they had been busy with work projects.",
            "[2026-04-10] The user asked about the weather in London.",
        ]
        digest = self._digest("what should I watch tonight?", diary)
        print(f"\n  Digest: {digest!r}")

        # Digest must not be empty — past film engagement is a preference signal.
        assert digest and digest.strip(), "🧠 Relevant memory must produce a nonempty digest"

        lowered = digest.lower()
        # At least one of the recently-engaged titles must surface.
        surfaced = [t for t in ("titanic", "possessor") if t in lowered]
        assert surfaced, (
            f"Digest did not surface any recently-engaged film as a preference "
            f"signal. Got: {digest!r}"
        )

    def test_restaurant_recommendation_surfaces_past_cuisine_interest(self):
        """Same principle, different domain — past food engagement surfaces
        for a restaurant recommendation query."""
        diary = [
            "[2026-04-18] The user asked about ramen shops near their office "
            "and the assistant listed three in Shoreditch.",
            "[2026-04-12] The user discussed cooking a Thai green curry and "
            "asked how to balance the fish sauce.",
            "[2026-04-05] The user mentioned they had a dentist appointment.",
        ]
        digest = self._digest("suggest a restaurant for dinner tonight", diary)
        print(f"\n  Digest: {digest!r}")

        assert digest and digest.strip(), "🧠 Relevant memory must produce a nonempty digest"

        lowered = digest.lower()
        # At least one of the engaged cuisines/items must surface.
        surfaced = [t for t in ("ramen", "thai", "curry") if t in lowered]
        assert surfaced, (
            f"Digest did not surface any recently-engaged cuisine as a "
            f"preference signal. Got: {digest!r}"
        )

    def test_unrelated_domain_still_returns_none(self):
        """Regression guard: the relaxation must not make the digest surface
        everything. Snippets from a wholly different domain should still NONE
        out for a recommendation query."""
        diary = [
            "[2026-04-18] The user asked about the population of Iceland; the "
            "assistant said it is roughly 380,000.",
            "[2026-04-12] The user asked for help debugging a Python import "
            "cycle in their work project.",
        ]
        digest = self._digest("what should I watch tonight?", diary)
        print(f"\n  Digest: {digest!r}")

        # Neither snippet is in the films/entertainment domain. The digest
        # should either return empty or at least not falsely invent a film
        # preference from population statistics or Python debugging.
        if digest:
            lowered = digest.lower()
            fabricated = any(
                t in lowered for t in ("film", "movie", "watch", "series", "show")
            )
            assert not fabricated, (
                f"Digest fabricated a film preference from unrelated snippets. "
                f"Got: {digest!r}"
            )
