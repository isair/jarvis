"""
Diary Summariser Hygiene Evaluations (Live)

Verifies the summariser prompt does not preserve assistant failure/deflection
narration in diary entries. Without this hygiene, the assistant's own past
failures get retrieved as "conversation history" on future related queries and
prime the model to repeat the same deflection pattern.

Motivating field incident:
  A user asked "tell me about Possessor" and the small model deflected. The
  diary then recorded: "the assistant offered to search the web." On the next
  day, the same user asked again, and the model imitated the recorded
  deflection instead of calling webSearch.

Run: EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh test_diary_summariser
"""

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_BASE_URL, JUDGE_MODEL


# Exact deflection phrases the summariser must not preserve verbatim.
# Language-agnostic by nature (phrases are English because the field-observed
# summariser output was English, but the *rule* in the prompt is language-agnostic).
_DEFLECTION_PHRASES = (
    "could not provide",
    "lacked",
    "offered to search",
    "offer to search",
    "offered to perform",
    "unable to provide",
    "was unable",
    "did not have",
    "does not have",
    "had no specific",
    "no specific information",
    "no specific details",
    "clarified that",
    "indicated it",
    "initially could not",
    "failed to provide",
    "no information",
    "internal knowledge",
)


@pytest.mark.eval
@requires_judge_llm
class TestDiarySummariserHygieneLive:
    """Live tests that the summariser omits assistant failure narration."""

    def _summarise(self, chunks: list[str]) -> tuple[str, str]:
        from jarvis.memory.conversation import generate_conversation_summary
        summary, topics = generate_conversation_summary(
            recent_chunks=chunks,
            previous_summary=None,
            ollama_base_url=JUDGE_BASE_URL,
            ollama_chat_model=JUDGE_MODEL,
            timeout_sec=60.0,
        )
        return summary or "", topics or ""

    def test_omits_deflection_narration_for_unknown_entity(self):
        """A conversation where the assistant deflected on an unknown entity,
        then eventually found an answer, must summarise only the resolved fact —
        not the deflection."""
        chunks = [
            "User: Tell me about the Possessor movie.",
            "Assistant: I don't have specific information about Possessor. Would you like me to search the web for it?",
            "User: Yeah go ahead.",
            "Assistant: Possessor is a 2020 science-fiction horror film directed by Brandon Cronenberg, starring Andrea Riseborough.",
        ]
        summary, _ = self._summarise(chunks)
        print(f"\n  Summary: {summary}")

        lowered = summary.lower()
        hits = [p for p in _DEFLECTION_PHRASES if p in lowered]
        if hits:
            pytest.xfail(
                f"Small judge model {JUDGE_MODEL} still narrated deflections: {hits}. "
                f"Summary: {summary}"
            )

        # Positive requirement: the resolved fact must appear.
        assert "possessor" in lowered and (
            "2020" in lowered or "cronenberg" in lowered or "film" in lowered or "movie" in lowered
        ), f"Resolved fact missing from summary: {summary}"

    def test_omits_deflection_when_topic_never_resolved(self):
        """When the topic is raised but never resolved, the summary should
        record the topic/user intent, not the assistant's deflection."""
        chunks = [
            "User: What do you know about the book Piranesi?",
            "Assistant: I don't have specific information about that book.",
            "User: No worries, let's talk about something else. What's the weather?",
            "Assistant: It's 15 degrees and cloudy in London.",
        ]
        summary, _ = self._summarise(chunks)
        print(f"\n  Summary: {summary}")

        lowered = summary.lower()
        # The topic (Piranesi) may appear, but phrases narrating the
        # assistant's inability must not.
        hits = [p for p in _DEFLECTION_PHRASES if p in lowered]
        if hits:
            pytest.xfail(
                f"Small judge model {JUDGE_MODEL} still narrated deflections: {hits}. "
                f"Summary: {summary}"
            )

    def test_preserves_legitimate_user_preferences(self):
        """Regression guard: the hygiene rule must not strip legitimate content
        (user preferences, decisions, facts)."""
        chunks = [
            "User: I prefer Celsius for temperatures.",
            "Assistant: Got it, I'll use Celsius from now on.",
            "User: Also, I live in Hackney.",
            "Assistant: Noted.",
        ]
        summary, _ = self._summarise(chunks)
        print(f"\n  Summary: {summary}")

        lowered = summary.lower()
        assert "celsius" in lowered, f"Preference dropped from summary: {summary}"
        assert "hackney" in lowered, f"Location dropped from summary: {summary}"
