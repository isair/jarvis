"""
Knowledge Extraction Evaluations

Tests the quality of knowledge extraction from conversation summaries.
Ensures the extraction prompt correctly handles:
1. Assistant self-references (should NOT be extracted)
2. Stale temporal snapshots (should NOT be extracted)
3. Common knowledge (should NOT be extracted)
4. Novel knowledge (SHOULD be extracted)
5. Proper reframing (requests → knowledge, not interaction descriptions)

Run:
    EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh knowledge
    EVAL_JUDGE_MODEL=gpt-oss:20b ./scripts/run_evals.sh knowledge
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from conftest import requires_judge_llm
from helpers import (
    MockConfig,
    JUDGE_MODEL,
    JUDGE_BASE_URL,
    call_judge_llm,
    JudgeVerdict,
)

from jarvis.memory.graph_ops import extract_graph_memories


# =============================================================================
# Test Data
# =============================================================================

@dataclass
class ExtractionTestCase:
    """A conversation summary with expected extraction outcomes."""
    summary: str
    date_utc: Optional[str] = None
    # Facts that SHOULD appear (checked by keyword matching)
    should_extract_keywords: List[str] = field(default_factory=list)
    # Patterns that should NOT appear in any extracted fact
    should_not_extract_patterns: List[str] = field(default_factory=list)
    # Minimum number of facts expected
    min_facts: int = 0
    # Maximum number of facts expected (0 = no upper limit)
    max_facts: int = 0


# ── Cases where extraction should produce good novel knowledge ──────────

GOOD_EXTRACTION_CASES = [
    pytest.param(
        ExtractionTestCase(
            summary=(
                "The user asked about boxing gyms in Hackney. I found that "
                "Trenches Boxing Club offers evening classes on weekdays from "
                "6-8pm, priced at 15 pounds per session. The user mentioned "
                "they've been living in Hackney for 2 years."
            ),
            date_utc="2026-04-10",
            should_extract_keywords=["Trenches", "Hackney", "boxing"],
            min_facts=2,
        ),
        id="Novel knowledge: local business details and user location",
    ),
    pytest.param(
        ExtractionTestCase(
            summary=(
                "The user follows an 1800 kcal daily meal plan with a target "
                "of 150g protein. They mentioned preferring air-fried chicken "
                "breast with a soy-oyster-teriyaki glaze — a recipe they've "
                "been perfecting over the past month."
            ),
            date_utc="2026-04-08",
            should_extract_keywords=["1800", "protein"],
            min_facts=2,
        ),
        id="Novel knowledge: user diet plan and preferred recipe",
    ),
    pytest.param(
        ExtractionTestCase(
            summary=(
                "The user is planning to move from London to Tbilisi, Georgia "
                "in June 2026. They've already secured a flat in Vera district "
                "for 800 USD per month. They work remotely as a software "
                "engineer for a UK-based startup called Equals Money."
            ),
            date_utc="2026-04-12",
            should_extract_keywords=["Tbilisi", "Equals Money"],
            min_facts=3,
        ),
        id="Novel knowledge: relocation plans and employment",
    ),
    pytest.param(
        ExtractionTestCase(
            summary=(
                "Kullanıcı Kadıköy'deki Çiya Sofrası restoranını sordu. "
                "Öğle yemeği menüsü 250 TL civarında, özellikle kuzu tandır "
                "ve enginar yemeği çok beğeniliyormuş. Kullanıcı İstanbul'da "
                "Kadıköy semtinde yaşıyor ve haftada 3 kez dışarıda yemek yiyor."
            ),
            date_utc="2026-04-11",
            should_extract_keywords=["Çiya", "Kadıköy"],
            min_facts=2,
        ),
        id="Novel knowledge: non-English summary (Turkish)",
    ),
]


# ── Cases where specific patterns should NOT appear ─────────────────────

BAD_PATTERN_CASES = [
    pytest.param(
        ExtractionTestCase(
            summary=(
                "The user asked about healthy meal options. I recommended "
                "adding more vegetables and lean protein to their diet. I "
                "suggested trying grilled salmon with quinoa and steamed "
                "broccoli. The user thanked me for the suggestions."
            ),
            date_utc="2026-04-10",
            should_not_extract_patterns=[
                r"(?i)assistant",
                r"(?i)recommend",
                r"(?i)suggest",
                r"(?i)I told",
                r"(?i)I advised",
            ],
            max_facts=1,  # Possibly 0 — there's no novel knowledge here
        ),
        id="Reject: assistant self-references (recommendations are not knowledge)",
    ),
    pytest.param(
        ExtractionTestCase(
            summary=(
                "The user asked for the current weather. The temperature in "
                "London is 20 degrees Celsius with partly cloudy skies. Wind "
                "is coming from the southwest at 15 km/h. It's currently "
                "3:45 PM on a Sunday afternoon."
            ),
            date_utc="2026-04-06",
            should_not_extract_patterns=[
                r"(?i)current(ly)? (weather|temperature|time|date)",
                r"(?i)20.*(degree|celsius|°)",
                r"(?i)3:45",
                r"(?i)wind.*southwest",
                r"(?i)partly cloudy",
            ],
            max_facts=1,  # Maybe "user is in London" but nothing else
        ),
        id="Reject: stale temporal snapshots (weather, time of day)",
    ),
]


# ── Cases testing proper reframing ──────────────────────────────────────

REFRAMING_CASES = [
    pytest.param(
        ExtractionTestCase(
            summary=(
                "The user asked about vegetarian restaurants near Covent "
                "Garden. I found Mildreds, which serves plant-based dishes "
                "and has 4.5 stars on Google. The user mentioned they've been "
                "vegetarian for 3 years. They also asked about Dishoom but "
                "decided against it since it's not fully vegetarian."
            ),
            date_utc="2026-04-10",
            should_extract_keywords=["Mildreds", "vegetarian"],
            should_not_extract_patterns=[
                r"(?i)user asked about",
                r"(?i)user enquired",
                r"(?i)user wanted to know",
            ],
            min_facts=2,
        ),
        id="Reframing: requests become knowledge, not interaction descriptions",
    ),
    pytest.param(
        ExtractionTestCase(
            summary=(
                "The user mentioned they started a new job at Equals Money "
                "on March 1st 2026 as a senior backend engineer. They're "
                "working with Python and FastAPI. Their team lead is someone "
                "called Hakan."
            ),
            date_utc="2026-04-05",
            should_extract_keywords=["Equals Money", "March"],
            should_not_extract_patterns=[
                r"(?i)user mentioned",
                r"(?i)user said",
                r"(?i)user told",
            ],
            min_facts=2,
        ),
        id="Reframing: life events framed as facts with temporal context",
    ),
]


# =============================================================================
# Helpers
# =============================================================================

def _run_extraction(case: ExtractionTestCase, config: MockConfig) -> list[str]:
    """Run extract_graph_memories with the given case and config.

    Returns a flat list of fact strings. The extractor now returns
    ``(branch_id, fact)`` tuples; these evals predate branch tagging
    and only care about the fact text. The new branch-routing evals
    live in ``test_graph_branch_routing.py``.
    """
    tagged = extract_graph_memories(
        summary=case.summary,
        ollama_base_url=config.ollama_base_url,
        ollama_chat_model=config.ollama_chat_model,
        timeout_sec=config.llm_chat_timeout_sec,
        thinking=False,
        date_utc=case.date_utc,
    )
    return [fact for _branch, fact in tagged]


def _fact_matches_keyword(facts: list[str], keyword: str) -> bool:
    """Check if any extracted fact contains the keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    return any(keyword_lower in fact.lower() for fact in facts)


def _any_fact_matches_pattern(facts: list[str], pattern: str) -> bool:
    """Check if any extracted fact matches a regex pattern."""
    compiled = re.compile(pattern)
    return any(compiled.search(fact) for fact in facts)


def _judge_extraction_quality(
    summary: str,
    facts: list[str],
    date_utc: Optional[str] = None,
) -> JudgeVerdict:
    """Use LLM-as-judge to evaluate overall extraction quality."""
    system_prompt = (
        "You are evaluating knowledge extraction quality. Given a conversation "
        "summary and the facts extracted from it, score the extraction.\n\n"
        "Score on these criteria (0-10 each):\n"
        "1. NOVELTY: Are the extracted facts genuinely novel (not common "
        "knowledge the model already knows)?\n"
        "2. SELF_CONTAINED: Is each fact a self-contained statement useful "
        "without the original conversation?\n"
        "3. NO_ASSISTANT_VOICE: Are facts written as knowledge, NOT as "
        "descriptions of what the assistant said/recommended?\n"
        "4. NO_STALE_DATA: Are transient details (weather, time of day) "
        "correctly excluded?\n"
        "5. COMPLETENESS: Were important novel facts captured?\n\n"
        "Output your evaluation in this EXACT format:\n"
        "NOVELTY: [0-10]\n"
        "SELF_CONTAINED: [0-10]\n"
        "NO_ASSISTANT_VOICE: [0-10]\n"
        "NO_STALE_DATA: [0-10]\n"
        "COMPLETENESS: [0-10]\n"
        "OVERALL: [PASS/FAIL]\n"
        "REASONING: [One paragraph explaining your verdict]"
    )

    facts_text = "\n".join(f"- {f}" for f in facts) if facts else "(no facts extracted)"
    date_info = f"\nDate context: {date_utc}" if date_utc else ""

    user_prompt = (
        f"Conversation summary:{date_info}\n{summary}\n\n"
        f"Extracted facts:\n{facts_text}"
    )

    response = call_judge_llm(system_prompt, user_prompt, timeout_sec=120.0)

    if not response:
        return JudgeVerdict(
            is_passed=False,
            score=0.0,
            reasoning="Judge LLM unavailable",
        )

    # Parse structured response
    from helpers import _parse_judge_response
    return _parse_judge_response(response)


# =============================================================================
# Test Classes
# =============================================================================

class TestKnowledgeExtractionQuality:
    """Tests that good novel knowledge is correctly extracted."""

    @requires_judge_llm
    @pytest.mark.parametrize("case", GOOD_EXTRACTION_CASES)
    def test_extracts_novel_knowledge(self, mock_config, case: ExtractionTestCase):
        """Verify that novel knowledge is extracted with expected keywords."""
        facts = _run_extraction(case, mock_config)

        # Should extract at least min_facts
        assert len(facts) >= case.min_facts, (
            f"Expected at least {case.min_facts} facts, got {len(facts)}: {facts}"
        )

        # Check that expected keywords appear in at least one fact
        for keyword in case.should_extract_keywords:
            assert _fact_matches_keyword(facts, keyword), (
                f"Expected keyword '{keyword}' in extracted facts: {facts}"
            )

        # Print for report visibility
        print(f"Extracted {len(facts)} facts:")
        for f in facts:
            print(f"  - {f}")


class TestKnowledgeExtractionRejection:
    """Tests that noise, stale data, and common knowledge are rejected."""

    @requires_judge_llm
    @pytest.mark.parametrize("case", BAD_PATTERN_CASES)
    def test_rejects_bad_patterns(self, mock_config, case: ExtractionTestCase):
        """Verify that known bad patterns are not present in extracted facts."""
        facts = _run_extraction(case, mock_config)

        # Check max_facts constraint
        if case.max_facts > 0:
            assert len(facts) <= case.max_facts, (
                f"Expected at most {case.max_facts} facts, got {len(facts)}: {facts}"
            )

        # Check that bad patterns don't appear
        for pattern in case.should_not_extract_patterns:
            assert not _any_fact_matches_pattern(facts, pattern), (
                f"Bad pattern '{pattern}' found in extracted facts: {facts}"
            )

        # Print for report visibility
        print(f"Extracted {len(facts)} facts (expected <= {case.max_facts}):")
        for f in facts:
            print(f"  - {f}")


class TestKnowledgeExtractionReframing:
    """Tests that interaction descriptions are reframed as knowledge."""

    @requires_judge_llm
    @pytest.mark.parametrize("case", REFRAMING_CASES)
    def test_reframes_as_knowledge(self, mock_config, case: ExtractionTestCase):
        """Verify facts are written as knowledge, not interaction descriptions."""
        facts = _run_extraction(case, mock_config)

        # Should extract enough facts
        assert len(facts) >= case.min_facts, (
            f"Expected at least {case.min_facts} facts, got {len(facts)}: {facts}"
        )

        # Should contain expected keywords
        for keyword in case.should_extract_keywords:
            assert _fact_matches_keyword(facts, keyword), (
                f"Expected keyword '{keyword}' in extracted facts: {facts}"
            )

        # Should NOT contain interaction-description patterns
        for pattern in case.should_not_extract_patterns:
            assert not _any_fact_matches_pattern(facts, pattern), (
                f"Interaction-description pattern '{pattern}' found in: {facts}"
            )

        # Print for report visibility
        print(f"Extracted {len(facts)} facts:")
        for f in facts:
            print(f"  - {f}")


class TestKnowledgeExtractionJudge:
    """LLM-as-judge evaluations of overall extraction quality."""

    @requires_judge_llm
    @pytest.mark.parametrize("case", GOOD_EXTRACTION_CASES)
    def test_judge_extraction_quality(self, mock_config, case: ExtractionTestCase):
        """Judge evaluates overall extraction quality on good summaries."""
        facts = _run_extraction(case, mock_config)

        verdict = _judge_extraction_quality(
            summary=case.summary,
            facts=facts,
            date_utc=case.date_utc,
        )

        # Print for report
        print(f"Score: {verdict.score:.2f}")
        print(f"Reasoning: {verdict.reasoning}")
        for criterion, score in verdict.criteria_scores.items():
            print(f"  {criterion}: {score:.1f}")

        # Accept if the judge passes OR the score is above 0.7 —
        # the judge can be overly strict on completeness for minor details
        assert verdict.is_passed or verdict.score >= 0.7, (
            f"Judge failed extraction quality (score={verdict.score:.2f}): "
            f"{verdict.reasoning}\nFacts: {facts}"
        )

    @requires_judge_llm
    def test_judge_empty_conversation_returns_empty(self, mock_config):
        """Empty or trivial conversations should produce no facts."""
        case = ExtractionTestCase(
            summary="The user said hello and I greeted them back. Nothing else was discussed.",
            date_utc="2026-04-12",
        )
        facts = _run_extraction(case, mock_config)

        assert len(facts) == 0, (
            f"Expected 0 facts from trivial conversation, got {len(facts)}: {facts}"
        )

        print("Correctly extracted 0 facts from trivial conversation")

    @requires_judge_llm
    def test_judge_mixed_summary_filters_noise(self, mock_config):
        """A summary with both novel knowledge and noise should only extract the novel parts."""
        case = ExtractionTestCase(
            summary=(
                "The user asked about the weather — it's 22 degrees and sunny "
                "in Hackney right now. I recommended they go for a walk in "
                "Victoria Park. The user mentioned they just adopted a cat "
                "named Miso from Battersea Dogs & Cats Home last week. They "
                "also asked what time it is."
            ),
            date_utc="2026-04-10",
        )
        facts = _run_extraction(case, mock_config)

        # Should capture the cat adoption (novel, specific)
        assert _fact_matches_keyword(facts, "Miso") or _fact_matches_keyword(facts, "cat"), (
            f"Should have extracted cat adoption fact: {facts}"
        )

        # Should NOT capture weather snapshot
        assert not _any_fact_matches_pattern(facts, r"(?i)22.*(degree|celsius|°)"), (
            f"Should not have extracted weather snapshot: {facts}"
        )

        # Should NOT capture assistant recommendation
        assert not _any_fact_matches_pattern(facts, r"(?i)(recommend|suggest).*walk"), (
            f"Should not have extracted assistant recommendation: {facts}"
        )

        print(f"Extracted {len(facts)} facts from mixed summary:")
        for f in facts:
            print(f"  - {f}")
