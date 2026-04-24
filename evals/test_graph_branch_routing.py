"""
Knowledge Graph Branch Routing Evaluations

Validates the extractor's per-fact branch classification (USER / DIRECTIVES
/ WORLD). The warm profile injected into every reply is the User +
Directives branches concatenated — misclassification here either leaks
directives out of the warm blob (the assistant forgets a standing rule)
or dumps world trivia into the blob (every reply carries irrelevant
background). Both are nasty, silent regressions, so the classification
accuracy needs its own eval.

Cases are deliberately adversarial around the swap-test boundary:
- User statements about themselves that a naive classifier might read
  as a directive ("I prefer short answers" → USER, not DIRECTIVES —
  it's a preference about the user, not an instruction).
- Imperatives to the assistant that a naive classifier might read as
  user preferences ("always reply briefly" → DIRECTIVES, not USER).
- World facts where the user is also the subject of the request but
  the fact itself is external attribution.

Run:
    EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh graph_branch_routing
    EVAL_JUDGE_MODEL=gpt-oss:20b ./scripts/run_evals.sh graph_branch_routing
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import pytest

from conftest import requires_judge_llm
from helpers import MockConfig

from jarvis.memory.graph import BRANCH_DIRECTIVES, BRANCH_USER, BRANCH_WORLD
from jarvis.memory.graph_ops import extract_graph_memories


# =============================================================================
# Test Data
# =============================================================================


@dataclass
class RoutingCase:
    """A summary and the branches we expect each keyword-identified fact
    to be routed into."""

    summary: str
    date_utc: Optional[str] = None
    # Each expectation is ``(keyword_or_alternatives, expected_branch_id)``.
    # If the first item is a tuple, any one of its strings satisfies the
    # match — use this when the model may paraphrase. Matching is
    # case-insensitive substring on fact text.
    expectations: List[Tuple[Union[str, Tuple[str, ...]], str]] = field(
        default_factory=list,
    )


ROUTING_CASES = [
    # ── Clear USER facts ────────────────────────────────────────────────
    pytest.param(
        RoutingCase(
            summary=(
                "The user mentioned they live in Brighton and have two "
                "cats, Miso and Kuma. They've been vegetarian for five "
                "years and work as a backend engineer."
            ),
            date_utc="2026-04-20",
            expectations=[
                ("Brighton", BRANCH_USER),
                ("Miso", BRANCH_USER),
                ("vegetarian", BRANCH_USER),
                ("engineer", BRANCH_USER),
            ],
        ),
        id="USER: identity, location, pets, diet, job",
    ),
    # ── Clear DIRECTIVES ─────────────────────────────────────────────────
    pytest.param(
        RoutingCase(
            summary=(
                "The user told me to always answer in British English, "
                "to keep replies under three sentences, and to never "
                "apologise or say sorry. They also asked me to address "
                "them as Boss going forward."
            ),
            date_utc="2026-04-20",
            expectations=[
                ("British English", BRANCH_DIRECTIVES),
                ("three sentences", BRANCH_DIRECTIVES),
                ("apologise", BRANCH_DIRECTIVES),
                ("Boss", BRANCH_DIRECTIVES),
            ],
        ),
        id="DIRECTIVES: tone, length, forbidden phrases, address form",
    ),
    # ── Clear WORLD facts ────────────────────────────────────────────────
    pytest.param(
        RoutingCase(
            summary=(
                "The user asked about Trenches Boxing Club. I found that "
                "it's on Mare Street in Hackney, offers evening classes "
                "on weekdays from 6-8pm at 15 pounds per session. I also "
                "confirmed that Possessor is a 2020 sci-fi horror film "
                "directed by Brandon Cronenberg."
            ),
            date_utc="2026-04-20",
            expectations=[
                ("Trenches", BRANCH_WORLD),
                ("Mare Street", BRANCH_WORLD),
                ("Possessor", BRANCH_WORLD),
                ("Cronenberg", BRANCH_WORLD),
            ],
        ),
        id="WORLD: local business details, film attribution",
    ),
    # ── Adversarial: preference vs directive ────────────────────────────
    pytest.param(
        RoutingCase(
            summary=(
                "The user said they prefer Thai food over Italian when "
                "eating out. They also told me to keep all food "
                "recommendations under five options, because longer "
                "lists overwhelm them."
            ),
            date_utc="2026-04-20",
            expectations=[
                # Preference about the user's own tastes → USER
                ("Thai", BRANCH_USER),
                # Instruction about assistant behaviour → DIRECTIVES
                ("five options", BRANCH_DIRECTIVES),
            ],
        ),
        id="Adversarial: food preference (USER) vs list-length rule (DIRECTIVES)",
    ),
    # ── Adversarial: mixed summary ──────────────────────────────────────
    pytest.param(
        RoutingCase(
            summary=(
                "The user has been vegetarian for three years and lives "
                "in central London. They told me to stop suggesting fish "
                "dishes when they ask about food — they consider "
                "pescatarian suggestions unhelpful. I confirmed that "
                "Mildreds in Covent Garden is a fully vegetarian "
                "restaurant with a Michelin Bib Gourmand rating."
            ),
            date_utc="2026-04-20",
            expectations=[
                ("Mildreds", BRANCH_WORLD),
                ("vegetarian for three years", BRANCH_USER),
                # Model phrases the directive either as "pescatarian
                # suggestions unhelpful" or "fish dishes" — accept
                # either; the classification is what matters.
                (("pescatarian", "fish"), BRANCH_DIRECTIVES),
            ],
        ),
        id="Adversarial: all three branches in one summary",
    ),
]


# =============================================================================
# Helpers
# =============================================================================


def _run_extraction(case: RoutingCase, config: MockConfig) -> list[tuple[str, str]]:
    return extract_graph_memories(
        summary=case.summary,
        ollama_base_url=config.ollama_base_url,
        ollama_chat_model=config.ollama_chat_model,
        timeout_sec=config.llm_chat_timeout_sec,
        thinking=False,
        date_utc=case.date_utc,
    )


def _find_branch_for_keyword(
    facts: list[tuple[str, str]],
    keyword: Union[str, Tuple[str, ...]],
) -> Optional[str]:
    """Return the branch_id of the first fact whose text contains keyword
    (case-insensitive), or None if no fact matches. If keyword is a tuple,
    any of its strings satisfies the match."""
    alternatives = (keyword,) if isinstance(keyword, str) else keyword
    lowered = [k.lower() for k in alternatives]
    for branch_id, fact in facts:
        fact_lower = fact.lower()
        if any(k in fact_lower for k in lowered):
            return branch_id
    return None


# =============================================================================
# Tests
# =============================================================================


class TestGraphBranchRouting:
    """Branch classification accuracy for the knowledge extractor."""

    @requires_judge_llm
    @pytest.mark.parametrize("case", ROUTING_CASES)
    def test_routes_facts_to_expected_branches(
        self, mock_config, case: RoutingCase,
    ):
        facts = _run_extraction(case, mock_config)

        # Print for report visibility
        print(f"Extracted {len(facts)} facts:")
        for branch_id, fact in facts:
            print(f"  [{branch_id}] {fact}")

        # Every expectation must be satisfied
        for keyword, expected_branch in case.expectations:
            actual_branch = _find_branch_for_keyword(facts, keyword)
            assert actual_branch is not None, (
                f"Expected a fact containing {keyword!r} (for branch "
                f"{expected_branch!r}), but no extracted fact matched. "
                f"Facts: {facts}"
            )
            assert actual_branch == expected_branch, (
                f"Keyword {keyword!r}: expected branch "
                f"{expected_branch!r}, got {actual_branch!r}. Facts: "
                f"{facts}"
            )
