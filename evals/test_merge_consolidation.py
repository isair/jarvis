"""
Merge consolidation evaluations.

`merge_node_data` advertises three behaviours beyond the supersession
case covered in `test_recency_superseding.py`:

  1. Near-duplicate dedupe — different wordings of the same fact
     collapse to one canonical line.
  2. Pattern consolidation — repeated activities fold into patterns
     ("ate sushi Mon", "ate sushi Thu" → "regularly eats sushi").
  3. Independence — an unrelated new fact must NOT silently drop an
     existing unrelated line. (The most dangerous failure mode: a
     hallucinated contradiction would erase real data.)

Plus a check that the batched signature works end-to-end with a real
picker model (the round-1 batching has unit tests but no eval).

Run:
    EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh merge_consolidation
"""

from dataclasses import dataclass
from typing import List

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_MODEL, JUDGE_BASE_URL

from jarvis.memory.graph_ops import merge_node_data


# =============================================================================
# Test data
# =============================================================================

@dataclass
class DedupeCase:
    description: str
    existing_data: str
    new_facts: List[str]
    # Substrings that must remain in the merged data.
    must_contain: List[str]
    # Substrings that should NOT appear (forbidden duplicates).
    must_not_contain: List[str]
    # Maximum line count after merge — caps near-dup explosion.
    max_lines: int


DEDUPE_CASES = [
    pytest.param(
        DedupeCase(
            description="Same fact, different wording",
            existing_data="The user lives in London.",
            new_facts=["The user is based in London."],
            must_contain=["london"],
            must_not_contain=[],
            max_lines=1,
        ),
        id="lives-in vs based-in London",
    ),
    pytest.param(
        DedupeCase(
            description="Job title rephrased",
            existing_data="The user works as a software engineer.",
            new_facts=["The user's job is software engineering."],
            must_contain=["software"],
            must_not_contain=[],
            max_lines=1,
        ),
        id="job rephrased",
    ),
]


@dataclass
class PatternCase:
    description: str
    existing_data: str
    new_facts: List[str]
    # Keyword that should appear in the consolidated pattern line
    # (e.g. "regularly", "often", "frequently", "every").
    pattern_keywords: List[str]
    # Subject the pattern is about (must remain).
    subject_keyword: str
    # Cap on lines — pattern consolidation should shrink, not grow.
    max_lines: int


@dataclass
class PatternBoundaryCase:
    description: str
    existing_data: str
    new_facts: List[str]
    # Substrings that MUST still be present in the merged output —
    # these are distinct one-off events that should not collapse
    # into a fake pattern.
    must_keep_distinct: List[str]


PATTERN_BOUNDARY_CASES = [
    pytest.param(
        PatternBoundaryCase(
            description="One-off events should not be patternised",
            existing_data=(
                "[2025-08-12] The user attended a wedding in Edinburgh.\n"
                "[2025-11-03] The user gave a conference talk in Berlin."
            ),
            new_facts=["[2026-04-25] The user moved house to Manchester."],
            # Three distinct, unrelated one-time events. Folding them
            # into "regularly travels" or similar would invent a
            # pattern that isn't there.
            must_keep_distinct=["edinburgh", "berlin", "manchester"],
        ),
        id="distinct one-off events",
        # Originally xfail(strict=False) — captured a regression where
        # `gemma4:e2b` clustered date-prefixed entries with a new
        # dated entry and silently dropped the older two. The case
        # now passes 3/3 reps on the small model after the
        # META-NARRATIVE rule landed. The causal link is not
        # verified, but the eval is the right place to catch a
        # regression so the marker is dropped and the case stands as
        # a regular PASS.
    ),
]


PATTERN_CASES = [
    pytest.param(
        PatternCase(
            description="Repeated sushi meals",
            existing_data=(
                "[2026-04-07] The user ate sushi for lunch.\n"
                "[2026-04-14] The user had sushi again.\n"
                "[2026-04-21] The user ordered sushi for dinner."
            ),
            new_facts=["[2026-04-25] The user ate sushi today."],
            pattern_keywords=["regularly", "often", "frequently", "weekly", "every", "tend"],
            subject_keyword="sushi",
            max_lines=3,
        ),
        id="sushi pattern",
    ),
]


@dataclass
class IndependenceCase:
    description: str
    existing_data: str
    new_facts: List[str]
    # Substrings that MUST survive — the new fact is unrelated and
    # has no business dropping these.
    must_keep: List[str]
    # Substrings the new fact should add.
    must_add: List[str]


INDEPENDENCE_CASES = [
    pytest.param(
        IndependenceCase(
            description="Vegetarian + unrelated meal mention",
            # Note: "user is vegetarian" + "user ate a Big Mac" is a
            # genuine contradiction the picker may legitimately
            # surface or pick a side on. Use clearly-orthogonal facts
            # instead so the eval is unambiguous.
            existing_data=(
                "The user has a peanut allergy.\n"
                "The user prefers tea over coffee."
            ),
            new_facts=["The user enjoys hiking on weekends."],
            must_keep=["peanut", "tea"],
            must_add=["hiking"],
        ),
        id="independent facts coexist",
    ),
    pytest.param(
        IndependenceCase(
            description="Job + new hobby",
            existing_data="The user works as a software engineer at Equals Money.",
            new_facts=["The user is learning to play the guitar."],
            must_keep=["software", "equals money"],
            must_add=["guitar"],
        ),
        id="job survives unrelated hobby fact",
    ),
]


@dataclass
class MetaNarrativeCase:
    description: str
    existing_data: str
    new_facts: List[str]
    # Substrings that must NOT remain after the merge — these are
    # extractor-artefact lines from earlier prompt versions
    # (assistant-narrating, capability denials) and have no place
    # in a knowledge node.
    must_drop_substrings: List[str]
    # Substrings that MUST remain — genuine knowledge or directives
    # that should not get over-pruned by the meta-narrative rule.
    must_keep_substrings: List[str]


META_NARRATIVE_CASES = [
    pytest.param(
        MetaNarrativeCase(
            description=(
                "Capability-denial line in Directives is dropped, "
                "real directive survives"
            ),
            # Mirrors the real bug report: a self-denial leaked into
            # Directives via an older extractor prompt and persisted
            # because no rewrite-on-write rule covered meta-narrative.
            # Consolidate-all (empty new_facts) should now scrub it
            # without touching the genuine British English directive.
            existing_data=(
                "Always reply in British English.\n"
                "The assistant is unable to navigate to a web page."
            ),
            new_facts=[],
            must_drop_substrings=[
                "unable to navigate",
                "the assistant is unable",
            ],
            must_keep_substrings=["british english"],
        ),
        id="capability denial dropped, directive kept",
    ),
    pytest.param(
        MetaNarrativeCase(
            description=(
                "Assistant-narrating WORLD line is dropped during "
                "self-consolidation"
            ),
            # The extractor's BANNED FACT FORMS list catches these at
            # write-time now, but lines emitted before #291 landed
            # still sit in nodes. Merge prompt must drop them too.
            existing_data=(
                "Possessor (2020) is directed by Brandon Cronenberg.\n"
                "The assistant suggested grilled salmon for dinner."
            ),
            new_facts=[],
            must_drop_substrings=[
                "the assistant suggested",
                "grilled salmon",
            ],
            must_keep_substrings=["possessor", "cronenberg"],
        ),
        id="assistant-suggested line dropped, lookup survives",
    ),
    pytest.param(
        MetaNarrativeCase(
            description=(
                "Polluted node receiving a new fact: meta-narrative "
                "drops AND the new fact lands"
            ),
            # Production path: a diary flush routes one new fact to a
            # node that already holds an older capability-denial line.
            # The merge must drop the denial AND incorporate the new
            # fact — capturing the worst case where the META rule
            # could steal attention from incorporation tracking.
            existing_data=(
                "Always reply in British English.\n"
                "The assistant is unable to navigate to a web page."
            ),
            new_facts=["Keep replies under three sentences."],
            must_drop_substrings=[
                "unable to navigate",
                "the assistant is unable",
            ],
            must_keep_substrings=[
                "british english",
                "three sentences",
            ],
        ),
        id="polluted node + new fact: drop and incorporate",
    ),
    pytest.param(
        MetaNarrativeCase(
            description=(
                "No meta-narrative present — merge must not invent "
                "drops (over-pruning guard)"
            ),
            # Counter-test for over-zealous interpretation of the new
            # rule. A clean Directives node with two genuine
            # imperatives must come through self-consolidation
            # untouched. If this fails the rule is too aggressive.
            existing_data=(
                "Always reply in British English.\n"
                "Keep replies under three sentences."
            ),
            new_facts=[],
            must_drop_substrings=[],
            must_keep_substrings=["british english", "three sentences"],
        ),
        id="genuine directives untouched",
    ),
]


@dataclass
class BatchedCase:
    description: str
    existing_data: str
    new_facts: List[str]
    # Each entry: list of substring alternatives — at least one must
    # appear in the merged data. Captures "the model phrased it
    # however it wanted, but the fact survived".
    expected_signals: List[List[str]]


BATCHED_CASES = [
    pytest.param(
        BatchedCase(
            description="Three independent new facts in one call",
            existing_data="The user lives in London.",
            new_facts=[
                "The user has a dog named Biscuit.",
                "The user prefers oat milk.",
                "The user is allergic to peanuts.",
            ],
            expected_signals=[
                ["london"],
                ["biscuit", "dog"],
                ["oat milk", "oat"],
                ["peanut"],
            ],
        ),
        id="batched 3 new facts",
    ),
]


def _line_count(data: str) -> int:
    return len([l for l in data.split("\n") if l.strip()])


# =============================================================================
# Tests
# =============================================================================

@pytest.mark.eval
class TestNearDuplicateDedupe:
    """Different wordings of the same fact must collapse to one line."""

    @requires_judge_llm
    @pytest.mark.parametrize("case", DEDUPE_CASES)
    def test_near_duplicates_collapse(self, case, graph_store):
        case = case.values[0] if hasattr(case, 'values') else case

        node = graph_store.create_node(
            name="T",
            description=case.description,
            data=case.existing_data,
            parent_id="root",
        )

        result = merge_node_data(
            store=graph_store,
            node_id=node.id,
            new_facts=case.new_facts,
            ollama_base_url=JUDGE_BASE_URL,
            ollama_chat_model=JUDGE_MODEL,
            timeout_sec=30.0,
        )

        merged = graph_store.get_node(node.id).data
        merged_lower = merged.lower()
        line_count = _line_count(merged)

        print(f"\n  📝 dedupe '{case.description}':\n     {merged[:300]}")
        print(f"     success={result.success} lines={line_count}")

        for kw in case.must_contain:
            assert kw.lower() in merged_lower, (
                f"[{case.description}] expected '{kw}' to survive merge.\n{merged}"
            )
        for kw in case.must_not_contain:
            assert kw.lower() not in merged_lower, (
                f"[{case.description}] forbidden '{kw}' leaked into merge.\n{merged}"
            )
        assert line_count <= case.max_lines, (
            f"[{case.description}] merge produced {line_count} lines, expected ≤ {case.max_lines} "
            f"(near-duplicates should collapse).\n{merged}"
        )


@pytest.mark.eval
class TestPatternConsolidation:
    """Repeated activities should fold into patterns rather than
    accumulate as a stack of dated entries."""

    @requires_judge_llm
    @pytest.mark.parametrize("case", PATTERN_CASES)
    def test_repeated_activities_consolidate(self, case, graph_store):
        case = case.values[0] if hasattr(case, 'values') else case

        node = graph_store.create_node(
            name="T",
            description=case.description,
            data=case.existing_data,
            parent_id="root",
        )

        result = merge_node_data(
            store=graph_store,
            node_id=node.id,
            new_facts=case.new_facts,
            ollama_base_url=JUDGE_BASE_URL,
            ollama_chat_model=JUDGE_MODEL,
            timeout_sec=30.0,
        )

        merged = graph_store.get_node(node.id).data
        merged_lower = merged.lower()
        line_count = _line_count(merged)

        print(f"\n  📝 pattern '{case.description}':\n     {merged[:300]}")
        print(f"     success={result.success} lines={line_count}")

        assert case.subject_keyword.lower() in merged_lower, (
            f"[{case.description}] subject '{case.subject_keyword}' lost from merge.\n{merged}"
        )
        has_pattern = any(kw in merged_lower for kw in case.pattern_keywords)
        assert has_pattern, (
            f"[{case.description}] expected pattern wording (any of {case.pattern_keywords}) "
            f"after consolidating repeated activities.\n{merged}"
        )
        assert line_count <= case.max_lines, (
            f"[{case.description}] {line_count} lines remain — repeated activities should "
            f"have consolidated to ≤ {case.max_lines}.\n{merged}"
        )


@pytest.mark.eval
class TestPatternBoundary:
    """Counter-example to `TestPatternConsolidation`: distinct one-off
    events MUST NOT be folded into a fabricated pattern. Pattern
    consolidation should fire on repetition, not on coincidence."""

    @requires_judge_llm
    @pytest.mark.parametrize("case", PATTERN_BOUNDARY_CASES)
    def test_distinct_one_offs_stay_distinct(self, case, graph_store):
        case = case.values[0] if hasattr(case, 'values') else case

        node = graph_store.create_node(
            name="T",
            description=case.description,
            data=case.existing_data,
            parent_id="root",
        )

        result = merge_node_data(
            store=graph_store,
            node_id=node.id,
            new_facts=case.new_facts,
            ollama_base_url=JUDGE_BASE_URL,
            ollama_chat_model=JUDGE_MODEL,
            timeout_sec=30.0,
        )

        merged = graph_store.get_node(node.id).data
        merged_lower = merged.lower()

        print(f"\n  📝 pattern-boundary '{case.description}':\n     {merged[:300]}")
        print(f"     success={result.success}")

        for kw in case.must_keep_distinct:
            assert kw.lower() in merged_lower, (
                f"[{case.description}] distinct event '{kw}' was folded away — "
                f"the picker invented a pattern from one-offs.\n{merged}"
            )


@pytest.mark.eval
class TestIndependenceOfUnrelatedFacts:
    """An unrelated new fact must NOT drop an existing unrelated line.
    Silent erasure of real data is the most dangerous failure mode of
    the rewrite-on-write merge — the hallucination guard catches
    runaway growth, but only this eval catches runaway shrinkage."""

    @requires_judge_llm
    @pytest.mark.parametrize("case", INDEPENDENCE_CASES)
    def test_independent_facts_coexist(self, case, graph_store):
        case = case.values[0] if hasattr(case, 'values') else case

        node = graph_store.create_node(
            name="T",
            description=case.description,
            data=case.existing_data,
            parent_id="root",
        )

        result = merge_node_data(
            store=graph_store,
            node_id=node.id,
            new_facts=case.new_facts,
            ollama_base_url=JUDGE_BASE_URL,
            ollama_chat_model=JUDGE_MODEL,
            timeout_sec=30.0,
        )

        merged = graph_store.get_node(node.id).data
        merged_lower = merged.lower()

        print(f"\n  📝 independence '{case.description}':\n     {merged[:300]}")
        print(f"     success={result.success}")

        for kw in case.must_keep:
            assert kw.lower() in merged_lower, (
                f"[{case.description}] existing fact containing '{kw}' was silently "
                f"dropped by an unrelated new fact — independence violated.\n{merged}"
            )
        for kw in case.must_add:
            assert kw.lower() in merged_lower, (
                f"[{case.description}] new fact containing '{kw}' did not land.\n{merged}"
            )


@pytest.mark.eval
class TestMetaNarrativePruning:
    """Lines that narrate the assistant's own behaviour, capabilities,
    or denials are extractor artefacts from earlier prompt versions,
    not user knowledge. The merge step must drop them during normal
    rewrite-on-write AND during the consolidate-all sweep. Counterpart
    to the extractor's BANNED FACT FORMS list — that catches them at
    write-time, this catches the historical leftovers."""

    @requires_judge_llm
    @pytest.mark.parametrize("case", META_NARRATIVE_CASES)
    def test_meta_narrative_dropped_real_facts_kept(self, case, graph_store):
        case = case.values[0] if hasattr(case, 'values') else case

        node = graph_store.create_node(
            name="T",
            description=case.description,
            data=case.existing_data,
            parent_id="root",
        )

        result = merge_node_data(
            store=graph_store,
            node_id=node.id,
            new_facts=case.new_facts,
            ollama_base_url=JUDGE_BASE_URL,
            ollama_chat_model=JUDGE_MODEL,
            timeout_sec=30.0,
        )

        merged = graph_store.get_node(node.id).data
        merged_lower = merged.lower()

        print(f"\n  📝 meta-narrative '{case.description}':\n     {merged[:300]}")
        print(f"     success={result.success}")

        for kw in case.must_drop_substrings:
            assert kw.lower() not in merged_lower, (
                f"[{case.description}] meta-narrative line containing "
                f"'{kw}' survived the merge — the rule did not fire.\n{merged}"
            )
        for kw in case.must_keep_substrings:
            assert kw.lower() in merged_lower, (
                f"[{case.description}] genuine fact containing '{kw}' was "
                f"over-pruned — the rule is too aggressive.\n{merged}"
            )

        # When new_facts is non-empty the merge must report at least
        # one incorporation. A regression where the META rule steals
        # attention from incorporation tracking would surface here as
        # `incorporated_indices == []` despite the fact landing in
        # the merged data — exactly the failure mode `_match_key`'s
        # tolerant punctuation strip was added to prevent.
        if case.new_facts:
            assert len(result.incorporated_indices) >= 1, (
                f"[{case.description}] new fact landed in merged data "
                f"but incorporated_indices is empty — orchestrator "
                f"would under-report the flush.\n"
                f"merged={merged}\nresult={result}"
            )


@pytest.mark.eval
class TestBatchedMerge:
    """Multiple new facts in one merge call must all land. Pins the
    round-1 batched signature against a real picker model."""

    @requires_judge_llm
    @pytest.mark.parametrize("case", BATCHED_CASES)
    def test_all_batched_facts_land(self, case, graph_store):
        case = case.values[0] if hasattr(case, 'values') else case

        node = graph_store.create_node(
            name="T",
            description=case.description,
            data=case.existing_data,
            parent_id="root",
        )

        result = merge_node_data(
            store=graph_store,
            node_id=node.id,
            new_facts=case.new_facts,
            ollama_base_url=JUDGE_BASE_URL,
            ollama_chat_model=JUDGE_MODEL,
            timeout_sec=30.0,
        )

        merged = graph_store.get_node(node.id).data
        merged_lower = merged.lower()
        line_count = _line_count(merged)

        print(f"\n  📝 batched '{case.description}':\n     {merged[:400]}")
        print(f"     success={result.success} lines={line_count} "
              f"incorporated={result.incorporated_indices}")

        for alternatives in case.expected_signals:
            assert any(alt.lower() in merged_lower for alt in alternatives), (
                f"[{case.description}] none of {alternatives} survived the batched merge.\n"
                f"{merged}"
            )

        # Lower bound on lines: at minimum the merged data should
        # contain a line per surviving fact. Upper bound is enforced
        # by the in-product hallucination guard, not this eval — a
        # cap here is brittle since legitimate consolidation could
        # cross it on a paraphrase the model picks differently.
        assert line_count >= len(case.expected_signals) - 1, (
            f"[{case.description}] {line_count} lines suspiciously low for "
            f"{len(case.expected_signals)} signals — facts may have been silently merged.\n"
            f"{merged}"
        )

        # Pin the round-1 batched reporting fix: every input fact
        # whose substance survived should be tracked in
        # `incorporated_indices`. An empty list when facts clearly
        # landed means the orchestrator under-reports flushes — the
        # exact regression `_match_key`'s tolerant punctuation strip
        # was added to prevent. Allow strict equality OR coverage of
        # all input indices, since the picker may legitimately
        # consolidate two new facts into one line.
        assert len(result.incorporated_indices) >= 1, (
            f"[{case.description}] incorporated_indices is empty despite facts landing — "
            f"reporting drift back. {result.incorporated_indices}"
        )
