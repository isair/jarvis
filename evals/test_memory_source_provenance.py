"""Live check that memory digestion preserves source authority and attribution."""

import re
import time

import pytest

from evals.conftest import _JUDGE_LLM_AVAILABLE
from evals.benchmark_report import Attempt, Scenario
from evals.helpers import JUDGE_MODEL, MockConfig
from evals.residence_assessment import has_current_residence_claim
from jarvis.reply.enrichment import digest_memory_for_query


def _has_unrefuted_place(text: str, place: str, *, allow_history: bool) -> bool:
    """Treat an old place as a claim unless each mention is explicitly qualified."""
    for match in re.finditer(rf"\b{re.escape(place)}\b", text):
        before = text[max(0, match.start() - 80):match.start()]
        after = text[match.end():match.end() + 60]
        negated = bool(
            re.search(r"\bnot\s+(?:in\s+)?$", before)
            or re.search(r"\bno\s+longer\s+(?:(?:lives?|living|resides?|based)\s+)?(?:in\s+)?$", before)
        )
        historical = allow_history and bool(
            re.search(r"\b(?:formerly|previously)\s+(?:(?:lived|living|resided|based|home|residence|in)\s+){0,3}$", before)
            or re.match(r"\s+(?:was|is)\s+(?:a\s+|the\s+|my\s+)?(?:former|previous)\b", after)
        )
        if not (negated or historical):
            return True
    return False


def _assess_residence_digest(digest: str) -> tuple[bool, bool]:
    """Return stale and unsupported residence claims for this source scenario."""
    text = digest.casefold()
    stale_retrieval = (
        not has_current_residence_claim(text, "bath")
        or _has_unrefuted_place(text, "bristol", allow_history=True)
    )
    unsupported_claim = _has_unrefuted_place(text, "paris", allow_history=False)
    return stale_retrieval, unsupported_claim


@pytest.mark.parametrize("digest, expected", [
    ("The user lives in Bath, not Bristol (said on 2026-03-01).", (False, False)),
    ("The user lives in Bath. Bristol is a former residence.", (False, False)),
    ("The user's current home is Bath, not Bristol.", (False, False)),
    ("Bath is where the user lives; Bristol is a former residence.", (False, False)),
    ("Current residence: Bath. Not Bristol.", (False, False)),
    ("The user lives in Bath, not Paris.", (False, False)),
    ("Bath is a city. The user's current residence is unknown; not Bristol.", (True, False)),
    ("The user does not live in Bath. Bristol is a former residence.", (True, False)),
    ("The user formerly lived in Bath; current residence unknown.", (True, False)),
    ("The user lives in Bristol, not Bath.", (True, False)),
    ("Not Bath, not Bristol; the location is unknown.", (True, False)),
    ("The user lives in Bath and Bristol.", (True, False)),
    ("Bath, not Bristol; Bristol is also their current home.", (True, False)),
    ("The user lives in Bath and Paris.", (False, True)),
    ("The user lives in Bath; Paris was an assistant claim.", (False, True)),
])
def test_digest_residence_assessment(digest, expected):
    assert _assess_residence_digest(digest) == expected


@pytest.mark.eval
def test_digest_prefers_cited_user_correction_over_unverified_claims(scenario_recorder):
    if not _JUDGE_LLM_AVAILABLE:
        scenario_recorder(Scenario("source precedence", "live_memory_provenance", [], availability="unavailable"))
        pytest.skip("Judge LLM not available")
    cfg = MockConfig()
    cfg.llm_chat_model = JUDGE_MODEL
    entries = [
        "[User statement; 2026-03-01; dialogue; evidence: I live in Bath now, not Bristol] "
        "The user lives in Bath.",
        "[Unverified legacy graph; User > Home; last edited 2026-08-01] "
        "The user lives in Bristol. This note has no source evidence and cannot supersede "
        "the explicit user correction, even if its edit date is later.",
        "[Diary summary; reference only] [2026-09-01] The assistant said the user "
        "lives in Paris during a conversation about travel. This was an assistant claim, "
        "not a user statement, and was not confirmed by the user.",
    ]
    started = time.perf_counter()
    digest_sec = None
    passed = False
    stale_retrieval = False
    unsupported_claim = False
    try:
        digest = digest_memory_for_query(
            query="Where do I live?", source_entries=entries,
            diary_entries=[], graph_parts=[], cfg=cfg, chat_model=JUDGE_MODEL,
            timeout_sec=60.0,
        )
        digest_sec = time.perf_counter() - started
        stale_retrieval, unsupported_claim = _assess_residence_digest(digest)
        assert not stale_retrieval
        assert not unsupported_claim
        passed = True
    finally:
        elapsed = time.perf_counter() - started
        scenario_recorder(Scenario(
            "source precedence", "live_memory_provenance",
            [Attempt(passed, stale_retrieval=stale_retrieval,
                     unsupported_claim=unsupported_claim)],
            stage_durations={"memory_digest": [digest_sec if digest_sec is not None else elapsed]},
            end_to_end_sec=elapsed,
        ))
