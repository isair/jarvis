"""Live check that memory digestion preserves source authority and attribution."""

import time

import pytest

from evals.conftest import _JUDGE_LLM_AVAILABLE
from evals.benchmark_report import Attempt, Scenario
from evals.helpers import JUDGE_MODEL, MockConfig
from jarvis.reply.enrichment import digest_memory_for_query


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
        lowered = digest.lower()
        stale_retrieval = "bath" not in lowered or ("bristol" in lowered and "former" not in lowered)
        unsupported_claim = "paris" in lowered
        assert "bath" in lowered
        assert not unsupported_claim
        assert "bristol" not in lowered or "former" in lowered
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
