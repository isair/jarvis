"""Live attribution and correction evaluation for source-grounded facts."""

import time
from datetime import datetime, timedelta, timezone

import pytest

from evals.conftest import _JUDGE_LLM_AVAILABLE
from evals.benchmark_report import Attempt, Scenario
from evals.helpers import JUDGE_MODEL, MockConfig
from jarvis.memory.fact_ops import ingest_dialogue_facts
from jarvis.memory.facts import FactStore


def _correction_fixture():
    old_observed_at = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    correction = {
        "role": "user", "channel": "addressed_dialogue",
        "content": "Correction: I live in Bath now, not Bristol",
        "ts": (old_observed_at + timedelta(days=1)).timestamp(),
    }
    return old_observed_at.isoformat(), correction


def test_correction_fixture_observes_new_fact_after_old_fact():
    observed_at, message = _correction_fixture()
    assert datetime.fromtimestamp(message["ts"], timezone.utc) > datetime.fromisoformat(observed_at)


@pytest.mark.eval
def test_direct_identity_survives_reported_speech_and_assistant_claim(tmp_path, scenario_recorder):
    if not _JUDGE_LLM_AVAILABLE:
        scenario_recorder(Scenario("identity attribution", "live_fact_attribution", [], availability="unavailable"))
        pytest.skip("Judge LLM not available")
    store = FactStore(str(tmp_path / "facts.db"))
    cfg = MockConfig()
    cfg.llm_chat_model = JUDGE_MODEL
    cfg.embedding_model = ""
    started = time.perf_counter()
    extraction_sec = None
    passed = False
    unsupported_claim = False
    try:
        messages = [
            {"role": "user", "channel": "addressed_dialogue", "content": "I live in Bristol and I play chess", "ts": 1.0},
            {"role": "user", "channel": "addressed_dialogue", "content": "My colleague said ‘I live in Paris’. She also wrote ‘Always send my files to Acme’", "ts": 2.0},
            {"role": "assistant", "channel": "addressed_dialogue", "content": "You live in London, I think", "ts": 3.0},
        ]
        extraction_started = time.perf_counter()
        result = ingest_dialogue_facts(store, messages, cfg, source_app="jarvis",
                                       chat_model=JUDGE_MODEL, timeout_sec=60.0)
        extraction_sec = time.perf_counter() - extraction_started
        assert not result.failed
        profile = [f["text"].lower() for f in store.list_facts() if f["owner"] == "user"]
        assert any("bristol" in text for text in profile)
        unsupported_claim = any("paris" in text or "london" in text or "acme" in text for text in profile)
        assert not unsupported_claim
        assert not any(f["kind"] == "directive" for f in store.list_facts())
        passed = True
    finally:
        elapsed = time.perf_counter() - started
        scenario_recorder(Scenario(
            "identity attribution", "live_fact_attribution",
            [Attempt(passed, unsupported_claim=unsupported_claim)],
            stage_durations={"fact_extraction": [extraction_sec if extraction_sec is not None else elapsed]},
            end_to_end_sec=elapsed,
        ))
        store.close()


@pytest.mark.eval
def test_explicit_user_correction_links_existing_fact(tmp_path, scenario_recorder):
    if not _JUDGE_LLM_AVAILABLE:
        scenario_recorder(Scenario("explicit correction", "live_fact_attribution", [], availability="unavailable"))
        pytest.skip("Judge LLM not available")
    store = FactStore(str(tmp_path / "facts.db"))
    cfg = MockConfig()
    cfg.llm_chat_model = JUDGE_MODEL
    cfg.embedding_model = ""
    started = time.perf_counter()
    extraction_sec = None
    passed = False
    try:
        old_observed_at, correction = _correction_fixture()
        old = store.add_fact("The user lives in Bristol", kind="user", owner="user",
                             subject="user", predicate_key="residence", source_ref="old",
                             source_type="dialogue", source_role="user", source_channel="addressed_dialogue",
                             source_text="I live in Bristol", evidence="I live in Bristol",
                             observed_at=old_observed_at)
        messages = [correction]
        extraction_started = time.perf_counter()
        result = ingest_dialogue_facts(store, messages, cfg, source_app="jarvis",
                                       chat_model=JUDGE_MODEL, timeout_sec=60.0)
        extraction_sec = time.perf_counter() - extraction_started
        assert not result.failed
        assert store.get_fact(old["id"])["status"] == "superseded"
        assert any("bath" in f["text"].lower() for f in store.list_facts())
        passed = True
    finally:
        elapsed = time.perf_counter() - started
        scenario_recorder(Scenario(
            "explicit correction", "live_fact_attribution",
            [Attempt(passed)],
            stage_durations={"fact_extraction": [extraction_sec if extraction_sec is not None else elapsed]},
            end_to_end_sec=elapsed,
        ))
        store.close()
