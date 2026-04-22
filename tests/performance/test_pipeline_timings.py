"""⏱️ Performance: time each LLM context in the reply pipeline.

Runs ``run_reply_engine`` N times against a live Ollama with a fixed tiny
prompt, records per-context timings via the monkey-patching recorder, and
asserts a few relative-shape invariants so the test fails when the pipeline
shape drifts (e.g. the evaluator becomes more expensive than the main turn).

Also includes a micro-benchmark that calls each configured model with a
tiny fixed prompt, giving a hardware baseline to diff against.

Run manually:
    pytest tests/performance/ -v -m performance -s

Requires:
    - Ollama reachable at http://localhost:11434
    - ``gemma4:e2b`` pulled (or override via env var)

The test is skipped automatically if Ollama is unreachable, so it's safe to
leave in the repo. Use ``-s`` to see the report table.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import requests

from tests.performance.timing_recorder import TimingRecorder


OLLAMA_URL = os.environ.get("JARVIS_PERF_OLLAMA_URL", "http://localhost:11434")
PERF_MODEL = os.environ.get("JARVIS_PERF_MODEL", "gemma4:e2b")
PERF_RUNS = int(os.environ.get("JARVIS_PERF_RUNS", "3"))
PERF_REPORT_DIR = Path(os.environ.get(
    "JARVIS_PERF_REPORT_DIR",
    str(Path(__file__).parent / "reports"),
))

# Tiny fixed prompts — the whole point of the baseline is to measure the
# per-call overhead and model warmup cost, not prompt-length effects.
TINY_SYSTEM = "Reply with the single word OK."
TINY_USER = "ping"

# Representative reply-pipeline queries. Keep them small and shape-diverse.
PIPELINE_QUERIES = [
    "hello",                      # pure chat, no tools needed
    "what's 2 plus 3?",           # math, one-shot
    "what time is it in Tokyo?",  # likely triggers a tool
]


def _ollama_reachable() -> bool:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if resp.status_code != 200:
            return False
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return any(PERF_MODEL.split(":")[0] in m for m in models)
    except Exception:
        return False


pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(
        not _ollama_reachable(),
        reason=f"Ollama at {OLLAMA_URL} with {PERF_MODEL} not available",
    ),
]


def _make_cfg():
    from evals.helpers import MockConfig
    cfg = MockConfig()
    cfg.ollama_base_url = OLLAMA_URL
    cfg.ollama_chat_model = PERF_MODEL
    cfg.intent_judge_model = PERF_MODEL
    # Let size-aware defaults kick in (evaluator + digests ON for small).
    cfg.evaluator_enabled = None
    cfg.memory_digest_enabled = None
    cfg.tool_result_digest_enabled = None
    # Force the LLM-based router so its timing shows up in the report.
    # MockConfig doesn't set this attribute, and the engine's default varies.
    cfg.tool_selection_strategy = "llm"
    cfg.tool_router_model = ""  # fall through the router chain
    cfg.evaluator_model = ""
    return cfg


def _write_report(rec: TimingRecorder, name: str) -> Path:
    PERF_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = PERF_REPORT_DIR / f"{name}-{stamp}.json"
    payload = {
        "name": name,
        "timestamp": stamp,
        "model": PERF_MODEL,
        "runs": PERF_RUNS,
        "summary": rec.to_dict(),
        "raw": [
            {
                "context": c.context,
                "duration_sec": round(c.duration_sec, 4),
                "model": c.model,
                "prompt_chars": c.prompt_chars,
                "response_chars": c.response_chars,
            }
            for c in rec.calls
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


# =============================================================================
# Micro-benchmark: tiny fixed prompt per configured model
# =============================================================================


@pytest.mark.performance
def test_micro_benchmark_tiny_prompt():
    """Baseline: how long does a single tiny round-trip to Ollama take?

    This is the floor for every context's per-call cost. If the floor moves,
    every context's total moves with it. Reported separately from the
    pipeline test so hardware drift is obvious in the numbers.
    """
    # Import the module (not the function) so the recorder's patch on
    # jarvis.llm is visible at call time.
    from jarvis import llm as _llm

    with TimingRecorder() as rec:
        # Warmup (first call pays weight-loading cost)
        _llm.call_llm_direct(
            base_url=OLLAMA_URL,
            chat_model=PERF_MODEL,
            system_prompt=TINY_SYSTEM,
            user_content=TINY_USER,
            timeout_sec=30.0,
        )
        # Measured runs
        for _ in range(PERF_RUNS):
            _llm.call_llm_direct(
                base_url=OLLAMA_URL,
                chat_model=PERF_MODEL,
                system_prompt=TINY_SYSTEM,
                user_content=TINY_USER,
                timeout_sec=30.0,
            )

    rec.print_report(title=f"Micro-benchmark — tiny prompt × {PERF_RUNS + 1} on {PERF_MODEL}")
    path = _write_report(rec, "micro")
    print(f"   📄 saved: {path}")

    # Shape check: warm calls should be noticeably faster than cold.
    # Not a strict assertion (too noisy) — just make sure we got calls.
    assert len(rec.calls) == PERF_RUNS + 1


# =============================================================================
# Full pipeline: run_reply_engine × N, per-context timings
# =============================================================================


@pytest.mark.performance
def test_pipeline_timings_by_context():
    """Run the full reply pipeline N times, record per-context timings.

    Relative-shape invariants (not absolute numbers):
      1. If the evaluator fires, it must be cheaper on average than the main
         chat turn — otherwise we're paying more for the decision than for
         the answer. This is the whole reason the evaluator uses a small
         model.
      2. The tool router, if it fires, must be cheaper than a main chat
         turn on p50 — it's a classification call on the warm small model.
      3. Enrichment extractor, if it fires, must run on the router chain
         (same model as the router). This locks in the demotion we just did.
    """
    from jarvis.memory.db import Database
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.reply.engine import run_reply_engine

    cfg = _make_cfg()

    with TimingRecorder() as rec:
        for query in PIPELINE_QUERIES:
            db = Database(":memory:", sqlite_vss_path=None)
            dlg = DialogueMemory(inactivity_timeout=300, max_interactions=20)
            try:
                for _ in range(PERF_RUNS):
                    run_reply_engine(db, cfg, None, query, dlg)
            finally:
                db.close()

    rec.print_report(title=f"Pipeline timings — {len(PIPELINE_QUERIES)} queries × {PERF_RUNS} runs on {PERF_MODEL}")
    path = _write_report(rec, "pipeline")
    print(f"   📄 saved: {path}")

    assert rec.calls, "no LLM calls recorded — pipeline did not invoke the LLM"

    # Surface unmapped callers so new contexts show up in review.
    other = [c for c in rec.calls if c.context.startswith("other:")]
    if other:
        unmapped = sorted({c.context for c in other})
        print(f"   ⚠️  unmapped callers (add to _CALLER_TO_CONTEXT): {unmapped}")

    # Shape invariants
    main_p50 = rec.p50("main_chat_turn")
    if main_p50 > 0:
        ev_p50 = rec.p50("evaluator")
        if ev_p50 > 0:
            assert ev_p50 <= main_p50 * 1.5, (
                f"evaluator p50 ({ev_p50:.2f}s) exceeds main chat turn p50 "
                f"({main_p50:.2f}s) by >50% — evaluator should be cheaper"
            )
        router_p50 = rec.p50("tool_router")
        if router_p50 > 0:
            assert router_p50 <= main_p50 * 1.5, (
                f"tool router p50 ({router_p50:.2f}s) exceeds main chat turn p50 "
                f"({main_p50:.2f}s) by >50% — router should be cheaper"
            )

    # Locking in the demotion: enrichment extractor must use the router chain.
    enrich_calls = [c for c in rec.calls if c.context == "enrichment_extract"]
    router_calls = [c for c in rec.calls if c.context == "tool_router"]
    if enrich_calls and router_calls:
        enrich_models = {c.model for c in enrich_calls}
        router_models = {c.model for c in router_calls}
        assert enrich_models == router_models, (
            f"enrichment extractor should share the router model chain "
            f"(enrichment={enrich_models}, router={router_models})"
        )
