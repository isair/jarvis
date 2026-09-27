"""⏱️ Performance measurements for either local LLM provider.

The harness reports backend-call timings by context and request wall time.
It does not observe text streaming or audio output, so first useful text and
first useful spoken output remain unmeasured.

Run manually:
    pytest tests/performance/ -v -m performance -s

Requires a reachable local provider and an already installed model.

The test skips when the model is unavailable. Use ``-s`` to see timings.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from evals.benchmark_report import latency_summary
from scripts.eval_provider import probe_provider
from tests.performance.timing_recorder import TimingRecorder


PERF_BASE_URL = os.environ.get(
    "JARVIS_PERF_BASE_URL",
    os.environ.get("JARVIS_PERF_OLLAMA_URL", "http://localhost:11434"),
)
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


PROVIDER_STATUS = probe_provider(PERF_BASE_URL, PERF_MODEL)


pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(
        not PROVIDER_STATUS.available,
        reason=f"Model unavailable: {PROVIDER_STATUS.reason} at {PERF_BASE_URL}",
    ),
]


def _make_cfg():
    from evals.helpers import MockConfig
    cfg = MockConfig()
    cfg.ollama_base_url = (
        PERF_BASE_URL.rstrip("/")[:-3]
        if PROVIDER_STATUS.provider == "ollama" and PERF_BASE_URL.rstrip("/").endswith("/v1")
        else PERF_BASE_URL
    )
    cfg.ollama_chat_model = PERF_MODEL
    cfg.fast_model = PERF_MODEL
    cfg.llm_provider = PROVIDER_STATUS.provider or "ollama"
    if cfg.llm_provider == "openai_compatible":
        cfg.llm_base_url = PERF_BASE_URL if PERF_BASE_URL.rstrip("/").endswith("/v1") else f"{PERF_BASE_URL.rstrip('/')}/v1"
        cfg.llm_chat_model = PERF_MODEL
    # Let size-aware defaults kick in (evaluator + digests ON for small).
    cfg.evaluator_enabled = None
    cfg.memory_digest_enabled = None
    cfg.tool_result_digest_enabled = None
    # Force the LLM-based router so its timing shows up in the report.
    # MockConfig doesn't set this attribute, and the engine's default varies.
    cfg.tool_selection_strategy = "llm"
    return cfg


def _write_report(
    rec: TimingRecorder,
    name: str,
    *,
    end_to_end_sec: list[float] | None = None,
    initial_call_sec: float | None = None,
    unnecessary_tool_calls: int | None = None,
) -> Path:
    PERF_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = PERF_REPORT_DIR / f"{name}-{stamp}.json"
    payload = {
        "name": name,
        "timestamp": stamp,
        "model": PERF_MODEL,
        "provider": PROVIDER_STATUS.provider,
        "base_url": PERF_BASE_URL,
        "availability": "available",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "runs": PERF_RUNS,
        "warm_condition": "one unmeasured warm-up call; prior server state unknown",
        "initial_call_sec": initial_call_sec,
        "end_to_end_sec": latency_summary(end_to_end_sec or []),
        "first_useful_text_sec": None,
        "first_useful_spoken_sec": None,
        "unnecessary_tool_calls": unnecessary_tool_calls,
        "llm_contexts": rec.to_dict(),
        "raw": [
            {
                "context": c.context,
                "duration_sec": round(c.duration_sec, 4),
                "model": c.model,
                "prompt_chars": c.prompt_chars,
                "response_chars": c.response_chars,
                "provider": c.provider,
                "outcome": c.outcome,
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
    """Baseline: how long does a single tiny round-trip to the provider take?

    This is the floor for every context's per-call cost. If the floor moves,
    every context's total moves with it. Reported separately from the
    pipeline test so hardware drift is obvious in the numbers.
    """
    from jarvis.llm.factory import get_llm_backend

    backend = get_llm_backend(_make_cfg())
    initial_call = time.perf_counter()
    backend.direct(
        chat_model=PERF_MODEL,
        system_prompt=TINY_SYSTEM,
        user_content=TINY_USER,
        timeout_sec=30.0,
    )
    initial_call_sec = time.perf_counter() - initial_call
    with TimingRecorder() as rec:
        for _ in range(PERF_RUNS):
            backend.direct(
                chat_model=PERF_MODEL,
                system_prompt=TINY_SYSTEM,
                user_content=TINY_USER,
                timeout_sec=30.0,
            )

    rec.print_report(title=f"Micro-benchmark: tiny prompt × {PERF_RUNS} on {PERF_MODEL}")
    path = _write_report(rec, "micro", initial_call_sec=initial_call_sec)
    print(f"   📄 saved: {path}")

    assert len(rec.calls) == PERF_RUNS


# =============================================================================
# Full pipeline: run_reply_engine × N, per-context timings
# =============================================================================


@pytest.mark.performance
def test_pipeline_timings_by_context():
    """Run representative text requests and record observed timings."""
    from jarvis.memory.db import Database
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.reply.engine import run_reply_engine

    cfg = _make_cfg()

    from jarvis.reply import engine as reply_engine

    wall_times = []
    unnecessary_tools = 0
    current_query = ""
    run_tool = reply_engine.run_tool_with_retries

    def observed_tool(*args, **kwargs):
        nonlocal unnecessary_tools
        tool_name = kwargs.get("tool_name") or (args[2] if len(args) > 2 else "")
        if current_query == "hello" and tool_name != "stop":
            unnecessary_tools += 1
        return run_tool(*args, **kwargs)

    with TimingRecorder() as rec, patch.object(reply_engine, "run_tool_with_retries", observed_tool):
        for query in PIPELINE_QUERIES:
            current_query = query
            db = Database(":memory:", sqlite_vss_path=None)
            dlg = DialogueMemory(inactivity_timeout=300, max_interactions=20)
            try:
                for _ in range(PERF_RUNS):
                    start = time.perf_counter()
                    run_reply_engine(db, cfg, None, query, dlg)
                    wall_times.append(time.perf_counter() - start)
            finally:
                db.close()

    rec.print_report(title=f"Pipeline timings — {len(PIPELINE_QUERIES)} queries × {PERF_RUNS} runs on {PERF_MODEL}")
    path = _write_report(
        rec, "pipeline", end_to_end_sec=wall_times,
        unnecessary_tool_calls=unnecessary_tools,
    )
    print(f"   📄 saved: {path}")

    assert rec.calls, "no LLM calls recorded — pipeline did not invoke the LLM"
    assert len(wall_times) == len(PIPELINE_QUERIES) * PERF_RUNS

    # Surface unmapped callers so new contexts show up in review.
    other = [c for c in rec.calls if c.context.startswith("other:")]
    if other:
        unmapped = sorted({c.context for c in other})
        print(f"   ⚠️  unmapped callers (add to _CALLER_TO_CONTEXT): {unmapped}")
