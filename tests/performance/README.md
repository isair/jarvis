# Performance tests

Per-context timings for the reply pipeline. Excluded from the default pytest run
(see `pytest.ini`'s `addopts = -m "not performance"`).

## Running

```bash
pytest tests/performance/ -v -m performance -s
```

The `-s` flag lets the report table print to stdout. Tests auto-skip when Ollama
is unreachable, so the harness is safe to leave in the repo.

## Env vars

| Var | Default | Description |
|-----|---------|-------------|
| `JARVIS_PERF_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `JARVIS_PERF_MODEL` | `gemma4:e2b` | Model pulled in Ollama for the run |
| `JARVIS_PERF_RUNS` | `3` | Runs per query (bump for tighter p95) |
| `JARVIS_PERF_REPORT_DIR` | `tests/performance/reports/` | JSON report output |

`PERF_RUNS=3` is a fast-iteration default. For stable p95 numbers when
benchmarking a change, use `JARVIS_PERF_RUNS=10` or higher.

## What it measures

- **`test_micro_benchmark_tiny_prompt`** — one warmup + N tiny round-trips.
  Hardware baseline: the floor for every context's per-call cost.
- **`test_pipeline_timings_by_context`** — three representative queries × N runs
  of `run_reply_engine`, with per-context timings bucketed via stack-frame
  inspection in [`timing_recorder.py`](timing_recorder.py).

Shape invariants (not absolute numbers):
- Evaluator p50 ≤ main chat turn p50 × 1.5.
- Tool router p50 ≤ main chat turn p50 × 1.5.
- Enrichment extractor shares the router model chain.

Unmapped callers print as `other:<qualname>` — that's a signal to update the
`_CALLER_TO_CONTEXT` map in `timing_recorder.py` alongside `docs/llm_contexts.md`.

Reports are written to `reports/` and git-ignored.
