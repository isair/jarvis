# Performance tests

The manual Ollama harness measures model round trips and per-context reply
pipeline calls. It is excluded from the default pytest run.

```bash
pytest tests/performance/ -v -m performance -s
```

Availability is checked only when selected benchmarks run. Importing or
collecting the harness does not contact the model server. Benchmarks skip when
the configured model is unavailable. Set these variables before running it:

| Variable | Default | Purpose |
|----------|---------|---------|
| `JARVIS_PERF_OLLAMA_URL` | `http://localhost:11434` | Local Ollama endpoint |
| `JARVIS_PERF_MODEL` | `gemma4:e2b` | Installed model |
| `JARVIS_PERF_RUNS` | `3` | Measured repetitions per query |
| `JARVIS_PERF_REPORT_DIR` | `tests/performance/reports/` | JSON output directory |

The micro-benchmark includes a warmup call. Pipeline runs use three small queries.
Three repetitions provide smoke evidence; sample p50/p95 values do not establish
stable tail latency. More repetitions help assess variability.

`TimingRecorder` measures Ollama and OpenAI-compatible backend methods, including
existing instances, factory-created instances and function-style helpers. It
counts a nested backend implementation once per outer operation, while retaining
separate concurrent requests. Reports keep empty results and exceptions alongside
returned text or tool-call responses. A `response` outcome describes transport
content, not a correct answer or successful tool execution.

Durations cover each complete backend operation. Their sum can include overlapping
requests and is not pipeline wall time, time to first token or time to first audio.
Reports contain context, model, provider, prompt/response sizes and outcome counts.
They do not contain prompt or response text.

Benchmarks fail for empty model samples, raised requests or empty pipeline
answers. JSON reports retain observed calls and their outcomes on failure;
exceptions keep their original type. Reports mark completed/failed runs and
record the exception class without its potentially private message. A report-write
failure preserves an existing request exception and is diagnosed by class only;
it fails a completed run. Call counts and timing-shape checks still apply to
usable runs.

Unknown callers appear as `other:<qualname>`. Keep `_CALLER_TO_CONTEXT` in
`timing_recorder.py` aligned with `docs/llm_contexts.md` when contexts change.
Reports are git-ignored.
