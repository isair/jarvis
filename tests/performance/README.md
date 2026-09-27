# Performance benchmarks

The harness measures backend LLM calls by context and end-to-end wall time for
representative text requests. It works with an already available local Ollama
or OpenAI-compatible model. These benchmarks are excluded from the default
test run.

```bash
JARVIS_PERF_BASE_URL=http://localhost:11434 \
JARVIS_PERF_MODEL=gemma4:e2b \
pytest tests/performance/ -v -m performance -s
```

For an OpenAI-compatible server such as oMLX, set `JARVIS_PERF_BASE_URL` to
its local endpoint and `JARVIS_PERF_MODEL` to a model returned by `/v1/models`.
The base URL may include `/v1`. The harness probes the exact model before
running and reports unavailable models as skipped. It never downloads models.

| Variable | Default | Meaning |
|---|---|---|
| `JARVIS_PERF_BASE_URL` | `http://localhost:11434` | Local provider URL |
| `JARVIS_PERF_MODEL` | `gemma4:e2b` | Exact installed model name |
| `JARVIS_PERF_RUNS` | `3` | Measured repetitions per request |
| `JARVIS_PERF_PREPARATION` | `staged` | `staged` or experimental `combined` preparation |
| `JARVIS_PERF_REPORT_DIR` | `tests/performance/reports/` | JSON output directory |

The micro benchmark records one unmeasured warm-up call and the measured
round trips separately. The pipeline benchmark reports p50 and p95 for each
observed LLM context and for full request wall time. A p95 from three runs is
only a smoke check, so use at least ten runs for comparisons. Reports include
the provider, model, endpoint, commit, raw samples, and any unmapped context.

Compare preparation modes in separate, sequential runs with the same model and
run count. Each request has a fresh temporary database and dialogue. Tool replies
are deterministic local fixtures, including the normal stop signal; no external
tool runs. One measured tiny warm-up precedes each pipeline run. Earlier server
cache state is unknown. Timings do not establish answer accuracy.
Reports retain failed and empty model responses as such. The pipeline smoke
fails if any call produces neither text nor tool calls; fast failure is not a
successful latency result.

The harness does not observe streaming text or audio playback. First useful
text and first useful spoken output are `null` in its JSON reports. An
unnecessary tool count is measured for the greeting request, which needs no
tool. Accuracy evals and offline mechanism tests are reported separately in
`EVALS.md` and `EVALS_SCENARIOS*.json`.
