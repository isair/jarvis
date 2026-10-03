# Query and transcript comparison

## Scope

`query_context_comparison.py`, `query_context_cases.py` and
`run_query_context_comparison.py` provide an evaluation-only replay of the
production intent judge, tool router, planner and reply engine. They do not
change listener behaviour, application settings or runtime prompts.

## Arms and context

- `rewrite`: the full-context intent judge produces a self-contained query.
  The downstream engine receives that query and the fixture's shared dialogue.
- `raw_only`: the engine receives the current utterance unchanged and the same
  dialogue. This diagnostic control omits ambient transcript context.
- `raw_context`: the engine receives the current utterance unchanged. Every
  model request receives a separate fenced reference block containing the
  judge's actual transcript format, current-segment and timestamp markers,
  wake context, hot-window state and TTS echo information.

The context block treats earlier speech as data. It gives the current query
priority, excludes unrelated instructions and TTS echo, and JSON-quotes the
transcript with escaped angle brackets so speech cannot close the fence.
This prompt is a candidate evaluated by the replay, not a runtime contract.

## Isolation and measurement

All fixtures are synthetic and known to be directed. Candidate arms assume
successful classification, so this comparison does not qualify a classifier
or measure wake, stop or rejection accuracy. Classification has its own suite.

Only loopback OpenAI-compatible endpoints are accepted. Embedded credentials,
query strings and fragments are rejected before configuration or reports are
written. Requests ignore environment proxies and cannot follow redirects.
API keys remain in memory; request headers and configuration objects are
excluded from reports.

Each trial creates an empty in-memory database and fresh dialogue memory.
Empty diary retrieval is stubbed; tools return fixture data only when their
names and arguments match. No real tool action, web search or persistent
memory write occurs. The real router, planner and reply engine still run.

Temperature is zero for all arms. The same reasoning-template setting applies
to every request, and production output caps are retained. Explicit timeout
overrides are recorded. Arm order rotates across cases and repeats to reduce
ordering bias. Server caches are not reset and cached token counts are traced.

## Scoring

Reports separate initial routing, tool arguments, grounded answer facts,
unexpected tools, request validity and end-to-end completion. A completed
trial requires correct arguments and answer, no unexpected tool or replay
error, and valid model responses throughout. A later fallback cannot hide
an earlier empty, truncated, malformed or failed model request.

Argument checks use the registered tool's public schema: required arguments
must be present, string properties must contain strings, and only recognised
property values can establish the referent. Unused fields cannot supply a
matching entity. The fixtures use built-in tools with string properties.

All trials stay in the accuracy denominator. Initial router mistakes are
reported separately because later planning can recover them. Latency includes
the rewrite pass where applicable, and both all-trial and successful-trial
medians are reported. Missing token usage is unknown; token means use only
trials with complete usage, with that count reported.

The fixed facts are lightweight behavioural checks, not an LLM answer judge.
Tool results and speech are controlled fixtures rather than production data.
Repeated trials measure consistency on a small case set, not independent
samples or a general accuracy guarantee.
