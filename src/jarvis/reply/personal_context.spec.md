# Missing personal tool context

The reply layer owns personal-context resolution. Tools identify an unresolved
personal input with `ToolExecutionResult.missing_context`, naming the destination
argument. A successful tool call, an explicit argument, and a failure without a
missing-context request bypass resolution. Tools return raw data and do not search
personal memory themselves.

`ContextualToolRunner` wraps both planned direct execution and normal model tool
execution. An unresolved supported input triggers one local evidence extraction
and at most one retry with the grounded argument. Successes and misses are cached
by input name for one reply only. Memory/backend failures preserve the tool's
clarification. Resolution does not write inferred facts or change configuration.
Only `location` has a resolution policy. Adding another personal input requires
its own evidence and freshness policy; the execution/retry protocol is shared.

## Location policy

Weather requests missing location after its ordinary explicit/detected/current-
utterance fallbacks can use active user dialogue or remembered home residence.
Current-query locations take precedence over active dialogue. Within dialogue,
the latest explicit user location takes precedence. An active residence correction
supersedes stored home defaults. Conflicting stored residences require clarification;
record write order does not establish which residence is correct. Being away without
a known city blocks the home default. Stored temporary locations cannot supply a
current location, even when written today.

Evidence sources are the current redacted query, the latest six dialogue messages
(user messages only, excluding tool results), the existing graph User subtree and
dated diary summaries. `memory_enrichment_source` controls diary/graph retrieval.
World and Directives branches are excluded. Retrieval uses local SQLite reads,
without embeddings, graph mutations or a planner searchMemory requirement.

Bounds: 8,000 evidence characters, 1,000 characters per record, 20 diary rows from
the latest 180 days and 32 graph nodes through depth eight. Query, dialogue, diary
and graph have reserved character budgets so one source cannot crowd out another.
A missing table or source is non-fatal. Graph update dates are storage metadata,
not observation times. Stored home records older than 180 days, future dates and
invalid dates are unusable. These bounds can leave relevant evidence outside the
retrieved set; absence of evidence leads to clarification.

The FAST-tier model extracts candidate value, source ID, relationship kind and
an exact supporting quote from fenced JSON evidence. Its generation budget is
1,024 tokens including reasoning, with `llm_tools_timeout_sec`. It must reject
former residences, trips, hypotheticals, third-party locations, assistant guesses
and instructions embedded in evidence. Deterministic checks require valid JSON,
a known source, an exact non-empty evidence span, a literal short place name,
an allowed source/kind combination and admissible date. They enforce precedence
and reject conflicting cities. Semantic attribution depends on model extraction
and is exercised by live multilingual and adversarial evals.

Retried tool output carries a location-basis note. Remembered residence is labelled
as a home default with source and storage date, never as detected physical presence.
The chat model receives this metadata alongside the raw weather result. The
unified system prompt requires replies to explicitly label remembered defaults
and permits clarification when resolution fails.
