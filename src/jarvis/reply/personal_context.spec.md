# Missing personal tool context

The reply layer owns personal-context resolution. Tools identify an unresolved
personal input with `ToolExecutionResult.missing_context`, naming the destination
argument. A successful tool call, an explicit argument, and a failure without a
missing-context request bypass resolution. Explicit false, zero and empty-list
arguments retain their meaning; absent/null/blank strings permit resolution.
Malformed argument shapes preserve the original tool error. Tools return raw data and do not search
personal memory themselves.

`ContextualToolRunner` wraps both planned direct execution and normal model tool
execution. An unresolved supported input triggers one local evidence extraction
and semantic verification
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
Dialogue is collected newest first so older messages cannot hide the latest
user location update.
A missing table or source is non-fatal. Graph update dates are storage metadata,
not observation times. Stored home records older than 180 days, future dates and
invalid dates are unusable. These bounds can leave relevant evidence outside the
retrieved set; absence of evidence leads to clarification.

The FAST-tier model extracts candidate value, source ID and relationship kind
from the JSON evidence records. It cites original sources rather than generating
supporting quotations. Deterministic checks require valid JSON, a known source,
a literal short place name in that source, an allowed source/kind combination
and admissible date.
Candidates with a relationship kind ineligible for their source are discarded
individually: a remembered task destination cannot supply a requested/current
location, or discard a separate admissible residence. Malformed extraction still
preserves clarification. Semantic verification checks all admissible candidates
against the original records, including active travel that prevents home defaults.

A CHAT-tier semantic verification pass compares every admissible candidate
against its original source record and the active dialogue. It distinguishes
actual user assertions from quotations requested for translation/explanation,
hypotheticals, third-party facts, visits, former homes, assistant guesses and
instructions embedded in data. Reporting that the user said they live somewhere
can establish a home; merely requesting translation of that sentence cannot.
Home defaults are ineligible when active dialogue indicates another current
location or an unknown destination while away.

Each candidate receives a boolean support verdict identified by its integer
index. The response must cover every candidate exactly once, without duplicates,
unknown indices or invalid types. Unsupported candidates are discarded. An
unavailable or malformed verification aborts resolution and preserves clarification.
Deterministic precedence and conflict checks apply to the supported candidates.
Both calls use temperature zero, a 1,024-token allowance including reasoning and
`llm_tools_timeout_sec` per request. There is at most one extraction/verification
pair per missing input per reply. Semantic attribution remains model-dependent
and is exercised by live multilingual and adversarial evals.

Retried tool output carries a location-basis note. Remembered residence is labelled
as a home default with source and storage date, never as detected physical presence.
The chat model receives this metadata alongside the raw weather result. The
unified system prompt requires replies to explicitly label remembered defaults
and permits clarification when resolution fails.
