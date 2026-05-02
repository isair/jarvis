# Recall Gate

A deterministic, no-LLM heuristic that lets the reply engine skip diary, graph and memory-digest enrichment when the hot window already grounds the user's follow-up.

The gate is a cheap pre-flight check, not a routing decision. It either tells the engine "keep going as planned" (recall) or "the hot window has this covered, you can short-circuit enrichment" (skip).

## Scope

- File: `src/jarvis/memory/recall_gate.py`.
- Caller: `run_reply_engine` in `src/jarvis/reply/engine.py`, between the planner's `needs_memory` decision and the diary/graph search.
- Inputs: the redacted user query, the recent dialogue messages (already including tool-carryover rows from prior replies in the hot window).
- Output: `True` to recall, `False` to skip.

## When the gate runs

The gate runs only when:

1. The planner did **not** explicitly emit a `searchMemory` step. An explicit planner intent always wins; the gate does not second-guess it.
2. There is at least one recent message in the hot window.

When the planner returned an empty plan (fail-open), the gate is allowed to short-circuit. When the planner returned a concrete plan that doesn't include `searchMemory`, the engine is already skipping enrichment, so the gate is a no-op.

## Heuristic

The gate returns `False` (skip enrichment) only if both hold:

1. The hot window contains at least one tool-related message — i.e. an entry for which `is_tool_message()` returns true. This is the freshness signal: a tool was already invoked in this conversation, so grounded data is sitting in the messages array.
2. The query's content words have ≥ 50% overlap with the words in the hot-window transcript. Coverage is asymmetric (`|overlap| / |query_words|`), not Jaccard — long histories shouldn't penalise a short follow-up.

Anything else returns `True`. On any exception the gate fails open with `True`.

## Language-agnostic by construction

Per the project's no-hardcoded-language-patterns rule, content-word extraction uses `re.findall(r"\w{3,}", text, flags=re.UNICODE)`. The unicode flag makes `\w` match Cyrillic, CJK, Arabic, Hebrew, etc.

A small English stopword list (`is`, `the`, `what`, etc.) filters function words before scoring. Non-English queries simply skip stopword filtering — the worst case is a slightly more conservative (i.e. more recall-prone) decision, which is the safe direction for a fail-open gate. Adding language-specific stopword lists is out of scope; the heuristic is intentionally conservative and the cost of recalling unnecessarily is one extractor LLM call, not user-visible failure.

## Privacy

The overlap words can include user-supplied query terms. Before they reach `debug_log`, they are passed through `redact()` so emails, JWTs, and other structurally-detectable secrets in the query don't leak into logs. The gate does not store anything itself.

## Why not have the planner do this?

The planner is an LLM call and runs once per turn regardless. Adding "is the hot window enough?" to its prompt would make every planner call slower and more brittle. The gate is a 1 ms pure-Python pass that only fires after the planner has decided memory might be useful, so it's strictly additive and trivially removable.

## Failure mode

`should_recall()` returns `True` on every exception path. The gate cannot make a turn worse by failing — at most it stops being an optimisation.
