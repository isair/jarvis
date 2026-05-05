# Diary Summariser Specification

## Overview

The diary summariser (`conversation.py::generate_conversation_summary`) condenses raw conversation chunks into a daily `conversation_summaries` row. That row feeds every downstream memory consumer — direct diary retrieval for enrichment, vector search, FTS, and knowledge-graph extraction. A corrupted summary therefore poisons every consumer, often silently: downstream code has no way to tell that a summary misrepresents what actually happened.

The summariser prompt enforces a fixed set of hygiene rules. Each rule exists because a specific field incident produced corrupted diary entries that misled later sessions. Rules are cumulative — none supersedes another.

The summariser prompt is the only write-time defence. There is no post-process scrub — the prompt is single-source-of-truth, language-agnostic, and improves automatically as the underlying chat model improves. Historical entries written before the prompt was tightened can be cleaned via a user-triggered LLM rewrite (see [LLM Rewrite Sweep](#llm-rewrite-sweep)).

## Core Behaviour

- Input: recent conversation chunks (last 10) plus, if present, the previous summary for the same day.
- Output: a free-form summary (≤ 200 words) and 3–5 comma-separated topic keywords.
- Storage: one row per `(date_utc, source_app)` in `conversation_summaries`, upserted on each update.
- Embedding: the concatenation of summary + topics is embedded and stored for vector retrieval.
- LLM failure is non-fatal — the summariser returns `(None, None)` and the update is skipped entirely. Pending messages remain queued for the next cycle.

## Hygiene Rules

### 1. No deflection narration
The summariser must not record the assistant's own failures, uncertainty, or offers to search. Those events are transient. If preserved, they are retrieved by future sessions as "conversation history" and prime the model to repeat the same deflection pattern.

- If the assistant eventually answered (e.g. after a tool call), record only the final answer.
- If the topic was raised but never resolved, record only the topic and the user's intent — strip every phrase describing the assistant's inability, uncertainty, or offer to help.

### 2. Attribution preservation
Claims the assistant made about third-party entities (films, books, products, people, places, scientific facts) must be attributed in the summary — "the assistant said X" rather than bare "X". The attribution lets downstream readers treat the claim with appropriate scepticism.

- Never paraphrase an attributed claim into an unattributed assertion. Unattributed claims poison enrichment by reading as established fact.
- If the user later corrects the assistant, record both the original claim and the correction. Do not silently replace.
- Tool-grounded data (weather, time, calculator results) and user-stated facts about the user themselves are safe without attribution caveats.

### 3. Topic separation
Unrelated topics must never be welded into one grammatical clause. No shared "and", shared appositive, or shared relative clause across distinct referents. Each topic gets its own sentence.

- A welded clause like "the film X and the character Y, identified as Z" is read by downstream retrievers as a single claim about both referents and silently corrupts future enrichment.
- A dangling appositive attaching to multiple antecedents is the exact failure mode — small models produce it frequently when two topics are raised in one conversation.

## Applicability

All three rules apply in any language, not only English. The prompt states this explicitly because small models otherwise assume the rule is keyed to the English phrases it names.

## LLM Rewrite Sweep

`rewrite_all_diary_summaries(db, ollama_base_url, ollama_chat_model, ...)` is a user-triggered bulk operation that walks every row in `conversation_summaries` and asks the chat model to remove deflection narration from each. It exists for cleaning **historical** poisoning from rows written before the summariser prompt was tightened. There is no equivalent on the write path — new writes rely on the prompt alone.

**Why an LLM rather than regex:** the leak shows up in any language the user speaks, in any phrasing the model invents. A regex set is English-first by definition (you can only enumerate the patterns you can think of) and grows into a whack-a-mole. A small instruction-following model handles the semantic check in one shot, in any language, and improves automatically as the user's chat model upgrades. Mirrors `optimise_diary_topics` in shape and privacy guarantees.

**Prompt contract (`_REWRITE_DEFLECTION_SYSTEM_PROMPT`):**
- Return the entry with EVERY sentence removed whose subject is the assistant and whose verb describes inability, deflection, hesitation, or non-knowledge.
- Keep every other sentence verbatim — no paraphrasing, reordering, translating, or "improving".
- Keep attributed assistant claims ("the assistant said Possessor is a 2020 film") — those carry information.
- Keep user-stated facts and tool-grounded data — those are not assistant failures.
- Output the cleaned text only. Empty string if the entire summary is deflection. Verbatim input if nothing needs removing.
- Applies in every language; do NOT translate the output.

**Untrusted-input fence:** the diary text is wrapped in `<<<BEGIN UNTRUSTED WEB EXTRACT>>>` / `<<<END UNTRUSTED WEB EXTRACT>>>` markers (the same fence used for web-search content) before being passed to the model, so a row containing what looks like instructions is treated as data, not as a directive to follow. The fence markers, if echoed back, are stripped from the response.

**Empty-rewrite guard:** if the model returns an empty string (a row that was *entirely* deflection), the original is kept and a `would_empty: true` flag is surfaced. An empty diary entry is worse than a slightly-leaky one — downstream retrieval treats absence as "no record" and the user loses the topic entirely.

**Privacy:** the sweep streams per-row events as `{date_utc, chars_before, chars_after, rewritten, would_empty, embedding_refreshed, error?}` — counts and booleans only, never raw summary text. The `error` value is the exception class name only (e.g. `"RuntimeError"`), never the stringified exception message, because Python exception messages can echo offending input back to the caller. The progress-event key set is locked behind a whitelist test so any future field addition forces deliberate review (`tests/test_memory_viewer_diary_scrub_api.py::test_progress_event_keys_are_a_known_whitelist`). The diary clean button must not become a data-exfiltration channel through the streaming progress UI.

**Audit trail:** preserves each row's original `ts_utc` on rewrite. A maintenance pass that stomped `ts_utc` would make every cleaned row look as though it had been written today, destroying the only signal users have to verify when each diary entry was actually authored.

**Vector embedding:** when a row is rewritten, the embedding stored alongside the summary is regenerated inline from the cleaned text if the caller passes both an `ollama_base_url` and an `ollama_embed_model`. Without an embed model the rewrite still happens (FTS stays consistent via SQLite triggers); the vector embedding stays anchored to the pre-rewrite text until the next user-driven write to that date. Per-row embedding refresh is best-effort: an embedding-service failure is logged but does not roll back the summary write.

**Fail-open at every layer:**
- LLM call failure on a row → row is left untouched and reported with `error` set to the exception class name.
- Empty rewrite → row is left untouched, `would_empty: true` surfaced.
- Per-row write failure → row is reported with `error`, the sweep continues.

**Cache invariant:** diary content is never cached across turns. The reply engine's hot cache holds the warm-profile block (graph-derived, not diary), the per-query router decision, and the per-query memory-extractor parameters. None are derived from diary text, so the rewrite sweep does not need a listener-style invalidation hook. The actual diary search hits SQLite live on every enrichment-bearing turn. Concurrency between the sweep and an in-flight reply is handled by SQLite WAL. There is one inherent limitation: the previous turn's already-spoken assistant reply lives in `DialogueMemory._messages`. If a follow-up lands on the recall-gate fast path, the user is answered from rolling dialogue rather than a fresh enrichment. The rewrite does not retroactively rewrite spoken history; the next turn that triggers fresh enrichment sees the cleaned diary.

**Read paths:** none. The rewrite only touches the bulk sweep. Read-time diary retrieval is untouched.

## Bulk Sweep UI

The memory viewer's diary tab carries a Maintenance section in the sidebar with two operations:

**"🧹 Clean up deflection narration"** — asks the chat model to rewrite each old diary entry, removing only sentences that narrate assistant failures. The rest of each entry is preserved verbatim, no diary entries are deleted, and a summary that is *entirely* deflection narration is kept rather than emptied. Requires the chat model to be running. Backed by `POST /api/diary/scrub-deflections` (NDJSON-streaming) which calls `rewrite_all_diary_summaries`. The endpoint URL still says "scrub" for backwards compatibility; the implementation is now LLM-driven.

**"🏷️ Optimise tags"** — normalises topic tags across all diary entries using the configured chat model. Because each diary write generates topics independently, the same concept may accumulate multiple surface forms over time ("cook", "cooking", "meal prep"). The optimiser collects all unique tags, makes a single LLM call to propose a normalised taxonomy (merging synonyms, splitting compound tags), then applies the mapping to every row whose tags change. Backed by `POST /api/diary/optimise-topics` (NDJSON-streaming) which calls `optimise_diary_topics`. Requires the chat model to be running. Diary text is untouched; only the `topics` column is rewritten. Preserves `ts_utc` on every rewrite. Re-embeds updated rows best-effort. Fail-open: LLM failure or bad JSON leaves all rows unchanged.

## Tag Optimisation

`optimise_diary_topics(db, ollama_base_url, ollama_chat_model, ...)` in `conversation.py` implements the bulk tag normalisation sweep:

1. Collect all unique topic strings from every `conversation_summaries` row (one pass, in memory).
2. One `call_llm_direct` call to `ollama_chat_model` with `_TOPIC_OPTIMISE_SYSTEM_PROMPT` — returns a JSON object mapping each input tag to its normalised form (string for merge, list for split).
3. Apply the mapping via `_apply_topic_mapping()` to each row's comma-separated topics string. Deduplicates the result while preserving order so a merge that produces two identical tags (e.g. "cook, cooking" → "cooking, cooking") collapses cleanly.
4. Write back only rows whose topics changed, preserving `ts_utc` (same contract as the deflection rewrite).
5. Re-embed updated rows if an embed model is configured.

Yields one event per row: `{date_utc, topics_changed, old_topic_count, new_topic_count, error?}`. No raw tag values in events — counts only.

Idempotent once the mapping has been applied: a second run finds no tags to change.

## Evals and Regression Guards

| Test | Location | Guards |
|------|----------|--------|
| `test_omits_deflection_narration_for_unknown_entity` | `evals/test_diary_summariser_hygiene.py` | Rule 1, resolved case |
| `test_omits_deflection_when_topic_never_resolved` | `evals/test_diary_summariser_hygiene.py` | Rule 1, unresolved case |
| `test_unrelated_topics_are_not_welded_into_one_clause` | `evals/test_diary_summariser_hygiene.py` | Rule 3 |
| `test_preserves_legitimate_user_preferences` | `evals/test_diary_summariser_hygiene.py` | Cross-rule: hygiene must not strip real content |
| `TestSummariserForbidsDeflectionNarration` | `tests/test_diary_poisoning_defence.py` | Prompt-content regression (rules 1–3) |
| `TestRewriteSweepBehaviour` | `tests/test_diary_rewrite_sweep.py` | LLM-rewrite bulk sweep DB integration, fail-open, audit trail |
| `TestDiaryScrubEndpoint` | `tests/test_memory_viewer_diary_scrub_api.py` | Endpoint streaming + privacy contract |
| `TestOptimiseContract` / `TestOptimiseMerge` / `TestOptimiseSplit` / `TestOptimiseDeduplicate` / `TestOptimiseAuditTrail` / `TestOptimiseFailOpen` / `TestOptimiseIdempotence` | `tests/test_diary_topic_optimise.py` | Tag optimisation — generator contract, merge/split semantics, dedup, audit trail, fail-open, idempotence |

Live evals target the smallest supported model (gemma4:e2b) and `xfail` softly on weaker models rather than hard-failing, documenting residual risk instead of masking it.

## Relationship to Other Systems

- **Diary retrieval** (`engine.py`): injects retrieved summaries under a "reference only" framing, not as authoritative instructions. This partially mitigates corrupted summaries, but the primary defence is the summariser itself — see `reply.spec.md`.
- **Knowledge graph** (`graph.spec.md`): ingests summaries via `update_graph_from_dialogue()`. Graph extraction inherits whatever corruption the summary contains; hygiene at the summariser is the only place to fix this at source.
