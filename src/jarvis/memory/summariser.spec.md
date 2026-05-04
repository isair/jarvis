# Diary Summariser Specification

## Overview

The diary summariser (`conversation.py::generate_conversation_summary`) condenses raw conversation chunks into a daily `conversation_summaries` row. That row feeds every downstream memory consumer — direct diary retrieval for enrichment, vector search, FTS, and knowledge-graph extraction. A corrupted summary therefore poisons every consumer, often silently: downstream code has no way to tell that a summary misrepresents what actually happened.

The summariser prompt enforces a fixed set of hygiene rules. Each rule exists because a specific field incident produced corrupted diary entries that misled later sessions. Rules are cumulative — none supersedes another.

The prompt is the first layer. A second deterministic layer (`scrub_deflection_sentences`) runs on every diary write and is also exposed as a one-shot bulk sweep for cleaning historical poisoning. This mirrors the two-layer defence the knowledge graph already has (extractor BANNED FACT FORMS at write-time, deterministic merge-time rewrite for historical data — see [`graph.spec.md`](graph.spec.md)).

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

## Post-Process Scrub (Deterministic Safety Net)

`scrub_deflection_sentences(text)` runs every diary write and is also exposed as the bulk sweep `scrub_all_diary_summaries(db)` for cleaning historical rows.

**Why a second layer:** field measurement on the smallest supported model (gemma4:e2b) showed roughly 40% of post-rule writes still contained banned phrasing despite rule 6 of the prompt. The prompt reduces the leak; it does not eliminate it on small models. A deterministic pass catches what slips through.

**What it does:**
- Splits the summary into sentences and drops any sentence whose content matches `DEFLECTION_PATTERNS` — narrow regexes covering "the assistant `<failure verb>`" shapes:
  - Direct: `was unable`, `could not`/`couldn't`, `did not have`/`didn't have`, `does not have`/`doesn't have`, `did not know`/`didn't know`, `does not know`/`doesn't know`, `was/is not able`/`wasn't/isn't able`, `failed to`, `offered to search/help/look`, `suggested checking`, `recommended consulting`.
  - Capability denial: `lacks/cannot/can't access`, `lacks the ability/information/details`.
  - Reporting verb + `it` + denial: `clarified/explained/stated/indicated [that] it could not …` (canonical), plus `said/noted/acknowledged/admitted/apologised/reported [that] it <denial>`. The required `it` between the reporting verb and the denial prevents stripping sentences whose subject is the user or a third-party entity (e.g. "The assistant said the film is from 2020" stays).
- Drops the **whole sentence** containing a match, never just the phrase. Half-sentences corrupt the record worse than the original leak.
- Returns the input unchanged if scrubbing would empty the summary outright. An empty diary entry is worse than a slightly-leaky one — downstream retrieval treats absence as "no record" and the user loses the topic of the conversation entirely. The "would have removed" count is still surfaced so callers can log the near-miss.
- Idempotent — running twice produces the same output, so the bulk sweep is safe to re-run.

**Language scope:** the regex set is English-first because every poisoned row in the field sample was English. The prompt rule itself is multilingual, so the LLM-side defence still covers non-English writes; the regex is the deterministic safety net for the dominant case.

**Privacy:** the bulk sweep streams per-row events as `{date_utc, sentences_removed, chars_before, chars_after, kept_original, embedding_refreshed, error?}` — counts only, never raw summary text. The `error` value is the exception class name only (e.g. `"RuntimeError"`), never the stringified exception message, because Python exception messages can echo offending input back to the caller. The progress-event key set is locked behind a whitelist test so any future field addition forces deliberate review (`tests/test_memory_viewer_diary_scrub_api.py::test_progress_event_keys_are_a_known_whitelist`). The diary clean button must not become a data-exfiltration channel through the streaming progress UI.

**Audit trail:** the bulk sweep preserves each row's original `ts_utc` on rewrite. A maintenance pass that stomped `ts_utc` would make every cleaned row look as though it had been written today, destroying the only signal users have to verify when each diary entry was actually authored.

**Vector embedding:** when the bulk sweep rewrites a row, the embedding stored alongside the summary is regenerated inline from the cleaned text if the caller passes both an `ollama_base_url` and an `ollama_embed_model`. Without an embed model the rewrite still happens (FTS stays consistent via SQLite triggers); the vector embedding stays anchored to the pre-scrub text until the next user-driven write to that date. Per-row embedding refresh is best-effort: an embedding-service failure is logged but does not roll back the summary write.

**Cache invariant:** diary content is never cached across turns. The reply engine's hot cache (`DialogueMemory._hot_cache`) holds three kinds of value — the warm-profile block (graph-derived, not diary), the per-query router decision, and the per-query memory-extractor parameters (keywords, questions, time window). None are derived from diary text, so the scrub does not need a listener-style invalidation hook. The actual diary search (`search_conversation_memory_by_keywords` → `db.search_hybrid`) hits SQLite live on every enrichment-bearing turn. Concurrency between the sweep and an in-flight reply is handled by SQLite WAL: separate `Database` instances on different connections, but WAL serialises writes against reads at the file level. Counterpart to the graph's `invalidate_warm_profile` listener — the graph needs one because the warm profile is a *derived cached projection* of node data; the diary has no such projection, so no listener is needed. Future diary-content caches (if any) must follow this rule: either invalidate on scrub via a listener, or do not cache across turns. There is one inherent limitation independent of caching: the previous turn's already-spoken assistant reply lives in `DialogueMemory._messages`. If a follow-up lands on the recall-gate fast path (hot window covers the topic), the user is answered from rolling dialogue rather than a fresh enrichment. The scrub does not retroactively rewrite spoken history; the next turn that triggers fresh enrichment sees the cleaned diary.

**Read paths:** none. The scrub only touches writes (per-summary on `update_daily_conversation_summary`) and the bulk sweep (`scrub_all_diary_summaries`). Read-time diary retrieval is untouched — by design, retrieval of cleaned data needs no extra filtering.

## Bulk Sweep UI

The memory viewer's diary tab carries a Maintenance section in the sidebar with two operations:

**"🧹 Clean up deflection narration"** — drops sentences that narrate assistant failures from old diary entries. The rest of each entry is preserved, no diary entries are deleted, and a summary that is *entirely* deflection narration is kept rather than emptied. Backed by `POST /api/diary/scrub-deflections` (NDJSON-streaming) which calls `scrub_all_diary_summaries`.

**"🏷️ Optimise tags"** — normalises topic tags across all diary entries using the configured chat model. Because each diary write generates topics independently, the same concept may accumulate multiple surface forms over time ("cook", "cooking", "meal prep"). The optimiser collects all unique tags, makes a single LLM call to propose a normalised taxonomy (merging synonyms, splitting compound tags), then applies the mapping to every row whose tags change. Backed by `POST /api/diary/optimise-topics` (NDJSON-streaming) which calls `optimise_diary_topics`. Requires the chat model to be running. Diary text is untouched; only the `topics` column is rewritten. Preserves `ts_utc` on every rewrite. Re-embeds updated rows best-effort. Fail-open: LLM failure or bad JSON leaves all rows unchanged.

## Tag Optimisation

`optimise_diary_topics(db, ollama_base_url, ollama_chat_model, ...)` in `conversation.py` implements the bulk tag normalisation sweep:

1. Collect all unique topic strings from every `conversation_summaries` row (one pass, in memory).
2. One `call_llm_direct` call to `ollama_chat_model` with `_TOPIC_OPTIMISE_SYSTEM_PROMPT` — returns a JSON object mapping each input tag to its normalised form (string for merge, list for split).
3. Apply the mapping via `_apply_topic_mapping()` to each row's comma-separated topics string. Deduplicates the result while preserving order so a merge that produces two identical tags (e.g. "cook, cooking" → "cooking, cooking") collapses cleanly.
4. Write back only rows whose topics changed, preserving `ts_utc` (same contract as the deflection scrub).
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
| `test_scrub_catches_residual_leak_when_prompt_lets_through_deflection` | `evals/test_diary_summariser_hygiene.py` | Post-process scrub safety net |
| `TestSummariserForbidsDeflectionNarration` | `tests/test_diary_poisoning_defence.py` | Prompt-content regression (rules 1–3) |
| `TestScrubDropsBannedSentences` / `TestScrubPreservesLegitimateContent` / `TestScrubEdgeCases` | `tests/test_diary_deflection_scrub.py` | Scrub-function unit tests |
| `TestScrubSweepBehaviour` | `tests/test_diary_scrub_sweep.py` | Bulk-sweep DB integration |
| `TestDiaryScrubEndpoint` | `tests/test_memory_viewer_diary_scrub_api.py` | Endpoint streaming + privacy contract |
| `TestOptimiseContract` / `TestOptimiseMerge` / `TestOptimiseSplit` / `TestOptimiseDeduplicate` / `TestOptimiseAuditTrail` / `TestOptimiseFailOpen` / `TestOptimiseIdempotence` | `tests/test_diary_topic_optimise.py` | Tag optimisation — generator contract, merge/split semantics, dedup, audit trail, fail-open, idempotence |

Live evals target the smallest supported model (gemma4:e2b) and `xfail` softly on weaker models rather than hard-failing, documenting residual risk instead of masking it.

## Relationship to Other Systems

- **Diary retrieval** (`engine.py`): injects retrieved summaries under a "reference only" framing, not as authoritative instructions. This partially mitigates corrupted summaries, but the primary defence is the summariser itself — see `reply.spec.md`.
- **Knowledge graph** (`graph.spec.md`): ingests summaries via `update_graph_from_dialogue()`. Graph extraction inherits whatever corruption the summary contains; hygiene at the summariser is the only place to fix this at source.
