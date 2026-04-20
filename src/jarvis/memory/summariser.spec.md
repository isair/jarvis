# Diary Summariser Specification

## Overview

The diary summariser (`conversation.py::generate_conversation_summary`) condenses raw conversation chunks into a daily `conversation_summaries` row. That row feeds every downstream memory consumer — direct diary retrieval for enrichment, vector search, FTS, and knowledge-graph extraction. A corrupted summary therefore poisons every consumer, often silently: downstream code has no way to tell that a summary misrepresents what actually happened.

The summariser prompt enforces a fixed set of hygiene rules. Each rule exists because a specific field incident produced corrupted diary entries that misled later sessions. Rules are cumulative — none supersedes another.

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

## Evals and Regression Guards

| Test | Location | Guards |
|------|----------|--------|
| `test_omits_deflection_narration_for_unknown_entity` | `evals/test_diary_summariser_hygiene.py` | Rule 1, resolved case |
| `test_omits_deflection_when_topic_never_resolved` | `evals/test_diary_summariser_hygiene.py` | Rule 1, unresolved case |
| `test_unrelated_topics_are_not_welded_into_one_clause` | `evals/test_diary_summariser_hygiene.py` | Rule 3 |
| `test_preserves_legitimate_user_preferences` | `evals/test_diary_summariser_hygiene.py` | Cross-rule: hygiene must not strip real content |
| `TestSummariserForbidsDeflectionNarration` | `tests/test_diary_poisoning_defence.py` | Prompt-content regression (rules 1–3) |

Live evals target the smallest supported model (gemma4:e2b) and `xfail` softly on weaker models rather than hard-failing, documenting residual risk instead of masking it.

## Relationship to Other Systems

- **Diary retrieval** (`engine.py`): injects retrieved summaries under a "reference only" framing, not as authoritative instructions. This partially mitigates corrupted summaries, but the primary defence is the summariser itself — see `reply.spec.md`.
- **Knowledge graph** (`graph.spec.md`): ingests summaries via `update_graph_from_dialogue()`. Graph extraction inherits whatever corruption the summary contains; hygiene at the summariser is the only place to fix this at source.
