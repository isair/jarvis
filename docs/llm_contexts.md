# LLM Contexts Map

Every distinct LLM call in Jarvis, what feeds it, what consumes it, and how it is gated. This is the reference for optimising the app's main bottleneck (LLM latency). Keep it in sync with the code — see the note at the bottom.

---

## 1. Main Reply Loop (agentic messages loop)

- **File**: [src/jarvis/reply/engine.py](src/jarvis/reply/engine.py) — `reply()` and the loop at ~lines 1370-1650; native tool-call path in `chat_with_messages()` (~1424, 1455).
- **Trigger**: every user message. Runs up to `agentic_max_turns` (default 8) iterations per reply.
- **Model / gating**: `cfg.ollama_chat_model` (the big model). Not optional. No size branching on the loop itself — size branching affects the digests/evaluator around it.
- **Inputs**:
  - Redacted user query
  - Recent dialogue (last 5 minutes)
  - Unified system prompt from [src/jarvis/system_prompt.py](src/jarvis/system_prompt.py) + ASR note + tool-protocol guidance
  - Digested memory enrichment (optional, see #4)
  - Time + location context (re-injected each turn)
  - Tool schema: native via `generate_tools_json_schema()` ([src/jarvis/tools/registry.py](src/jarvis/tools/registry.py)) or text fallback via `_text_tool_call_guidance()` ([engine.py:68](src/jarvis/reply/engine.py:68))
  - Tool results from prior turns (raw or digested — see #5)
- **Output**: OpenAI-style `{content, tool_calls, thinking}`. Consumed by the tool orchestrator and TTS pipeline. Natural-language content is delivered immediately; no post-turn evaluator runs.
- **Limits**: `num_ctx: 8192` (explicit). Timeout `llm_chat_timeout_sec` (45s). Auto-fallback from native to text tool-calls on HTTP 400 (`ToolsNotSupportedError`), sticky for the session. Risk: `fetch_web_page` truncates at 50,000 chars (~37k tokens) — mitigated for SMALL models by tool-result digest (#5) which compresses the payload before it enters the messages history. LARGE models receive the raw payload and may silently see a truncated context.

## 2. Intent Judge

- **File**: [src/jarvis/listening/intent_judge.py](src/jarvis/listening/intent_judge.py) — `IntentJudge.evaluate()`.
- **Trigger**: on a speech segment *only if* there is an engagement signal (wake word detected, hot-window active, or TTS playing). Pure ambient speech skips it.
- **Model / gating**: `cfg.intent_judge_model` (default `gemma4:e2b`, ~2B). Falls back to text-based wake detection if Ollama is unavailable.
- **Inputs**:
  - Rolling transcript buffer (last 120s, with timestamps)
  - Wake-word timestamp (if any), normalised aliases
  - Last TTS text + finish time (echo rejection)
  - State flags (wake_word_mode, hot_window_mode, during_tts)
- **System prompt**: `SYSTEM_PROMPT_TEMPLATE` at [intent_judge.py:135](src/jarvis/listening/intent_judge.py:135). Teaches query extraction, echo detection, stop commands, pronoun/topic disambiguation, imperative re-addressing, declaratives to the wake word.
- **Output**: strict JSON `IntentJudgment{directed, query, stop, confidence, reasoning}` ([intent_judge.py:94](src/jarvis/listening/intent_judge.py:94)). Consumed by the listening state machine which dispatches to the reply engine.
- **Limits**: `intent_judge_timeout_sec` (15s). `num_ctx: 4096` (explicit — transcript buffer can reach 400+ tokens; Ollama's model default of 2048 would silently truncate it).

## 3. Memory Enrichment Extractor

- **File**: [src/jarvis/reply/enrichment.py](src/jarvis/reply/enrichment.py) — `extract_search_params_for_memory()` (~line 71).
- **Trigger**: once per reply before the loop.
- **Model / gating**: resolved via `resolve_tool_router_model(cfg)` — `tool_router_model → intent_judge_model → ollama_chat_model`. Small classification task; rides the same small/warm model as the router. Not optional; silent empty-dict on failure.
- **Inputs**: user query, optional context hint (live-context compact summary), UTC now.
- **System prompt**: inline at [enrichment.py:35-63](src/jarvis/reply/enrichment.py:35).
- **Output**: `{keywords, from?, to?, questions?}`. Consumed by memory search at ~engine.py:1359.
- **Limits**: up to 2 retries; timeout from `llm_tools_timeout_sec`.

## 4. Memory Digest (optional, SMALL models)

- **File**: [src/jarvis/reply/enrichment.py](src/jarvis/reply/enrichment.py) — `digest_memory_for_query()` + `_distil_batch()`.
- **Trigger**: once per reply when enrichment returns hits AND `memory_digest_enabled` (default OFF; `null` = auto-ON for SMALL ≤7B / OFF for LARGE). Skipped if raw < `_DIGEST_MIN_CHARS` (400). Batched if raw > `_DIGEST_BATCH_MAX_CHARS` (2000).
- **Model / gating**: `ollama_chat_model`. Gated by `memory_digest_enabled`.
- **Inputs**: user query, raw diary entries, raw graph nodes.
- **System prompt**: `_DIGEST_SYSTEM_PROMPT` at [enrichment.py:122](src/jarvis/reply/enrichment.py:122). Teaches relevance filtering, preference-signal detection, attribution preservation, `NONE` sentinel, identity queries.
- **Output**: ≤400 chars text per batch (`_DIGEST_MAX_CHARS`) injected as reference-only memory context into the main loop's system message. Empty on failure.
- **Limits**: `llm_digest_timeout_sec` (8s, shared).

## 5. Tool-Result Digest (optional, opt-in)

- **File**: [src/jarvis/reply/enrichment.py](src/jarvis/reply/enrichment.py) — `digest_tool_result_for_query()` + `_distil_tool_batch()`.
- **Trigger**: after each tool result in the loop, if `tool_result_digest_enabled` (default `null` = auto-ON for SMALL ≤7B, OFF for LARGE). Primary motivation on small models: prevents `fetch_web_page`'s 50k-char payloads from filling the 8192 num_ctx window. Skipped if raw < 400 chars (`_TOOL_DIGEST_MIN_CHARS`); batched if > 2500 (`_TOOL_DIGEST_BATCH_MAX_CHARS`).
- **Model / gating**: `ollama_chat_model`. Gated by `tool_result_digest_enabled`.
- **Inputs**: user query, tool name, raw tool result (e.g. webSearch payload inside UNTRUSTED WEB EXTRACT fence).
- **System prompt**: `_TOOL_DIGEST_SYSTEM_PROMPT`. Teaches attributed fact extraction, `NONE` sentinel, no inference.
- **Output**: ≤600 chars per batch (`_TOOL_DIGEST_MAX_CHARS`) replacing the raw payload in the messages stream. Falls back to raw on `NONE`.
- **Limits**: `llm_digest_timeout_sec` (8s, shared).

## 6. Max-Turn Loop Digest

- **File**: [src/jarvis/reply/enrichment.py](src/jarvis/reply/enrichment.py) — `digest_loop_for_max_turns()` (~line 847).
- **Trigger**: when the loop exhausts `agentic_max_turns` without producing a natural-language reply (e.g. pure tool-call loop). The evaluator no longer drives this — termination on content is immediate.
- **Model / gating**: `_resolve_loop_digest_model(cfg)` — prefers `intent_judge_model`, falls back to `ollama_chat_model`.
- **Inputs**: user query + loop activity (tool calls, results summaries, any prose).
- **System prompt**: `_LOOP_DIGEST_SYSTEM_PROMPT` — caveat-prefixed, user-language, concise.
- **Output**: caveat-prefixed final reply. Fails open to the last raw candidate or generic error.
- **Limits**: `llm_digest_timeout_sec` (8s, shared).

## 7. Tool Router (pre-loop tool selection)

- **File**: [src/jarvis/tools/selection.py](src/jarvis/tools/selection.py) — `select_tools_with_llm()` (~line 331).
- **Trigger**: once per reply before the loop, if `tool_selection_strategy == "llm"` (default). Other strategies: `all`, `keyword`, `embedding`.
- **Model / gating**: `resolve_tool_router_model(cfg)` chain — `tool_router_model → intent_judge_model → ollama_chat_model`.
- **Inputs**: user query, tool catalogue (builtin + MCP with descriptions), optional narrow-down hint.
- **System prompt**: inline (~lines 260-315). Teaches pick up-to-5 tools or `none`.
- **Output**: comma-separated tool names or `none`. Capped at `_LLM_MAX_SELECTED` (5). Always-included tools (`stop`, `toolSearchTool`) are unioned in regardless.
- **Limits**: `llm_timeout_sec`. On failure → all tools.

## 8. Tool Searcher (mid-loop escape hatch)

- **File**: [src/jarvis/tools/builtin/tool_search.py](src/jarvis/tools/builtin/tool_search.py) — `toolSearchTool`.
- **Trigger**: when the model explicitly invokes `toolSearchTool` during the loop. Capped at `tool_search_max_calls` (3) per reply.
- **Model**: reuses the tool router (#7) — no separate LLM call here.
- **Inputs**: self-contained query from the model.
- **Output**: newline-separated tool names + one-liners, merged into the allow-list for the next turn.

## 9. Conversation Summariser

- **File**: [src/jarvis/memory/conversation.py](src/jarvis/memory/conversation.py) — `generate_conversation_summary()` (~lines 350/355).
- **Trigger**: background, periodic — when unsaved dialogue reaches `dialogue_memory_timeout`. One per day per `source_app`.
- **Model / gating**: `ollama_chat_model`. Respects `llm_thinking_enabled`. Uses streaming when a token callback is provided, else direct.
- **Inputs**: recent conversation chunks + prior same-day summary (for incremental update).
- **System prompt**: inline (~lines 310-320). Hygiene rules per [src/jarvis/memory/summariser.spec.md](src/jarvis/memory/summariser.spec.md): no deflection narration, attribution preservation, topic separation. ≤200 words + 3-5 topic keywords.
- **Output**: `(summary_text, topics_text)` → `conversation_summaries` table, embedded for vector search, feeds enrichment (#3) and graph extraction (#10).
- **Limits**: `timeout_sec` (30s default).

## 10. Knowledge Graph Fact Extraction

- **File**: [src/jarvis/memory/graph_ops.py](src/jarvis/memory/graph_ops.py) — `_llm_extract_facts()` (~line 98).
- **Trigger**: after each daily summary (#9). Background.
- **Model**: `ollama_chat_model`.
- **Inputs**: summary text + optional date.
- **Output**: JSON array of novel fact strings → memory graph nodes.
- **Limits**: `timeout_sec`. Failures → empty list.

## 11. Knowledge Graph Best-Child Picker

- **File**: [src/jarvis/memory/graph_ops.py](src/jarvis/memory/graph_ops.py) — `_llm_pick_best_child()` (~line 167).
- **Trigger**: during graph insertion, per fact, to place it under the best existing category. Background.
- **Model**: uses `picker_model` when passed through from `update_graph_from_dialogue` (daemon resolves it via `resolve_tool_router_model(cfg)` → small model when available). Falls back to `ollama_chat_model` when no small model is configured.
- **Inputs**: fact text + numbered list of candidate child nodes (name + description).
- **System prompt**: inline (~lines 156-161) — answer with number or `NONE`.
- **Output**: child node id or `None` (fact still inserted, just not under an optimal parent).

## 12. Task-list Planner (pre-loop decomposition)

- **File**: [src/jarvis/reply/planner.py](src/jarvis/reply/planner.py) — `plan_query()`.
- **Trigger**: once per reply, after tool selection and before the agentic loop. Skipped when `cfg.planner_enabled = False`, when the query is shorter than `MIN_QUERY_CHARS` (20), or when no model / base URL is available.
- **Model / gating**: resolution chain `planner_model → tool_router_model → intent_judge_model → ollama_chat_model`. Classification-shaped, rides the warm small model.
- **Inputs**: user query, dialogue context, memory context, selected tool names + one-line descriptions.
- **System prompt**: `_PROMPT_TEMPLATE` at [planner.py:73](src/jarvis/reply/planner.py:73). Teaches short imperative steps, angle-bracket entity placeholders, final synthesis step, same-language output, no numbering.
- **Output**: list of plan steps (max `MAX_STEPS` = 5). Consumed by the engine to build the `ACTION PLAN:` system-message block and drive the direct-exec loop for small models.
- **Limits**: `planner_timeout_sec` (6s). Fail-open → `[]`. Trivial single-reply plans are dropped.

## 13. Plan Step Resolver (per direct-exec turn, small models)

- **File**: [src/jarvis/reply/planner.py](src/jarvis/reply/planner.py) — `resolve_next_tool_call()`.
- **Trigger**: top of each agentic-loop iteration when `use_text_tools` is True AND the plan from #12 still has unexecuted tool steps. Runs instead of the chat model for that turn. **Fast path skips the LLM entirely** when the step is fully concrete (tool name + `key='value'` args, no `<placeholder>`); the LLM call only fires when entity substitution or key remapping is needed.
- **Model**: same chain as #12.
- **Inputs**: next planned step text, prior tool calls (name + args + result excerpt), per-turn tool schema.
- **System prompt**: `_STEP_RESOLVER_SYSTEM` at [planner.py:300](src/jarvis/reply/planner.py:300). Teaches one-JSON-object output, placeholder substitution from prior results, `null` for synthesis steps.
- **Output**: `(tool_name, arguments)` tuple or `None`. Unknown tool names are rejected via the allow-list guard.
- **Limits**: `planner_timeout_sec`. Fail-open → `None` (engine falls back to the chat-model turn).

## 14. Tool-specific LLM calls

- **Weather** ([src/jarvis/tools/builtin/weather.py](src/jarvis/tools/builtin/weather.py), ~line 60) — `ollama_chat_model`, parses location/time/unit from the query.
- **Nutrition log_meal** ([src/jarvis/tools/builtin/nutrition/log_meal.py](src/jarvis/tools/builtin/nutrition/log_meal.py), lines 48 & 136) — `ollama_chat_model`, extracts nutrients, confirms logging.

---

## Frequency / Size Summary

| # | Context | Per reply | Optional? | Model tier |
|---|---------|-----------|-----------|------------|
| 1 | Main chat loop | 1-8 | No | LARGE |
| 2 | Intent judge | 1 (voice only) | fallback available | SMALL |
| 3 | Memory enrichment extract | 1 | No | SMALL (via router chain) |
| 4 | Memory digest | 0-N | auto by size | SMALL (uses chat model) |
| 5 | Tool-result digest | 0-N | auto by size | SMALL (uses chat model) |
| 6 | Max-turn digest | 0-1 | No | SMALL |
| 7 | Tool router | 1 | yes (strategy) | SMALL |
| 8 | Tool searcher | 0-3 | model-initiated | SMALL (reuses #7) |
| 9 | Summariser | ~1/session | No (background) | LARGE |
| 10 | Graph extraction | ~1/session | No (background) | LARGE |
| 11 | Graph best-child | 0-N | No (background) | SMALL (via router chain) |
| 12 | Planner (plan_query) | 1 | yes (planner_enabled) | SMALL (via router chain) |
| 13 | Plan step resolver | 0-N (SMALL only) | auto by size + plan | SMALL (via router chain) |
| 14 | Tool-specific | per-tool | n/a | LARGE |

## Size-aware auto switches

Driven by `detect_model_size(model_name) → SMALL (≤7B) | LARGE (8B+)`:

| Feature | SMALL | LARGE |
|---------|-------|-------|
| Memory digest | ON | OFF |
| Tool-result digest | ON | OFF |
| Text-based tool calling | ON | OFF (native) |
| Planner direct-exec | ON | OFF |

## Config keys

- Models: `ollama_chat_model`, `intent_judge_model`, `tool_router_model`
- Flags: `memory_digest_enabled`, `tool_result_digest_enabled`, `llm_thinking_enabled`, `intent_judge_thinking_enabled`, `tool_selection_strategy`
- Timeouts: `llm_chat_timeout_sec` (45s), `llm_digest_timeout_sec` (8s, shared across #4/#5/#6), `llm_tools_timeout_sec`, `intent_judge_timeout_sec` (15s)
- Caps: `agentic_max_turns` (8), `tool_search_max_calls` (3), `_LLM_MAX_SELECTED` (5), `_DIGEST_MAX_CHARS` (400), `_TOOL_DIGEST_MAX_CHARS` (600)

## Flow

```
user input
  └─▶ [2] Intent Judge            (voice only, SMALL)
        └─▶ [3] Enrichment extract
              └─▶ [4] Memory digest  (optional, SMALL path)
                    └─▶ [7] Tool router
                          └─▶ [12] Planner (pre-loop)
                                └─▶ AGENTIC LOOP  (≤ agentic_max_turns)
                                      ├─ [13] Plan step resolver (SMALL, direct-exec)
                                      ├─ [1] Main chat turn
                                      ├─ tool execution
                                      │    └─ [5] Tool-result digest (optional)
                                      │    └─ [8] Tool searcher (model-initiated)
                                      └─ content → deliver immediately
                                      └─ if max turns → [6] Max-turn digest
                          └─▶ TTS / output
                          └─▶ background: [9] summariser → [10] graph extract → [11] best-child
```

## Optimisation ideas (seed list)

1. Batch multi-chunk memory digests (#4) into a single call with explicit markers.
2. Parallelise multiple tool-result digests (#5) when several results land at once.
3. Pre-warm the intent-judge model before TTS finishes.
4. Cache tool-router (#7) output by query hash.
5. Give each digest its own timeout budget rather than sharing `llm_digest_timeout_sec` (today a slow memory digest can starve the max-turn digest).
6. Consider single-model deployments: router+planner prefer `intent_judge_model`; loading a second model hurts cold-start latency on small hardware.
7. Narrow `llm_thinking_enabled` to router/planner only, not every context.
8. Reduce `intent_judge_timeout_sec` (15s) or race it against text-based wake detection to avoid blocking the audio loop.

---

## Measuring

`tests/performance/test_pipeline_timings.py` times each context in this graph against a live Ollama. Run:

```
pytest tests/performance/ -v -m performance -s
```

It records per-context p50/p95 latencies using a monkey-patch recorder that infers the context from the caller's `__qualname__` (see `_CALLER_TO_CONTEXT` in `tests/performance/timing_recorder.py`). Dumps a JSON report to `tests/performance/reports/`. A micro-benchmark with a tiny fixed prompt runs alongside to give a per-call floor — if that floor moves, every context's total moves with it, so hardware/model drift is visible immediately.

Baseline on a local gemma4:e2b (as of 2026-04-22, 3 queries × 3 runs): main chat turn p50 ~4.5s, enrichment extract p50 ~0.9s (small-model chain), micro-prompt floor ~0.15s. Sample sizes: main 25 calls, enrichment 9. Use these as rough reference points — the assertions in the test are relative-shape (router ≤ 1.5× main chat turn), not absolute.

When you add or change a context, update `_CALLER_TO_CONTEXT` so it shows up in the report instead of landing in the `other:` bucket.

## Keep this doc in sync

This graph is the reference for LLM-latency optimisation. Treat it as authoritative: whenever code changes affect an LLM call — a new context, a removed one, a changed model/timeout/cap/gating/prompt source, or a new data-flow edge — update this file in the same PR. If the update would be more than a one-line tweak, reflect it in the relevant `*.spec.md` too.
