# LLM Contexts Map

Every distinct LLM call in Jarvis, what feeds it, what consumes it, and how it is gated. This is the reference for optimising the app's main bottleneck (LLM latency). Keep it in sync with the code — see the note at the bottom.

> **Front ends.** Contexts 1 and 3-onwards are entered from either front end: the voice listener ([src/jarvis/listening/listener.py](src/jarvis/listening/listener.py)) or the text chat ([src/jarvis/chat/cli.py](src/jarvis/chat/cli.py), [spec](../src/jarvis/chat/chat.spec.md)). Speech-specific contexts (the intent judge, dictation filler removal) exist only on the voice path.

> **Backend abstraction.** Every context below routes through `jarvis.llm` ([spec](../src/jarvis/llm/llm.spec.md)) via `get_llm_backend(cfg)` / `get_embedding_backend(cfg)`. Picking `llm_provider: openai_compatible` swaps the wire shape end-to-end without touching call sites. The active chat model is read directly from `cfg.llm_chat_model` (the `Settings` field that always carries the resolved value, populated by config-load from `ollama_chat_model` when the provider-aware key is left empty).

---

## 1. Main Reply Loop (agentic messages loop)

- **File**: [src/jarvis/reply/engine.py](src/jarvis/reply/engine.py) — `reply()` and the loop at ~lines 1370-1650; native tool-call path in `chat_with_messages()` (~1424, 1455).
- **Trigger**: every user message. Runs up to `agentic_max_turns` (default 8) iterations per reply.
- **Model / gating**: `cfg.llm_chat_model` via `get_llm_backend(cfg)`. Not optional. No size branching on the loop itself — size branching affects the digests/evaluator around it.
- **Inputs**:
  - Redacted user query
  - Recent dialogue (last 5 minutes), including in-loop tool-call + tool-role messages from prior replies within the active conversation (tool carryover, `DialogueMemory.record_tool_turn` / `get_recent_turns_with_tools` in [src/jarvis/memory/conversation.py](src/jarvis/memory/conversation.py); per-prompt cap via `cfg.tool_carryover_max_turns` / `tool_carryover_per_entry_chars`; storage cap `_tool_turns_max_storage = 16`; cleared on `stop` signal AND on new-conversation entry; UNTRUSTED WEB EXTRACT fence markers preserved on truncation; both `content` and `tool_calls[*].function.arguments` scrubbed on write)
  - Unified system prompt from [src/jarvis/system_prompt.py](src/jarvis/system_prompt.py) + ASR note + tool-protocol guidance
  - **Warm profile block** (query-agnostic User + Directives excerpt from the knowledge graph, composed by `build_warm_profile()` / `format_warm_profile_block()` in [src/jarvis/memory/graph_ops.py](src/jarvis/memory/graph_ops.py) at Step 3.5 of `reply()`; no LLM call, pure SQLite read; injected unconditionally so personalisation is the default; result cached in `DialogueMemory._hot_cache` under `DialogueMemory.WARM_PROFILE_CACHE_KEY` for the lifetime of the active conversation. Invalidated on `stop`, on new-conversation entry, AND on User/Directives graph mutations via `install_warm_profile_invalidation()` in [src/jarvis/memory/graph_ops.py](src/jarvis/memory/graph_ops.py), wired at startup by both front ends ([src/jarvis/daemon.py](src/jarvis/daemon.py) and [src/jarvis/chat/cli.py](src/jarvis/chat/cli.py)) against `register_graph_mutation_listener` in [src/jarvis/memory/graph.py](src/jarvis/memory/graph.py); World-branch writes are ignored)
  - Digested memory enrichment (optional, see #4)
  - Time + location context (computed once per reply, placed at the END of the system message's dynamic region — never the head — so every in-loop call sends a byte-identical system message and the server's KV/prefix cache can reuse the whole prompt head; in text-tools mode it sits just before the tool-call syntax guidance so the instruction block stays final)
  - Tool schema: native via `generate_tools_json_schema()` ([src/jarvis/tools/registry.py](src/jarvis/tools/registry.py)) or text fallback via `_text_tool_call_guidance()` ([engine.py:68](src/jarvis/reply/engine.py:68))
  - Tool results from prior turns (raw or digested — see #5)
- **Output**: OpenAI-style `{content, tool_calls, thinking}`. Consumed by the tool orchestrator and TTS pipeline. Natural-language content is delivered immediately; no post-turn evaluator runs.
- **Limits**: `num_ctx: 8192` (explicit). Timeout `llm_chat_timeout_sec` (45s). Auto-fallback from native to text tool-calls on HTTP 400 (`ToolsNotSupportedError`), sticky for the session. Risk: `fetch_web_page` truncates at 50,000 chars (~37k tokens) — mitigated for SMALL models by tool-result digest (#5) which compresses the payload before it enters the messages history. LARGE models receive the raw payload and may silently see a truncated context.

## 2. Intent Judge

- **File**: [src/jarvis/listening/intent_judge.py](src/jarvis/listening/intent_judge.py) — `IntentJudge.evaluate()`.
- **Trigger**: on a speech segment *only if* there is an engagement signal (wake word detected, hot-window active, or TTS playing). Pure ambient speech skips it.
- **Model / gating**: FAST tier — `resolve_model(cfg, Tier.FAST)` via `get_llm_backend(cfg).chat(...)`. Provider-aware default at config load: `gemma4:e2b` (~2B) on the Ollama chat path; on an OpenAI-compatible chat provider an unset `fast_model` resolves to the active `llm_chat_model` (the Ollama pull-name does not exist on the user's server). An explicit `fast_model` in config.json wins on both paths. The backend re-raises `ConnectionError` so the judge can apply a 30s cooldown after the server actively refuses; falls back to text-based wake detection while the cooldown is active.
- **Inputs**:
  - Rolling transcript buffer (last 120s, with timestamps)
  - Wake-word timestamp (if any), normalised aliases
  - Last TTS text + finish time (echo rejection)
  - State flags (wake_word_mode, hot_window_mode, during_tts)
- **System prompt**: `SYSTEM_PROMPT_TEMPLATE` at [intent_judge.py:135](src/jarvis/listening/intent_judge.py:135). Teaches query extraction, echo detection, stop commands, pronoun/topic disambiguation, imperative re-addressing, declaratives to the wake word.
- **Output**: strict JSON `IntentJudgment{directed, query, stop, confidence, reasoning}` ([intent_judge.py:94](src/jarvis/listening/intent_judge.py:94)). Consumed by the listening state machine which dispatches to the reply engine. When `content` is empty **or truncated mid-JSON** (reasoning models count thinking tokens against the generation cap), the judge also recovers the JSON answer from `reasoning_content` — reasoning models typically end their thinking with the full structured answer.
- **Limits**: `intent_judge_timeout_sec` (6s). `num_ctx: 8192` (explicit). `max_tokens: 1500` (canonical cap — covers reasoning + answer on reasoning models; OpenAI-compatible backends get it at the payload root; Ollama maps it to `num_predict`).

## 3. Memory Enrichment Extractor

- **File**: [src/jarvis/reply/enrichment.py](src/jarvis/reply/enrichment.py) — `extract_search_params_for_memory()` (~line 71).
- **Trigger**: once per reply, **only when the pre-flight planner (#12) emitted a `searchMemory` directive or returned an empty plan (fail-open)**. Pure reply-only plans skip this entirely — saves one LLM call per greeting / small-talk turn.
- **Model / gating**: FAST tier — `resolve_model(cfg, Tier.FAST)`. Factory-dispatched. Small classification task; rides the same small/warm model as the router. Silent empty-dict on failure (early-return when no chat model is configured — no wasted LLM round-trip).
- **Inputs**: user query (with the planner's `topic` hint appended when present), optional context hint (live-context compact summary) or UTC-now anchor, both carried in the USER message.
- **System prompt**: inline at [enrichment.py:35-63](src/jarvis/reply/enrichment.py:35). Byte-static — no hint block, no timestamp — so the system prompt is identical across every extractor call and stays cacheable; the per-call hint / UTC anchor rides at the end of the user content.
- **Output**: `{keywords, from?, to?, questions?}`. Consumed by memory search in the reply engine.
- **Limits**: up to 2 retries; timeout from `llm_tools_timeout_sec`. `max_tokens: 50`.
- **Caching**: result cached in `DialogueMemory._hot_cache` under key `enrichment:{redacted_query[+topic_hint]}` for the lifetime of the active conversation. Identical follow-ups within the same conversation reuse the dict and skip the LLM hop. Cleared by `clear_hot_cache()` on the `stop` signal and on new-conversation entry.

## 3b. Recall Gate (pre-enrichment short-circuit)

- **File**: [src/jarvis/memory/recall_gate.py](src/jarvis/memory/recall_gate.py) — `should_recall()`.
- **Trigger**: once per reply, before diary/graph/digest enrichment runs (after the planner has decided memory is potentially needed).
- **Model / gating**: NO LLM — deterministic keyword-coverage heuristic. Cheap.
- **Inputs**: query, recent dialogue (incl. tool carryover rows).
- **Output**: `False` only if hot-window contains a fresh tool result AND ≥50% of the query's content words appear in the hot-window transcript → skips diary, graph, and memory digest for this reply. Else `True`. Fail-open on any exception. Content-word extraction uses `\w{3,}` with `re.UNICODE`, so the gate works for Latin, Cyrillic, CJK, Arabic, Hebrew, etc. (per CLAUDE.md "no hardcoded language patterns"). Overlap words are run through `redact()` before being written to debug logs.
- **Planner precedence**: when the planner explicitly emitted a `searchMemory` step, the gate is bypassed — the planner has more signal than coverage and overriding it would silently drop intent. The gate only short-circuits the fail-open empty-plan path.
- **Rationale**: prevents re-running diary/graph lookups when the hot window already grounds the follow-up (e.g. "his most famous song" after a Bieber webSearch).

## 4. Memory Digest (optional, SMALL models)

- **File**: [src/jarvis/reply/enrichment.py](src/jarvis/reply/enrichment.py) — `digest_memory_for_query()` + `_distil_batch()`.
- **Trigger**: once per reply when enrichment returns hits AND `memory_digest_enabled` (default OFF; `null` = auto-ON for SMALL ≤7.5B / OFF for LARGE). Skipped if raw < `_DIGEST_MIN_CHARS` (400). Batched if raw > `_DIGEST_BATCH_MAX_CHARS` (2000).
- **Model / gating**: `cfg.llm_chat_model` via `get_llm_backend(cfg)`. Gated by `memory_digest_enabled`; the auto-on path reads the same chat model so model-size detection follows the active provider.
- **Inputs**: user query, raw diary entries, raw graph nodes.
- **System prompt**: `_DIGEST_SYSTEM_PROMPT` at [enrichment.py:122](src/jarvis/reply/enrichment.py:122). Teaches relevance filtering, preference-signal detection, attribution preservation, `NONE` sentinel, identity queries.
- **Output**: ≤400 chars text per batch (`_DIGEST_MAX_CHARS`) injected as reference-only memory context into the main loop's system message. Empty on failure.
- **Limits**: `llm_digest_timeout_sec` (8s, shared). `max_tokens: 200`.

## 5. Tool-Result Digest (optional, opt-in)

- **File**: [src/jarvis/reply/enrichment.py](src/jarvis/reply/enrichment.py) — `digest_tool_result_for_query()` + `_distil_tool_batch()`.
- **Trigger**: after each tool result in the loop, if `tool_result_digest_enabled` (default `null` = auto-ON for SMALL ≤7.5B, OFF for LARGE). Primary motivation on small models: prevents `fetch_web_page`'s 50k-char payloads from filling the 8192 num_ctx window. Skipped if raw < 400 chars (`_TOOL_DIGEST_MIN_CHARS`); batched if > 2500 (`_TOOL_DIGEST_BATCH_MAX_CHARS`).
- **Model / gating**: `cfg.llm_chat_model` via `get_llm_backend(cfg)`. Gated by `tool_result_digest_enabled` — auto-on for SMALL via `detect_model_size(cfg.llm_chat_model)`.
- **Inputs**: user query, tool name, raw tool result (e.g. webSearch payload inside UNTRUSTED WEB EXTRACT fence).
- **System prompt**: `_TOOL_DIGEST_SYSTEM_PROMPT`. Teaches attributed fact extraction, `NONE` sentinel, no inference.
- **Output**: ≤600 chars per batch (`_TOOL_DIGEST_MAX_CHARS`) replacing the raw payload in the messages stream. Falls back to raw on `NONE`.
- **Limits**: `llm_digest_timeout_sec` (8s, shared). `max_tokens: 300`.

## 6. Max-Turn Loop Digest

- **File**: [src/jarvis/reply/enrichment.py](src/jarvis/reply/enrichment.py) — `digest_loop_for_max_turns()` (~line 847).
- **Trigger**: when the loop exhausts `agentic_max_turns` without producing a natural-language reply (e.g. pure tool-call loop). The evaluator no longer drives this — termination on content is immediate.
- **Model / gating**: FAST tier — `resolve_model(cfg, Tier.FAST)`. Factory-dispatched.
- **Inputs**: user query + loop activity (tool calls, results summaries, any prose).
- **System prompt**: `_LOOP_DIGEST_SYSTEM_PROMPT` — caveat-prefixed, user-language, concise.
- **Output**: caveat-prefixed final reply. Fails open to the last raw candidate or generic error.
- **Limits**: `llm_digest_timeout_sec` (8s, shared). `max_tokens: 200`.

## 7. Tool Router (pre-loop tool selection)

- **File**: [src/jarvis/tools/selection.py](src/jarvis/tools/selection.py) — `select_tools_with_llm()` (~line 331).
- **Trigger**: once per reply, **at the very front of the flow before the planner (#12)**. Always runs — the router is the authoritative tool picker, and its narrowed catalogue is what the planner sees. When the planner later references tools, those names are unioned into the router's allow-list but never replace it; small models tend to default to `webSearch` where a dedicated tool like `getWeather` should win, and the router is tuned for that classification. `tool_selection_strategy == "llm"` is the default; other strategies (`all`, `keyword`, `embedding`) also run here.
- **Model / gating**: FAST tier — `resolve_model(cfg, Tier.FAST)`. Factory-dispatched.
- **Inputs**: user query, tool catalogue (builtin + MCP with descriptions), optional narrow-down hint. User-prompt order is KV-cache-disciplined: the mostly-static catalogue opens, the dynamic hint (time + dialogue) follows, the query is the final token — consecutive router calls in one conversation share the full catalogue as prefix.
- **System prompt**: inline (~lines 260-315). Teaches pick up-to-5 tools or `none`.
- **Output**: comma-separated tool names or `none`. Capped at `_LLM_MAX_SELECTED` (5). Always-included tools (`stop`, `toolSearchTool`) are unioned in regardless.
- **Limits**: `llm_timeout_sec`. `max_tokens: 50`. On failure → all tools.
- **Caching**: `routed_tools` cached in `DialogueMemory._hot_cache` under key `router:{redacted_query}|{strategy}|{builtin-names}|{mcp-names}` for the lifetime of the active conversation. The catalogue signature lets a mid-conversation MCP refresh invalidate the cache; `context_hint` is intentionally excluded so time/location drift inside one conversation doesn't bust it. Cleared by `clear_hot_cache()` on the `stop` signal and on new-conversation entry.
- **Carry-over guard (engine-side overlay)**: after the cache lookup/write, the engine inspects the previous assistant turn's tool calls. When a previous tool reported `success=False` on its `ToolExecutionResult` (read via the `tool_failed` flag stamped onto each recorded tool result), that tool name is unioned back into the local `routed_tools` for this turn only. Compensates for small routers that misroute follow-ups where the user is supplying missing info (e.g. "I'm in London" routing to `webSearch` after a stalled `getWeather` chain). Successful chains do not carry over — a genuine new short ask after a completed chain keeps the router pick clean. The augmentation never touches the cache; replays of the same query in future turns get the raw router output. See `src/jarvis/reply/reply.spec.md` §6 (Tool allow-list per turn) for the full contract.

## 8. Tool Searcher (mid-loop escape hatch)

- **File**: [src/jarvis/tools/builtin/tool_search.py](src/jarvis/tools/builtin/tool_search.py) — `toolSearchTool`.
- **Trigger**: when the model explicitly invokes `toolSearchTool` during the loop. Capped at `tool_search_max_calls` (3) per reply.
- **Model**: reuses the tool router (#7) — no separate LLM call here.
- **Inputs**: self-contained query from the model.
- **Output**: newline-separated tool names + one-liners, merged into the allow-list for the next turn.

## 9. Conversation Summariser

- **File**: [src/jarvis/memory/conversation.py](src/jarvis/memory/conversation.py) — `generate_conversation_summary()` (~lines 350/355).
- **Trigger**: background, periodic — when unsaved dialogue reaches `dialogue_memory_timeout`. One per day per `source_app`.
- **Model / gating**: `cfg.llm_chat_model` via `get_llm_backend(cfg)`. Respects `llm_thinking_enabled`. Uses streaming when a token callback is provided, else direct.
- **Inputs**: recent conversation chunks + prior same-day summary (for incremental update).
- **System prompt**: inline (~lines 310-320). Hygiene rules per [src/jarvis/memory/summariser.spec.md](src/jarvis/memory/summariser.spec.md): no deflection narration, attribution preservation, topic separation. The deflection rule (rule 6) is enumerated with concrete BAD/GOOD pairs in English plus parallel pairs in Turkish and Spanish so small models don't assume the rule is keyed to English phrasing. ≤200 words + 3-5 topic keywords.
- **Output**: `(summary_text, topics_text)` → `conversation_summaries` table, embedded for vector search, feeds enrichment (#3) and graph extraction (#10). No post-process scrub — the prompt is single-source-of-truth, language-agnostic, and improves automatically as the chat model upgrades.
- **Deflection rewrite (separate bulk op)**: `rewrite_all_diary_summaries()` (`POST /api/diary/scrub-deflections`) — cleans historical rows. One `cfg.llm_chat_model` call per row with `_REWRITE_DEFLECTION_SYSTEM_PROMPT`, asking the model to drop sentences that narrate the assistant's own failures while keeping everything else verbatim. Diary text is fenced as untrusted data (same fence used by the web tool). Preserves `ts_utc`; re-embeds updated rows best-effort via `get_embedding_backend(cfg)`. Empty-rewrite guard keeps the original if the model would have emptied the row. Fail-open at every layer (LLM call, write-back, embed). User-triggered from the Maintenance section in the diary sidebar.
- **Topic optimisation (separate bulk op)**: `optimise_diary_topics()` (`POST /api/diary/optimise-topics`) — collects all unique tags from `conversation_summaries`, makes one `cfg.llm_chat_model` call with `_TOPIC_OPTIMISE_SYSTEM_PROMPT` to propose a normalised taxonomy (merge synonyms, split compound tags), then applies the mapping to every row that needs updating. Preserves `ts_utc`; re-embeds updated rows best-effort. User-triggered from the Maintenance section in the diary sidebar.
- **Limits**: `timeout_sec` (30s default). `max_tokens: 400` on the direct (non-streaming) path so a full 200-word summary + TOPICS line is never truncated; the streaming path is uncapped.

## 10. Knowledge Graph Fact Extraction + Branch Classification

- **File**: [src/jarvis/memory/graph_ops.py](src/jarvis/memory/graph_ops.py) — `extract_graph_memories()`.
- **Trigger**: after each daily summary (#9). Background.
- **Model**: `cfg.llm_chat_model` via `get_llm_backend(cfg)`.
- **Inputs**: summary text + optional date.
- **System prompt**: inline — asks for JSON array of `{"branch": "USER|DIRECTIVES|WORLD", "fact": "..."}` objects, with a heuristic ("user telling the assistant how to behave → DIRECTIVES; user telling the assistant about themselves → USER; external facts → WORLD"). Unknown branches default to USER. The DO-NOT-EXTRACT block hardens two recurring traps: assistant-generated recommendations (would-a-different-assistant-give-the-same-answer? heuristic separates these from external lookups, which DO count as facts) and transient snapshots like the current weather / time of day (described as "moments not facts" so the model stops conflating ephemera with persistent climate / location knowledge).
- **Output**: list of `(branch_id, fact_text)` tuples → routed into the tagged branch via branch-pinned descent (no cross-branch contamination).
- **Limits**: `timeout_sec`. Failures → empty list.

## 11. Knowledge Graph Best-Child Picker

- **File**: [src/jarvis/memory/graph_ops.py](src/jarvis/memory/graph_ops.py) — `_llm_pick_best_child()` (~line 167).
- **Trigger**: during graph insertion, per fact, to place it under the best existing category. Background.
- **Model**: `picker_model` when passed through from `update_graph_from_dialogue` (daemon resolves it via `resolve_model(cfg, Tier.FAST)` → small model when available); falls back to `cfg.llm_chat_model`. Factory-dispatched.
- **Inputs**: fact text + numbered list of candidate child nodes (name + description).
- **System prompt**: inline (~lines 156-161) — answer with number or `NONE`.
- **Output**: child node id or `None` (fact still inserted, just not under an optimal parent).

## 11b. Knowledge Graph Node Merge (rewrite-on-write consolidation)

- **File**: [src/jarvis/memory/graph_ops.py](src/jarvis/memory/graph_ops.py) — `merge_node_data()` (system prompt at `_MERGE_SYSTEM_PROMPT`).
- **Trigger**: **once per (node, flush)** during `update_graph_from_dialogue`. The orchestrator first applies the exact-match dedupe fast-path, then groups the remaining facts by their resolved `node_id` so a 5-fact flush hitting the User node fires one rewrite, not five. Cold-start writes (empty target node) skip straight to plain append. Also invoked with `new_facts=[]` by the `consolidate_all_populated_nodes` maintenance op (powering the memory viewer's 🧹 button) to re-apply current rules to historical data.
- **Model**: same `picker_model` as #11 (the fast tier when the caller resolves it, falling back to `cfg.llm_chat_model`). Factory-dispatched. Temperature 0 — the task is rule-following classification.
- **Inputs**: existing node `data` + the batch of new facts (zero or more) routed to that node in this flush.
- **System prompt**: defines an ordered rule set — contradiction/reversal drops the old version, near-duplicate phrasings collapse to one, repeated daily activities consolidate into patterns, independent attributes coexist (visible contradictions are NOT silently dropped), common-knowledge facts are pruned. Demands a bare `{"facts": [...]}` JSON object. Parser tries direct `json.loads` first, then a scoped regex (no greedy `\{.*\}`) before giving up.
- **Output**: `MergeResult(success: bool, incorporated_indices: list[int])`. The revised fact list is written back as the node's full `data`; `incorporated_indices` tells the orchestrator which inputs survived as new lines (under NFKC + casefold matching) so consolidated-out facts aren't reported as "newly stored". Subsumes per-flush supersession, near-duplicate dedupe, and ongoing consolidation in a single call. Because the latest prompt rewrites the whole node, updated conventions propagate to old data without a separate migration step.
- **Limits**: 20s timeout. **Hallucination guard**: rewrites with more than `len(existing) + len(new) + 2` lines are rejected as runaway output. Fail-open on any error, parse failure, oversized rewrite, or empty rewrite → caller falls back to plain `append_to_node` for each new fact so they still land (a contradiction is recoverable; a silent wipe or hallucinated bloat is not).

## 12. Task-list Planner (pre-flight decomposition, gates the whole turn)

- **File**: [src/jarvis/reply/planner.py](src/jarvis/reply/planner.py) — `plan_query()`.
- **Trigger**: once per reply, **after the tool router and before memory search**. Skipped when `cfg.planner_enabled = False`, when the query is shorter than `MIN_QUERY_CHARS` (4), when no model / base URL is available, or when the **engine-level fast-path skip** fires (the tool router returned no real tools AND the query is ≤ 8 words — the engine injects `["Reply to the user."]` as the plan without calling the LLM).
- **Model / gating**: CHAT tier — `resolve_model(cfg, Tier.CHAT)`. Factory-dispatched. The planner tracks the active chat model so upgrading it (via setup wizard, config, or provider switch) automatically upgrades plan quality.
- **Inputs**: user query, dialogue context, **router-narrowed** tool catalogue (names + one-line descriptions) — not the full 30+ list. When the carry-over guard from #7 fires, the previous turn's failed tool name is unioned into this catalogue before the planner sees it, so the planner can plan a re-call without `toolSearchTool` round-tripping. **No** memory context — the planner decides *whether* memory is needed.
- **System prompt**: `_PROMPT_TEMPLATE` in `planner.py`. Teaches the `searchMemory topic='...'` directive for prior-conversation lookups, short imperative tool steps, angle-bracket entity placeholders, final synthesis step, same-language output, no numbering.
- **Output**: list of plan steps (max `MAX_STEPS` = 5). Gates memory enrichment (#3 / #4) and augments the tool router (#7 — planner's picks are unioned in, not replacing). Single-step `["Reply to the user."]` plans are the planner's positive "no memory, no tools" signal. An empty list is fail-open — the engine reverts to running #3 unconditionally. A **stop-only plan** (every step is `stop`) is also rejected by a deterministic post-plan guard and returns `[]` — same fail-open path as an LLM failure — so the engine falls through to the tool router and chat model rather than silently dismissing the conversation. Consumed further by the engine to build the `ACTION PLAN:` system-message block and drive the direct-exec loop (#13) for small models.
- **Limits**: `planner_timeout_sec` (3s). `max_tokens: 150`. Fail-open → `[]`.

## 13. Plan Step Resolver (per direct-exec turn, small models)

- **File**: [src/jarvis/reply/planner.py](src/jarvis/reply/planner.py) — `resolve_next_tool_call()`.
- **Trigger**: top of each agentic-loop iteration when `use_text_tools` is True, the plan from #12 still has unexecuted tool steps, AND the plan is not under-specified (`plan_has_unresolved_tool_steps` returns False — steps that paraphrase tools without naming them skip direct-exec so the resolver doesn't guess arguments). Runs instead of the chat model for that turn. **Fast path skips the LLM entirely** when the step is fully concrete (tool name + `key='value'` args, no `<placeholder>`); the LLM call only fires when entity substitution or key remapping is needed.
- **Model**: same chain as #12.
- **Inputs**: next planned step text, prior tool calls (name + args + result excerpt), per-turn tool schema.
- **System prompt**: `_STEP_RESOLVER_SYSTEM` at [planner.py:300](src/jarvis/reply/planner.py:300). Teaches one-JSON-object output, placeholder substitution from prior results, `null` for synthesis steps.
- **Output**: `(tool_name, arguments)` tuple or `None`. Unknown tool names are rejected via the allow-list guard.
- **Limits**: `planner_timeout_sec` (3s). `max_tokens: 100`. Fail-open → `None` (engine falls back to the chat-model turn).

## 14. Tool-specific LLM calls

- **Weather** ([src/jarvis/tools/builtin/weather.py](src/jarvis/tools/builtin/weather.py), ~line 60) — factory-dispatched. Place extraction is a FAST-tier pass (`resolve_model(cfg, Tier.FAST)`) so small/warm models handle the parse without paging in the chat model. `max_tokens: 50`. Parses location/time/unit from the query.
- **Nutrition log_meal** ([src/jarvis/tools/builtin/nutrition/log_meal.py](src/jarvis/tools/builtin/nutrition/log_meal.py), lines 48 & 136) — factory-dispatched. Both the nutrition extractor and the follow-up generator use `cfg.llm_chat_model`. Extractor `max_tokens: 200`, follow-up `max_tokens: 100`. Extracts nutrients, confirms logging.

## 15. Server Capability Probe (setup-time, OpenAI-compatible only)

- **File**: [src/jarvis/llm/openai_compatible.py](src/jarvis/llm/openai_compatible.py) — `OpenAICompatibleBackend.check_capabilities()`. Called from the setup wizard's `_CapabilityWorker` ([src/desktop_app/setup_wizard.py](src/desktop_app/setup_wizard.py)).
- **Trigger**: not part of the runtime pipeline. Fires when the user clicks **Connect** on the OpenAI-compatible wizard page (once per connection attempt). The desktop startup reachability check (`_check_openai_compat_reachable` in [src/desktop_app/app.py](src/desktop_app/app.py)) uses only `list_models`, not this probe.
- **Model / gating**: the chat model the user selected on the page (and the selected embedding model, if any). Off the UI thread.
- **Inputs**: a fixed `"ping"` message; a trivial no-op tool schema; a `"ping"` embedding input. No user or memory data.
- **Output**: `ServerCapabilities{reachable, chat, tools, embeddings, models}`. Consumed only by the wizard to render an honest capability summary and offer the Ollama-embeddings fallback. Never persisted.
- **Limits**: `timeout_sec` default 8s per sub-request. Issues up to two `/chat/completions` calls (plain + tool), one `/embeddings`, one `/models`. Fail-soft: every error collapses to a `False` flag; a `ConnectionError` short-circuits to `reachable=False`.

---

## Frequency / Size Summary

| # | Context | Per reply | Optional? | Model tier |
|---|---------|-----------|-----------|------------|
| 1 | Main chat loop | 1-8 | No | LARGE |
| 2 | Intent judge | 1 (voice only) | fallback available | SMALL |
| 3 | Memory enrichment extract | 0-1 | gated by planner | SMALL (FAST tier) |
| 4 | Memory digest | 0-N | auto by size | SMALL (uses chat model) |
| 5 | Tool-result digest | 0-N | auto by size | SMALL (uses chat model) |
| 6 | Max-turn digest | 0-1 | No | SMALL |
| 7 | Tool router | 1 | always runs; planner picks unioned in | SMALL |
| 8 | Tool searcher | 0-3 | model-initiated | SMALL (reuses #7) |
| 9 | Summariser | ~1/session | No (background) | LARGE |
| 10 | Graph extraction | ~1/session | No (background) | LARGE |
| 11 | Graph best-child | 0-N | No (background) | SMALL (FAST tier) |
| 11b | Graph node merge | 0-N (per node, batched) | No (background) | SMALL (FAST tier) |
| 12 | Planner (plan_query) | 1 | yes (planner_enabled) | LARGE/SMALL (tracks chat model) |
| 13 | Plan step resolver | 0-N (SMALL only) | auto by size + plan | tracks chat model (CHAT tier; runs only when that model is SMALL) |
| 14 | Tool-specific | per-tool | n/a | LARGE |

## Size-aware auto switches

Driven by `detect_model_size(model_name) → SMALL (≤7.5B) | LARGE (>7.5B)` — uses a regex to extract the parameter count from the model name, handles MoE (`8x7b`) as LARGE, and defaults bare `gemma4` names (no size tag) to SMALL while sized variants (e.g. `gemma4:12b`) follow the threshold:

| Feature | SMALL | LARGE |
|---------|-------|-------|
| Memory digest | ON | OFF |
| Tool-result digest | ON | OFF |
| Text-based tool calling | ON | OFF (native) |
| Planner direct-exec | ON | OFF |

## Config keys

- Models: `llm_chat_model` (CHAT tier), `fast_model` (FAST tier). Every context resolves via `resolve_model(cfg, tier)`. Legacy on-disk keys (`ollama_chat_model` as a v1 → v2 alias; `intent_judge_model` / `tool_router_model` / `evaluator_model` / `planner_model` folded into `fast_model` by the v2 → v3 migration) are readable but no longer part of `Settings`.
- Flags: `memory_digest_enabled`, `tool_result_digest_enabled`, `llm_thinking_enabled`, `intent_judge_thinking_enabled`, `tool_selection_strategy`
- Timeouts: `llm_chat_timeout_sec` (45s), `llm_digest_timeout_sec` (8s, shared across #4/#5/#6), `llm_tools_timeout_sec`, `intent_judge_timeout_sec` (6s), `planner_timeout_sec` (3s)
- Caps: `agentic_max_turns` (8), `tool_search_max_calls` (3), `_LLM_MAX_SELECTED` (5), `_DIGEST_MAX_CHARS` (400), `_TOOL_DIGEST_MAX_CHARS` (600). Per-context `max_tokens` caps listed above (50–1500 depending on task — the intent judge's 1500 covers reasoning + answer on reasoning models; rewrite tasks scale with input length).

## KV-cache discipline (prompt construction rules)

Every context is built against servers (Ollama, vLLM, SGLang, llama.cpp `llama-server`, LM Studio) that reuse the KV state of the longest matching prompt prefix. The first diverging token decides how much compute is saved, so these rules are load-bearing:

1. **System prompts are byte-static** — no timestamps, hints, or per-call data inside. Per-call data (time, location, dialogue) lives in the user message.
2. **Dynamic blocks go to the tail** — anything that changes per call (context line, hint blocks) is appended at the END of its message, never at the head.
3. **Stable-before-dynamic ordering** — the mostly-static block (persona, tool catalogue) opens the prompt; per-query blocks (digest, plan, hint) follow; the user query is the final token.
4. **Per-reply memoisation** — the main loop's time/location context string is computed once per reply, so all in-loop calls of one reply are byte-identical from token 1; the KV prefix extends through the whole history, not just the system message.
5. **Ollama payloads set `cache_prompt: true` explicitly** on `chat()`, `direct()`, and `streaming()` so the server always retains the request's KV state.

Anything that reorders messages between calls, injects a changing value at the head of a prompt, or rebuilds a system prompt with per-call content breaks prefix reuse for every token after the divergence point.

## Flow

```
user input
  └─▶ [2] Intent Judge            (voice only, SMALL)
        └─▶ [7] Tool router (narrows catalogue for the planner)
              └─▶ [12] Planner (gates memory; advisory for the router allow-list)
                    ├─ plan requests searchMemory  → [3] Enrichment extract → [4] Memory digest (optional)
                    ├─ plan empty (fail-open)      → [3] Enrichment extract → [4] Memory digest
                    └─ plan reply-only             → skip #3 and #4 entirely
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
6. Consider single-model deployments: the FAST tier prefers a small dedicated model while the planner tracks `llm_chat_model`; loading a second model hurts cold-start latency on small hardware. (On an OpenAI-compatible chat provider an unset `fast_model` already resolves to the chat model, so every context rides the one served model.)
7. Narrow `llm_thinking_enabled` to router/planner only, not every context.
8. `intent_judge_timeout_sec` was already reduced from 15s → 6s. Consider racing it against text-based wake detection to avoid blocking the audio loop entirely.

## 21. Model warm-up probe (OpenAI-compatible path)

- **Source**: `src/jarvis/llm/openai_compatible.py` — `warm_up()` (Phase 2)
- **Trigger**: once per model (chat, judge, router) at listener startup, run in parallel daemon threads; the embed model takes a separate path via `backend.embed()` (see Notes)
- **Model / gating**: the model being warmed, via direct `requests.post` (not via `chat()`, no response parsing). The two-phase probe (GET /models + POST /chat/completions) is specific to the `openai_compatible` provider; on the Ollama path the warmup uses `POST /api/generate` instead.
- **What is sent**: a fixed `{"role": "user", "content": "ping"}` message, `max_tokens=1`, `stream=False`
- **Gating**: the warmup always fires when a model is configured (regardless of provider). What differs between providers is the *probe behaviour*: the two-phase chat-completion probe described here is specific to `openai_compatible`; the Ollama warmup sends `POST /api/generate` with `keep_alive`.
- **Output**: `True`/`False` — consumed by the listener startup dashboard (shown as `⚠️ warmup failed — will load on first use`)
- **Limits**: preceded by `GET /models` (Phase 1, capped at 25 % of budget, max 5 s); Phase 2 timeout is `max(0.1, timeout_sec - list_to)` with caller default 60 s. The embed warmup path does not use this two-phase split — it passes the full timeout directly to `backend.embed()`.
- **Data flow**: `warm_up()` → raw `requests.post` → `resp.ok` (any 2xx) → `bool` returned to `_start_llm_warmup()` → listener startup print
- **Notes**: Best-effort and non-blocking. A failed warmup never prevents the listener from starting. Mirrors the Ollama warmup path (`POST /api/generate` in `src/jarvis/llm/ollama.py:335`), which is not tracked as a separate context since it sends no real inference (empty prompt, `num_predict=0`).

    The **embed warmup** is a separate path: `listener.py:_start_llm_warmup` calls `backend.embed("ping", embed_model)` instead of `backend.warm_up()`. This is because embedding-only models (e.g. nomic-embed-text, modernbert) are not served on the `/chat/completions` endpoint. The embed probe uses the `/embeddings` endpoint with a single-token input and has no Phase 1 reachability check.

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
