## Agentic-Loop Evaluator Spec

### Purpose

After each agentic-loop turn that produces natural-language content (as opposed to a tool call), a lightweight LLM decides whether the loop should **terminate** (the agent has done what it can) or **continue** (a tool in the agent's allow-list could directly perform the user's expressed action but the agent replied in prose instead).

The axis is deliberately binary: from the agentic loop's perspective, "satisfied" and "needs_user_input" are the same terminal state — both mean stop looping and hand back to the user. Collapsing them removes the accidental third class that the previous contract had, where a coherent-but-wrong prose reply (agent describes what it *could* do, but doesn't do it) was being marked `satisfied` and shipped.

### Input contract

`evaluate_turn(user_query, assistant_response_summary, available_tools, turns_used, cfg)`:

- `user_query` (str): the redacted user query that opened this reply. Defensively re-redacted on entry.
- `assistant_response_summary` (str): the natural-language content produced by the chat model on the current turn. Redacted on entry in case the model echoed sensitive user text.
- `available_tools` (list of `(name, one_line_description)` tuples): the agent's current allow-list. Engine-supplied, not user data, so not redacted. Names and one-liners only, no schema — enough for the evaluator to judge "could this turn have been a tool call?"
- `turns_used` (int): number of loop turns consumed so far.
- `cfg`: config object providing the base URL, model, and timeout.

### Output contract

`EvaluatorResult(terminal: bool, nudge: str = "", reason: str = "", tool_call: Optional[dict] = None)`.

- `terminal`: `True` means exit the loop and deliver the reply; `False` means keep looping.
- `nudge`: when `terminal=False`, a short directive to the agent telling it which tool to use and what to do with it. Injected into the next turn's system message as `[Agent nudge: ...]`, lasts exactly one turn. Empty when `terminal=True`.
- `reason`: free-text log hint only. Never shown to the user.
- `tool_call`: optional structured `{"name": str, "arguments": dict}` intent. When the judge has identified both a specific tool (that appears in the toolbox) and its arguments — either by salvaging a garbled tool-call attempt or by spotting an obvious missed invocation — it populates this field in addition to the free-form `nudge`. The engine uses the structured form to execute the tool directly on behalf of the agent, bypassing small chat models that ignore textual nudges. `None` when the judge is nudging for prose, is uncertain about arguments, or is returning terminal. The engine rejects the call if `name` is not in the current allow-list, falling back to the text-nudge path.

### Rubric

Return `continue` (non-terminal) when ALL of the following hold:

- the user expressed a clear action or request, AND
- a tool in the agent's toolbox could directly perform it, AND
- the agent's turn was prose (an offer, a suggestion, a description of what it could do) instead of invoking that tool.

Return `terminal` when the agent genuinely finished: delivered a real answer, successfully completed the action, or truthfully said it cannot do this because no tool fits.

Return `continue` when the agent's turn is **garbled** — raw tool-protocol markers (`tool_code` / `tool_output` blocks), special sentinel tokens (`<unused88>` and other `<unused…>` variants), bare `tool_calls:` text, truncated JSON, or code/data dumps where a prose answer should be. The deterministic `_is_malformed_model_output` guard in the engine catches the known shapes before the evaluator even runs; the evaluator's garbled-turn clause is defence-in-depth for novel leaks the guard has not learned yet.

When the garbled turn encodes a **failed tool-call attempt** (e.g. a `tool_code` block wrapping `google_search.search(query="…")`, a bare `tool_calls: [{"name": "webSearch", "arguments": {…}}]` JSON blob, or a `<unused…>` block wrapping a tool invocation), the evaluator salvages the intent: extract the intended tool and arguments from the garbled text, validate that the tool name appears in the turn's allow-list, and name the tool + args both in the free-form `nudge` and in the structured `tool_call` field, e.g. *nudge="call webSearch with query='sam smith biography'"*, *tool_call={"name": "webSearch", "arguments": {"search_query": "sam smith biography"}}*. The engine prefers the structured form: when `tool_call` is present and the name is in the allow-list, the engine runs the tool directly on behalf of the agent via the normal `run_tool_with_retries` path (same allow-list check, schema validation, and redaction guards as a model-emitted call). The structured path exists because small chat models routinely see the textual nudge and reply with more prose instead of actually emitting the tool-call protocol — one or two nudges burned, nudge cap fires, user gets an ungrounded reply. Unrecoverable shapes (truncated JSON with no name, bare `<unused88>` sentinels, random data dumps) fall back to a "produce a natural-language reply" nudge with `tool_call=None`. Arguments absent from the garbled turn must not be fabricated — salvage is strictly extraction.

### Prompt contract

Strict JSON `{"terminal": bool, "nudge": "...", "reason": "...", "tool_call": {"name": "...", "arguments": {...}} | null}`, no prose, no code fences. The parser is lenient (strips markdown fences, extracts embedded JSON objects). `tool_call` is optional and defaults to `null`; malformed shapes (missing `name`, non-string `name`, non-dict `arguments`) are normalised to `null` or an empty arguments dict rather than causing a parse failure.

### Fail-open behaviour

Any of the following collapse to `EvaluatorResult(terminal=True, reason="evaluator_failed_open")`:

- Missing base URL or resolvable model.
- Timeout, connection error, or any other exception from the LLM call.
- Empty response from the LLM.
- JSON parse failure.
- Missing or non-boolean `terminal` field.

The fail-open choice was flipped from the previous contract (which defaulted to `continue`). Biasing toward terminal is safer: spinning in a broken evaluator loop is worse than shipping a possibly-weak reply. `agentic_max_turns` remains as a hard backstop, and the nudge cap (`evaluator_nudge_max`) prevents infinite ping-pong even if the evaluator is live but consistently returns `continue`.

### Timeout

Shares `llm_digest_timeout_sec` (default 8 s) with memory/tool digests.

### Model resolution

`_resolve_evaluator_model(cfg)` picks the first non-empty candidate:

1. `cfg.evaluator_model` (explicit override)
2. `cfg.intent_judge_model` (small, already warm from wake-word path)
3. `cfg.ollama_chat_model` (last resort)

### Gating

`cfg.evaluator_enabled`:

- `None` (default) — auto: ON for SMALL models, OFF for LARGE. Large models terminate on the first natural-language content.
- `True` / `False` — force on/off regardless of model size.

### Relationship to the agentic loop

- Only invoked after a turn produces natural-language content. Tool-call turns bypass the evaluator and keep looping.
- Malformed-JSON fallback replies (canned error text) bypass the evaluator and terminate immediately.
- On `continue` the engine stashes the nudge in `pending_nudge`; the next turn's system-message rebuild appends `[Agent nudge: <text>]` at the end of the first system message and clears the slot. So each nudge lasts exactly one turn — if the model keeps producing prose, the evaluator fires again and generates a fresh nudge.
- On `continue` with a structured `tool_call` whose `name` is in the current allow-list AND is not `toolSearchTool`, the engine also stashes it in `pending_tool_call`. At the top of the next loop iteration — before any chat LLM call — the engine synthesises an assistant message carrying the `tool_calls` payload, runs the tool via `run_tool_with_retries`, records the tool signature in `recent_tool_signatures` for duplicate suppression, and appends the tool result with the same compound-query remainder hint the model-emitted path uses. The textual nudge is cleared for that turn (the tool has run, no need to also shout the directive at the model). This is the actual recovery path for small models: the evaluator-directed tool execution happens deterministically, the chat model only has to synthesise a reply from the tool result on the following turn. Tool calls that fail the allow-list guard, or that name `toolSearchTool` (whose allow-list-widening logic lives only on the model-emitted path), fall through to the textual-nudge path so the safety boundary is never bypassed.
- Before stashing `pending_tool_call`, the engine checks whether `(name, arguments)` duplicates a recent signature in `recent_tool_signatures`. Argument keys are lower-cased for the comparison so evaluator case-flips (`url` vs `URL`) collide. On a hit the loop terminates with the latest plausible candidate reply instead of re-executing — the tool has already run, the result is in history, re-running won't help and would spin until `agentic_max_turns`.
- `cfg.evaluator_nudge_max` (default 2) caps how many **textual** nudges can be issued per reply. Direct-executable `tool_call` results do NOT consume the nudge budget — they are deterministic actions, not directives the model can ignore. A structured `tool_call` that falls back to the textual-nudge path (allow-list miss, or `toolSearchTool`) DOES count. Once the cap is reached, the next textual-nudge `continue` is overridden to terminal. This stops nudge ping-pong when the model consistently ignores the directive.
- The loop tracks the latest plausible candidate and delivers it when `agentic_max_turns` is hit.

### Tests

- `tests/test_evaluator.py` covers parse edge cases, terminal and continue-with-nudge paths, timeout / connection-error fail-open (now terminal), missing-config fail-open, redaction, and the available-tools payload shape.
- `tests/test_engine_tool_search_loop.py` covers the integration with the agentic loop including the continue-then-nudge-then-tool-call sequence.
