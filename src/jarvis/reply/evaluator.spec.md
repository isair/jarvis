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

`EvaluatorResult(terminal: bool, nudge: str = "", reason: str = "")`.

- `terminal`: `True` means exit the loop and deliver the reply; `False` means keep looping.
- `nudge`: when `terminal=False`, a short directive to the agent telling it which tool to use and what to do with it. Injected into the next turn's system message as `[Agent nudge: ...]`, lasts exactly one turn. Empty when `terminal=True`.
- `reason`: free-text log hint only. Never shown to the user.

### Rubric

Return `continue` (non-terminal) when ALL of the following hold:

- the user expressed a clear action or request, AND
- a tool in the agent's toolbox could directly perform it, AND
- the agent's turn was prose (an offer, a suggestion, a description of what it could do) instead of invoking that tool.

Return `terminal` when the agent genuinely finished: delivered a real answer, successfully completed the action, or truthfully said it cannot do this because no tool fits.

### Prompt contract

Strict JSON `{"terminal": bool, "nudge": "...", "reason": "..."}`, no prose, no code fences. The parser is lenient (strips markdown fences, extracts embedded JSON objects).

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
- `cfg.evaluator_nudge_max` (default 2) caps how many nudges can be issued per reply. Once the cap is reached, the next `continue` is overridden to terminal. This stops nudge ping-pong when the model consistently ignores the directive.
- The loop tracks the latest plausible candidate and delivers it when `agentic_max_turns` is hit.

### Tests

- `tests/test_evaluator.py` covers parse edge cases, terminal and continue-with-nudge paths, timeout / connection-error fail-open (now terminal), missing-config fail-open, redaction, and the available-tools payload shape.
- `tests/test_engine_tool_search_loop.py` covers the integration with the agentic loop including the continue-then-nudge-then-tool-call sequence.
