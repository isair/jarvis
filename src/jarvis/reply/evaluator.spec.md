## Agentic-Loop Evaluator Spec

### Purpose

After each agentic-loop turn that produces natural-language content (as opposed to a tool call), a lightweight LLM decides whether the loop should terminate or keep working. This replaces the previous brittle heuristic of "first content wins" and the force-tool-invocation backstop.

### Input contract

`evaluate_turn(user_query, assistant_response_summary, turns_used, cfg)`:

- `user_query` (str): the redacted user query that opened this reply. Defensively re-redacted on entry.
- `assistant_response_summary` (str): the natural-language content produced by the chat model on the current turn. Redacted on entry in case the model echoed sensitive user text.
- `turns_used` (int): number of loop turns consumed so far, surfaced to the evaluator so it can factor urgency into its choice.
- `cfg`: config object providing the base URL, model, and timeout.

### Output contract

`EvaluatorResult(terminal: bool, reason: str, clarification_question: Optional[str])`.

`reason` is one of:

- `"satisfied"` - the reply addresses the user's query; terminate and deliver.
- `"needs_user_input"` - the reply is a clarifying question to the user; terminate. When `clarification_question` is a non-empty string, the engine overrides the raw candidate reply with this refined phrasing.
- `"continue"` - anything else; keep looping.

`terminal` is true iff `reason != "continue"`.

### Prompt contract

The evaluator is instructed to reply with strict JSON `{"terminal": bool, "reason": "...", "clarification_question"?: "..."}`, no prose, no code fences. The parser is lenient and strips markdown fences / extracts embedded JSON objects where needed.

### Fail-open behaviour

Any of the following collapse to `reason="continue"`:

- Missing base URL or resolvable model.
- Timeout, connection error, or any other exception from the LLM call.
- Empty response from the LLM.
- JSON parse failure.
- Unknown `reason` value.

This keeps `agentic_max_turns` as the single hard backstop and prevents a flaky evaluator from breaking replies.

### Timeout

Shares `llm_digest_timeout_sec` (default 8 s) with memory/tool digests. These are all short classification-shaped calls and should fail fast; the longer `llm_tools_timeout_sec` would stall the reply loop.

### Model resolution

`_resolve_evaluator_model(cfg)` picks the first non-empty candidate:

1. `cfg.evaluator_model` (explicit override)
2. `cfg.intent_judge_model` (small, already warm from wake-word path)
3. `cfg.ollama_chat_model` (last resort)

Mirrors `_resolve_tool_router_model` in the engine.

### Gating

`cfg.evaluator_enabled`:

- `None` (default) - auto: ON for SMALL models, OFF for LARGE. Large models terminate on the first natural-language content, matching prior behaviour.
- `True` / `False` - force on/off regardless of model size.

When gated off, the engine treats the first natural-language candidate as terminal.

### Relationship to the agentic loop

- Only invoked after a turn produces natural-language content. Tool-call turns bypass the evaluator and keep looping.
- Malformed-JSON fallback replies (canned error text) bypass the evaluator and terminate immediately.
- The loop tracks the latest plausible `continue` candidate and delivers it when `agentic_max_turns` is hit, rather than falling back to a generic error message.

### Tests

- `tests/test_evaluator.py` covers parse edge cases, each reason path, timeout / ConnectionError fail-open, missing-config fail-open, and the clarification_question override.
- `tests/test_engine_tool_search_loop.py` covers the integration with the agentic loop including the continue-then-toolSearchTool sequence.
