## toolSearchTool Spec

### Purpose

Expose the reply engine's tool-routing logic as a callable builtin tool so the agentic loop can widen its own allow-list mid-conversation when the initial routing turned out too narrow.

### Problem

Before each reply, `select_tools` runs once outside the loop and narrows the tool allow-list to the model's best guess given only the user's immediate turn. If the model later realises a different tool is needed (e.g. the user's request was ambiguous, or a clarification reshaped the intent), it cannot access any tool outside that pre-picked set — the loop is stuck with whatever the router picked at turn zero.

### Design

`toolSearchTool` is an escape hatch, not a replacement for `select_tools`. Initial narrow routing still happens once, outside the loop; the loop then exposes:

```
allow-list = <router's picks> + stop + toolSearchTool
```

When the model invokes `toolSearchTool(query=...)`, the tool re-runs the same routing logic (`select_tools` from `src/jarvis/tools/selection.py`) against the new query, and the returned tool names are merged into the loop's allow-list for subsequent turns. `stop` and `toolSearchTool` itself always remain in the allow-list.

### Contract

- **Name**: `toolSearchTool`
- **Description** (visible to the model): "Search the full tool registry for tools that can help with a task. Use this if none of the currently-available tools fit what the user actually needs. Pass a short self-contained description of what you are trying to accomplish."
- **Input schema**:
  - `query` (string, required): a self-contained natural-language description of the subtask needing a tool. Subject to the same `SELF-CONTAINED TOOL ARGUMENTS` rule as every other tool (pronouns and ellipsis resolved from conversation).
- **Output**: a newline-separated list of tool names and one-line descriptions for everything routing surfaced for `query`. On no matches: a short honest note saying no additional tools were found.

### Loop integration

The reply engine:
1. Runs `select_tools(text)` once pre-loop → `base_tools`.
2. Exposes `base_tools ∪ {stop, toolSearchTool}` per turn.
3. On a `toolSearchTool` call, dispatches it (running `select_tools(query)` with the same strategy config), appends the tool result as normal, and merges the returned tool names into the allow-list for the next turn. Duplicates collapse; the list only grows.
4. Neither `stop` nor `toolSearchTool` is ever removed.

Tools surfaced by `toolSearchTool` take effect from the NEXT turn onwards; the current turn's result is already committed. This is inherent to the agentic-loop rhythm and is not a bug.

The engine caps invocations per reply via `tool_search_max_calls` (default 3). Beyond the cap, further calls get a tool-error result telling the model to decide with the tools already available.

### What toolSearchTool is NOT

- Not a free-form tool discovery surface: it uses the same routing pipeline as the pre-loop call, not a raw "list every tool" dump. The router already applies allow/deny logic and MCP-awareness; reusing it keeps semantics consistent.
- Not a way to bypass authorisation: if the router would not have picked a tool pre-loop, `toolSearchTool` will not surface it either.
- Not free: each call is an LLM round-trip. The model is told to use it only when none of the currently-available tools fit.

### Testing

- Unit tests cover the merge-into-allow-list behaviour and the no-results branch.
- An eval scenario covers the "initial routing was too narrow" case: the user starts with a vague question that routes to one tool, then clarifies into a request that needs a different tool. The agent should invoke `toolSearchTool` and then the newly-surfaced tool.
