## Tool Selection Spec

Selects a subset of available tools relevant to a given user query, so the LLM receives only tools it is likely to need. Reduces noise for smaller models and lowers token cost.

### Strategies

Controlled by `tool_selection_strategy` in config:

| Value       | Behaviour                                                         | LLM call? |
|-------------|-------------------------------------------------------------------|-----------|
| `"all"`     | Pass every registered tool (current default).                     | No        |
| `"keyword"` | Score tools by keyword overlap with the query; return top matches.| No        |
| `"llm"`     | Ask a lightweight LLM call to pick relevant tool names.           | Yes       |

### Always-included Tools

Regardless of strategy, these tools are **always** included:
- `stop` — needed so the user can dismiss the assistant at any time.

### Keyword Strategy

1. Build a keyword index per tool from its `name` (camelCase split) and `description` (lowercased, stop-words removed).
2. Tokenise the user query (lowercase, split on whitespace/punctuation).
3. Score each tool: count of query tokens that appear in the tool's keyword set.
4. Return tools with score > 0, plus always-included tools.
5. If no tools score > 0, fall back to returning all tools (query is too vague to filter).

### LLM Strategy

1. Build a numbered list of tool names + one-line descriptions.
2. Send to `call_llm_direct` with a system prompt asking for a comma-separated list of relevant tool names.
3. Parse the response, matching against known tool names.
4. Return matched tools plus always-included tools.
5. On timeout or parse failure, fall back to returning all tools.

### Interface

```python
def select_tools(
    query: str,
    builtin_tools: Dict[str, Tool],
    mcp_tools: Dict[str, ToolSpec],
    strategy: str,               # "all", "keyword", or "llm"
    llm_base_url: str = "",      # needed only for "llm" strategy
    llm_model: str = "",         # needed only for "llm" strategy
    llm_timeout_sec: float = 8.0,
) -> List[str]:
    """Return list of tool names relevant to the query."""
```

### Integration

Called from the reply engine (Step 6) before `generate_tools_json_schema()` and `generate_tools_description()`. The returned list replaces the current `allowed_tools = list(BUILTIN_TOOLS.keys())`.

### Configuration

- Key: `tool_selection_strategy`
- Type: `str`
- Default: `"all"` (backwards-compatible)
- Valid values: `"all"`, `"keyword"`, `"llm"`
