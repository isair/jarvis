## Tool Selection Spec

Selects a subset of available tools relevant to a given user query, so the LLM receives only tools it is likely to need. Reduces noise for smaller models and lowers token cost.

### ToolSelectionStrategy Enum

```python
class ToolSelectionStrategy(Enum):
    ALL = "all"
    KEYWORD = "keyword"
    EMBEDDING = "embedding"
    LLM = "llm"
```

### Strategies

Controlled by `tool_selection_strategy` in config:

| Value         | Behaviour                                                           | LLM call? | Extra dependency |
|---------------|---------------------------------------------------------------------|-----------|------------------|
| `"all"`       | Pass every registered tool.                                         | No        | None             |
| `"keyword"`   | Score tools by keyword overlap with the query; return top matches.  | No        | None             |
| `"embedding"` | Rank tools by cosine similarity of embeddings via nomic-embed-text (default). | No | numpy      |
| `"llm"`       | Ask a lightweight LLM call to pick relevant tool names.             | Yes       | None             |

### Always-included Tools

Regardless of strategy, these tools are **always** included:
- `stop` — needed so the user can dismiss the assistant at any time.

### Keyword Strategy

1. Build a keyword index per tool from its `name` (camelCase split) and `description` (lowercased, stop-words removed).
2. Tokenise the user query (lowercase, split on whitespace/punctuation).
3. Score each tool: count of query tokens that appear in the tool's keyword set.
4. Return tools with score > 0, plus always-included tools.
5. If no tools score > 0, fall back to returning all tools (query is too vague to filter).

### Embedding Strategy

1. Embed the user query using `get_embedding()` (calls Ollama `/api/embeddings` with the configured embed model).
2. For each tool (excluding always-included), build a summary string from the tool name (camelCase split) and description, then embed it.
3. Compute cosine similarity between the query embedding and each tool embedding.
4. Select tools using a **relative threshold**: keep tools whose similarity >= `top_score * _RELATIVE_THRESHOLD` (0.85). This adapts to the actual score distribution rather than a fixed cutoff that either passes everything or nothing.
5. If fewer than `_MIN_SELECTED` (3) tools pass the threshold, return the top 3 by similarity.
6. Append always-included tools.
7. If the query embedding fails, fall back to returning all tools.

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
    strategy: ToolSelectionStrategy = ToolSelectionStrategy.ALL,
    llm_base_url: str = "",
    llm_model: str = "",
    llm_timeout_sec: float = 8.0,
    embed_model: str = "",
    embed_timeout_sec: float = 10.0,
) -> List[str]:
    """Return list of tool names relevant to the query."""
```

### Integration

Called from the reply engine (Step 6) before `generate_tools_json_schema()` and `generate_tools_description()`. The returned list replaces the current `allowed_tools = list(BUILTIN_TOOLS.keys())`.

### Configuration

- Key: `tool_selection_strategy`
- Type: `str` (validated against `ToolSelectionStrategy` enum values)
- Default: `"embedding"`
- Valid values: `"all"`, `"keyword"`, `"embedding"`, `"llm"`
