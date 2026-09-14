"""Builtin tool for searching explicitly configured local documents."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ...debug import debug_log
from ...memory.document_index import DocumentIndex
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


_FENCE_BEGIN = "<<<BEGIN UNTRUSTED LOCAL DOCUMENT>>>"
_FENCE_END = "<<<END UNTRUSTED LOCAL DOCUMENT>>>"


class DocumentSearchTool(Tool):
    @property
    def name(self) -> str:
        return "documentSearch"

    @property
    def description(self) -> str:
        return "Search configured local text and Markdown documents and return cited excerpts."

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question or terms to search for"},
                "top_k": {"type": "integer", "description": "Maximum cited excerpts to return"},
            },
            "required": ["query"],
        }

    def __init__(self, index_factory: Callable[..., DocumentIndex] = DocumentIndex) -> None:
        self._index_factory = index_factory

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        query = str((args or {}).get("query") or "").strip()
        if not query:
            return ToolExecutionResult(False, "documentSearch requires a search query.")
        if not bool(getattr(context.cfg, "document_search_enabled", False)) or not getattr(
            context.cfg, "document_search_paths", []
        ):
            return ToolExecutionResult(
                False,
                "Local document search is not configured. Add one or more document search folders and enable it in Settings.",
            )
        try:
            top_k = min(max(int((args or {}).get("top_k", 5)), 1), 10)
        except (TypeError, ValueError):
            top_k = 5
        try:
            index = self._index_factory(context.db, context.cfg)
            index.refresh()
            hits = index.search(query, top_k=top_k)
        except Exception as exc:
            debug_log(f"Local document search failed: {exc}", "documents")
            return ToolExecutionResult(False, f"Local document search failed: {exc}")
        if not hits:
            debug_log("Local document search returned no matches", "documents")
            return ToolExecutionResult(
                True,
                f"No matching documents were found for '{query}'. Do not infer an answer from the empty index.",
            )
        excerpts = []
        for hit in hits:
            excerpts.append(
                f"[{hit.path}, lines {hit.start_line}-{hit.end_line}]\n"
                f"{_FENCE_BEGIN}\n{hit.text}\n{_FENCE_END}"
            )
        return ToolExecutionResult(
            True,
            f"Local document results for '{query}'. Use only these cited excerpts:\n"
            + "\n\n".join(excerpts),
        )
