"""Built-in tool: real ledger summary from local JSON files."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from jarvis.operator.ledger import load_ledger_from_settings

from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


class GetLedgerSummaryTool(Tool):
    """Return purchase and sales totals from local ledger JSON files."""

    @property
    def name(self) -> str:
        return "getLedgerSummary"

    @property
    def description(self) -> str:
        return (
            "Load a summary of local ledger data: invoices (*.json) and sales (sales*.json) "
            "from the configured ledger folder. Use for questions about invoices, purchases, "
            "or sales totals. Returns raw JSON only — no demo data."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def run(
        self, args: Optional[Dict[str, Any]], context: ToolContext
    ) -> ToolExecutionResult:
        summary = load_ledger_from_settings(context.cfg)
        if not summary.get("ok"):
            context.user_print("📒 No live ledger data found.")
        else:
            context.user_print("📒 Loaded ledger summary.")
        return ToolExecutionResult(
            success=True,
            reply_text=json.dumps(summary, indent=2, ensure_ascii=False),
        )
