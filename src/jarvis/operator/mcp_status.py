"""MCP integration health — real discovery results, no demo stubs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jarvis.config import default_config_path
from jarvis.debug import debug_log


def _status_path() -> Path:
    return default_config_path().parent / "mcp_status.json"


def _group_tools_by_server(mcp_tools: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for full_name in mcp_tools:
        if "__" not in full_name:
            continue
        server, short = full_name.split("__", 1)
        out.setdefault(server, []).append(short)
    return out


def build_mcp_integrations_status(
    mcps_config: dict[str, Any],
    mcp_tools: dict[str, Any],
    errors: dict[str, str],
) -> dict[str, Any]:
    """Build per-server status from the latest discovery pass."""
    by_server = _group_tools_by_server(mcp_tools)
    servers: dict[str, Any] = {}
    for name, cfg in (mcps_config or {}).items():
        if not isinstance(cfg, dict):
            cfg = {}
        tools = by_server.get(name, [])
        err = errors.get(name)
        if tools:
            state = "ready"
            detail = f"{len(tools)} tool(s) available"
        elif err:
            state = "error"
            detail = err
        else:
            state = "empty"
            detail = "Server reachable but no tools listed"
        servers[name] = {
            "state": state,
            "detail": detail,
            "tool_count": len(tools),
            "tools_preview": tools[:8],
            "transport": cfg.get("transport", "stdio"),
            "comms": _is_comms_server(name),
        }
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "server_count": len(servers),
        "ready_count": sum(1 for s in servers.values() if s["state"] == "ready"),
        "servers": servers,
        "errors": {k: v for k, v in errors.items() if k != "_global"},
        "global_error": errors.get("_global"),
    }


def _is_comms_server(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in ("whatsapp", "gmail", "google", "slack", "telegram"))


def save_mcp_status(payload: dict[str, Any]) -> None:
    path = _status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    debug_log(f"mcp status written to {path}", "mcp")


def load_mcp_status() -> dict[str, Any]:
    path = _status_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def format_mcp_briefing_block(status: dict[str, Any]) -> str:
    servers = status.get("servers") or {}
    if not servers:
        return ""
    lines = [
        "MCP integrations (from last discovery; use matching tools when the user asks for comms or browser actions):"
    ]
    for name, info in servers.items():
        if not isinstance(info, dict):
            continue
        st = info.get("state", "unknown")
        preview = ", ".join(info.get("tools_preview") or [])[:120]
        lines.append(f"- {name}: {st} — {info.get('detail', '')}")
        if preview:
            lines.append(f"  Tools: {preview}")
    return "\n".join(lines)
