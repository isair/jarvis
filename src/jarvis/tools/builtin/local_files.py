"""Local files tool implementation for safe file operations."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


def execute_local_files(tool_args: Optional[Dict[str, Any]]) -> ToolExecutionResult:
    """Execute local file operations safely within user's home directory.

    Args:
        tool_args: Dictionary containing operation, path, and optional parameters

    Returns:
        ToolExecutionResult with operation results
    """
    try:
        # Safety: restrict to user's home directory by default
        home_root = Path(os.path.expanduser("~")).resolve()

        def _expand_user_path(p: str) -> str:
            if not isinstance(p, str):
                return str(p)
            if p == "~":
                return os.path.expanduser("~")
            if p.startswith("~/") or p.startswith("~\\"):
                return os.path.join(os.path.expanduser("~"), p[2:])
            return os.path.expanduser(p)

        def _resolve_safe(p: str) -> Path:
            resolved = Path(_expand_user_path(p)).resolve()
            try:
                # Allow exactly the home root or its descendants
                if resolved == home_root or str(resolved).startswith(str(home_root) + os.sep):
                    return resolved
            except Exception:
                pass
            raise PermissionError(f"Path not allowed: {resolved}")

        if not (tool_args and isinstance(tool_args, dict)):
            return ToolExecutionResult(success=False, reply_text="localFiles requires a JSON object with at least 'operation' and 'path'.")

        operation = str(tool_args.get("operation") or "").strip().lower()
        path_arg = tool_args.get("path")
        if not operation or not path_arg:
            return ToolExecutionResult(success=False, reply_text="localFiles requires 'operation' and 'path'.")

        target = _resolve_safe(str(path_arg))

        # list
        if operation == "list":
            recursive = bool(tool_args.get("recursive", False))
            glob_pat = str(tool_args.get("glob") or "*")
            base = target if target.is_dir() else target.parent
            if not base.exists() or not base.is_dir():
                return ToolExecutionResult(success=False, reply_text=f"Directory not found: {base}")
            try:
                if recursive:
                    items = sorted([str(p) for p in base.rglob(glob_pat)])
                else:
                    items = sorted([str(p) for p in base.glob(glob_pat)])
            except Exception as e:
                return ToolExecutionResult(success=False, reply_text=f"List failed: {e}")
            if not items:
                return ToolExecutionResult(success=True, reply_text=f"No matches under {base} for '{glob_pat}'.")
            preview = "\n".join(items[:200])
            more = "\n..." if len(items) > 200 else ""
            return ToolExecutionResult(success=True, reply_text=f"Listing for {base} ({'recursive' if recursive else 'non-recursive'}) with pattern '{glob_pat}':\n\n{preview}{more}")

        # read
        if operation == "read":
            if not target.exists() or not target.is_file():
                return ToolExecutionResult(success=False, reply_text=f"File not found: {target}")
            try:
                data = target.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return ToolExecutionResult(success=False, reply_text=f"Read failed: {e}")
            max_chars = 100_000
            if len(data) > max_chars:
                return ToolExecutionResult(success=True, reply_text=f"[Truncated to {max_chars} chars]\n" + data[:max_chars])
            return ToolExecutionResult(success=True, reply_text=data)

        # write
        if operation == "write":
            content = tool_args.get("content")
            if not isinstance(content, str):
                return ToolExecutionResult(success=False, reply_text="Write requires string 'content'.")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return ToolExecutionResult(success=True, reply_text=f"Wrote {len(content)} characters to {target}")
            except Exception as e:
                return ToolExecutionResult(success=False, reply_text=f"Write failed: {e}")

        # append
        if operation == "append":
            content = tool_args.get("content")
            if not isinstance(content, str):
                return ToolExecutionResult(success=False, reply_text="Append requires string 'content'.")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8", errors="replace") as f:
                    f.write(content)
                return ToolExecutionResult(success=True, reply_text=f"Appended {len(content)} characters to {target}")
            except Exception as e:
                return ToolExecutionResult(success=False, reply_text=f"Append failed: {e}")

        # delete
        if operation == "delete":
            try:
                if target.exists() and target.is_file():
                    target.unlink()
                    return ToolExecutionResult(success=True, reply_text=f"Deleted file: {target}")
                return ToolExecutionResult(success=False, reply_text=f"File not found: {target}")
            except Exception as e:
                return ToolExecutionResult(success=False, reply_text=f"Delete failed: {e}")

        return ToolExecutionResult(success=False, reply_text=f"Unknown localFiles operation: {operation}")
    except PermissionError as pe:
        return ToolExecutionResult(success=False, reply_text=f"Permission error: {pe}")
    except Exception as e:
        return ToolExecutionResult(success=False, reply_text=f"localFiles error: {e}")


class LocalFilesTool(Tool):
    """Tool for safe local file operations within user's home directory."""

    @property
    def name(self) -> str:
        return "localFiles"

    @property
    def description(self) -> str:
        return "Safely read, write, list, append, or delete files within your home directory."

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "description": "Operation to perform: list, read, write, append, delete"},
                "path": {"type": "string", "description": "File or directory path (relative to home directory)"},
                "content": {"type": "string", "description": "Content to write/append (for write/append operations)"},
                "glob": {"type": "string", "description": "Glob pattern for listing (default: *)"},
                "recursive": {"type": "boolean", "description": "Whether to search recursively (for list operation)"}
            },
            "required": ["operation", "path"]
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Execute the local files tool."""
        return execute_local_files(args)
