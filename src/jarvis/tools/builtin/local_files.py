"""Local files tool implementation for safe file operations."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


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

            if not (args and isinstance(args, dict)):
                return ToolExecutionResult(success=False, reply_text="localFiles requires a JSON object with at least 'operation' and 'path'.")

            operation = str(args.get("operation") or "").strip().lower()
            path_arg = args.get("path")
            if not operation or not path_arg:
                return ToolExecutionResult(success=False, reply_text="localFiles requires 'operation' and 'path'.")

            target = _resolve_safe(str(path_arg))

            # list
            if operation == "list":
                if not target.exists():
                    return ToolExecutionResult(success=False, reply_text=f"Path not found: {target}")
                if target.is_file():
                    return ToolExecutionResult(success=True, reply_text=f"File: {target.name}")

                glob_pattern = args.get("glob", "*")
                recursive = bool(args.get("recursive", False))

                try:
                    if recursive:
                        files = list(target.rglob(glob_pattern))
                    else:
                        files = list(target.glob(glob_pattern))

                    if not files:
                        return ToolExecutionResult(success=True, reply_text=f"No files found matching '{glob_pattern}' in {target}")

                    file_list = []
                    for f in sorted(files)[:50]:  # Limit to 50 files
                        relative_path = f.relative_to(target)
                        file_type = "DIR" if f.is_dir() else "FILE"
                        file_list.append(f"  {file_type}: {relative_path}")

                    result = f"Contents of {target}:\n" + "\n".join(file_list)
                    if len(files) > 50:
                        result += f"\n... and {len(files) - 50} more files"

                    return ToolExecutionResult(success=True, reply_text=result)
                except Exception as e:
                    return ToolExecutionResult(success=False, reply_text=f"List failed: {e}")

            # read
            if operation == "read":
                if not target.exists() or not target.is_file():
                    return ToolExecutionResult(success=False, reply_text=f"File not found: {target}")
                try:
                    data = target.read_text(encoding="utf-8", errors="replace")
                    max_chars = 10000
                    if len(data) > max_chars:
                        data = data[:max_chars] + f"\n... (truncated, showing first {max_chars} chars)"
                    return ToolExecutionResult(success=True, reply_text=data)
                except Exception as e:
                    return ToolExecutionResult(success=False, reply_text=f"Read failed: {e}")

            # write
            if operation == "write":
                content = args.get("content")
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
                content = args.get("content")
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
