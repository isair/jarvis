"""
Subprocess worker entry point.

This module is executed as a standalone subprocess by
:class:`~jarvis.execution.runner.ToolRunner` when a tool is classified
as HIGH-risk or DESTRUCTIVE and therefore requires process isolation.

The worker:
1. Reads a single JSON :class:`~jarvis.execution.worker_protocol.WorkerRequest`
   from stdin.
2. Imports and executes the requested built-in tool.
3. Writes a JSON :class:`~jarvis.execution.worker_protocol.WorkerResponse`
   to stdout.
4. Exits.

Security considerations
-----------------------
* The worker runs in the same user context as the parent process.
* Future hardening (Windows ICACLS / setuid on Unix) is applied *outside*
  this module — the launcher in :mod:`~jarvis.execution.runner` is
  responsible for setting up the execution context before spawning.
* Only built-in tools are permitted in the worker.  MCP tools are never
  run out-of-process through this path.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
import types


def _write_error(request_id: str, message: str) -> None:
    """
    Write a minimal error :class:`WorkerResponse` to stdout and flush.

    Used when the worker cannot parse or handle a request at all and
    needs to return an error without a full response object.
    """
    error_resp = {
        "request_id": request_id,
        "success": False,
        "reply_text": "",
        "error_message": message,
        "exit_code": 1,
    }
    sys.stdout.write(json.dumps(error_resp) + "\n")
    sys.stdout.flush()


def _timeout_handler(signum, frame):
    sys.stderr.write("worker: timeout — exiting\n")
    sys.exit(2)


def main() -> None:
    """Read one WorkerRequest from stdin, execute, write WorkerResponse to stdout."""
    # Attempt to set a signal-based timeout (not available on Windows)
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)  # type: ignore[attr-defined]
    except AttributeError:
        pass  # Windows — we rely on the parent's forced kill instead

    try:
        raw = sys.stdin.readline()
        if not raw.strip():
            _write_error("", "empty request")
            return

        from jarvis.execution.worker_protocol import WorkerRequest, WorkerResponse

        try:
            req = WorkerRequest.from_json(raw)
        except Exception as exc:
            _write_error("", f"failed to parse request: {exc}")
            return

        # Set alarm timeout (Unix only)
        try:
            signal.alarm(max(1, int(req.timeout_sec)))  # type: ignore[attr-defined]
        except AttributeError:
            pass

        # Execute the tool
        try:
            result = _run_builtin(req.tool_name, req.tool_args, req.safety_config)
            resp = WorkerResponse(
                request_id=req.request_id,
                success=result.get("success", False),
                reply_text=result.get("reply_text", ""),
                error_message=result.get("error_message"),
            )
        except Exception as exc:
            resp = WorkerResponse(
                request_id=req.request_id,
                success=False,
                error_message=str(exc),
            )

        sys.stdout.write(resp.to_json() + "\n")
        sys.stdout.flush()

    except Exception as exc:
        sys.stderr.write(f"worker: unhandled error: {exc}\n")
        sys.exit(1)


def _run_builtin(tool_name: str, tool_args: dict, safety_config: dict) -> dict:
    """
    Import and run the named built-in tool inside the worker subprocess.

    Args:
        tool_name: Canonical tool identifier.
        tool_args: Arguments forwarded from the parent request.
        safety_config: Path-safety constraints from the parent daemon config
            (``workspace_roots``, ``blocked_roots``, ``read_only_roots``,
            ``local_files_mode``).  Used to construct a minimal cfg object
            so that tools enforce the same path constraints as in-process.
    """
    # Lazy imports keep subprocess startup fast.
    # Only built-in tools are permitted here — MCP tools never run out-of-process.
    from jarvis.tools.registry import BUILTIN_TOOLS
    from jarvis.tools.base import ToolContext

    tool = BUILTIN_TOOLS.get(tool_name)
    if tool is None:
        return {"success": False, "reply_text": "", "error_message": f"Unknown tool: {tool_name}"}

    # Reconstruct a minimal cfg-like object from safety fields forwarded
    # by the runner.  This ensures path-safety constraints (workspace_roots,
    # blocked_roots, etc.) are enforced inside the worker even though the
    # full Settings object is not available in the subprocess.
    _safety = safety_config or {}
    _cfg = types.SimpleNamespace(
        workspace_roots=list(_safety.get("workspace_roots") or []),
        blocked_roots=list(_safety.get("blocked_roots") or []),
        read_only_roots=list(_safety.get("read_only_roots") or []),
        local_files_mode=str(_safety.get("local_files_mode") or "workspace"),
    )
    ctx = ToolContext(
        db=None,
        cfg=_cfg,
        system_prompt="",
        original_prompt="",
        redacted_text="",
        max_retries=1,
        user_print=lambda s: None,
    )

    try:
        result = tool.run(tool_args, ctx)
        return {
            "success": result.success,
            "reply_text": result.reply_text or "",
            "error_message": result.error_message,
        }
    except Exception as exc:
        return {"success": False, "reply_text": "", "error_message": str(exc)}


if __name__ == "__main__":
    main()
