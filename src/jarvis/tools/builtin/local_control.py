"""Permissioned, local-only computer control actions."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


APPROVAL_PHRASE = "i approve this local action"


def _setting(context: ToolContext, name: str, default: Any) -> Any:
    return getattr(getattr(context, "cfg", None), name, default)


def _is_under(path: Path, roots: list[str]) -> bool:
    for root in roots:
        try:
            root_path = Path(os.path.expanduser(str(root))).resolve()
            path.relative_to(root_path)
            return True
        except (OSError, ValueError):
            continue
    return False


class LocalControlTool(Tool):
    """Perform a small, explicitly approved set of local UI actions."""

    @property
    def name(self) -> str:
        return "localControl"

    @property
    def description(self) -> str:
        return (
            "With explicit user approval, open an allowlisted local application, "
            "open an HTTP(S) URL, or reveal an existing file or folder."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["open_application", "open_url", "reveal_path"],
                },
                "application": {
                    "type": "string",
                    "description": "Exact application command or path from the configured allowlist.",
                },
                "url": {
                    "type": "string",
                    "description": "HTTP or HTTPS URL to open.",
                },
                "path": {
                    "type": "string",
                    "description": "Existing file or folder under a configured local root.",
                },
            },
            "required": ["operation"],
            "additionalProperties": False,
        }

    def _blocked(self, message: str) -> ToolExecutionResult:
        debug_log(f"localControl blocked: {message}", "local-control")
        return ToolExecutionResult(success=False, reply_text=f"localControl blocked: {message}")

    def _approval(self, context: ToolContext, request: dict) -> Optional[ToolExecutionResult]:
        callback = getattr(context, "approval_callback", None)
        if callback is None:
            prompt = str(getattr(context, "original_prompt", "") or "").casefold()
            if _setting(context, "local_control_require_approval", True):
                if APPROVAL_PHRASE not in prompt:
                    return self._blocked(
                        "explicit approval is required. The user must say "
                        "'I approve this local action'."
                    )
            return None
        debug_log(f"localControl approval requested: {request['summary']}", "local-control")
        try:
            if not callback(request):
                return self._blocked("action rejected by the user.")
        except Exception as exc:
            return self._blocked(f"approval could not be completed: {exc}")
        return None

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        if not isinstance(args, dict):
            return self._blocked("a JSON object is required.")
        context.user_print("🖥️ Checking permission for a local control action…")
        if not _setting(context, "local_control_enabled", False):
            return self._blocked("local control is disabled in settings.")

        operation = str(args.get("operation") or "").strip().lower()
        if operation == "open_application":
            application = args.get("application")
            if not isinstance(application, str) or not application.strip():
                return self._blocked("open_application requires 'application'.")
            application = application.strip()
            allowlist = {
                str(value).strip().casefold()
                for value in (_setting(context, "local_control_allowed_applications", []) or [])
                if str(value).strip()
            }
            if application.casefold() not in allowlist:
                return self._blocked(f"application is not in the allowlist: {application}")
            approval = self._approval(context, {
                "operation": operation,
                "summary": f"Open application: {application}",
                "risk": "Launches a local application.",
                "reason": "The application is allowlisted and ready to launch.",
            })
            if approval is not None:
                return approval
            try:
                subprocess.Popen([application], shell=False)
            except OSError as exc:
                return self._blocked(f"application could not be opened: {exc}")
            debug_log(f"localControl opened application: {application}", "local-control")
            context.user_print(f"✅ Opened local application: {application}")
            return ToolExecutionResult(success=True, reply_text=f"Opened application: {application}")

        if operation == "open_url":
            url = args.get("url")
            if not isinstance(url, str) or not url.strip():
                return self._blocked("open_url requires 'url'.")
            url = url.strip()
            parsed_url = urlparse(url)
            scheme = parsed_url.scheme.casefold()
            if scheme not in {"http", "https"} or not parsed_url.hostname:
                return self._blocked("only http and https URL schemes are allowed.")
            try:
                socket.getaddrinfo(parsed_url.hostname, parsed_url.port, type=socket.SOCK_STREAM)
            except socket.gaierror as exc:
                return self._blocked(
                    f"DNS resolution failed for URL host '{parsed_url.hostname}': {exc}"
                )
            except OSError as exc:
                return self._blocked(
                    f"network validation failed for URL host '{parsed_url.hostname}': {exc}"
                )
            approval = self._approval(context, {
                "operation": operation,
                "summary": f"Open URL: {url}",
                "risk": "Opens a URL in the system browser.",
                "reason": "The URL uses an allowed HTTP(S) scheme and resolved successfully.",
            })
            if approval is not None:
                return approval
            if not webbrowser.open(url):
                return self._blocked("the system browser declined the URL.")
            debug_log(f"localControl opened URL: {url}", "local-control")
            context.user_print(f"✅ Opened URL: {url}")
            return ToolExecutionResult(success=True, reply_text=f"Opened URL: {url}")

        if operation == "reveal_path":
            path_arg = args.get("path")
            if not isinstance(path_arg, str) or not path_arg.strip():
                return self._blocked("reveal_path requires 'path'.")
            path = Path(os.path.expanduser(path_arg.strip())).resolve()
            roots = _setting(context, "local_control_allowed_roots", []) or []
            if not _is_under(path, list(roots)):
                return self._blocked(f"path is not allowed by configured roots: {path}")
            if not path.exists():
                return self._blocked(f"path does not exist: {path}")
            approval = self._approval(context, {
                "operation": operation,
                "summary": f"Reveal path: {path}",
                "risk": "Opens a local file or folder.",
                "reason": "The existing path is inside a configured allowed root.",
            })
            if approval is not None:
                return approval
            try:
                if sys.platform == "win32":
                    os.startfile(str(path))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(path)], shell=False)
                else:
                    subprocess.Popen(["xdg-open", str(path)], shell=False)
            except OSError as exc:
                return self._blocked(f"path could not be opened: {exc}")
            debug_log(f"localControl revealed path: {path}", "local-control")
            context.user_print(f"✅ Opened local path: {path}")
            return ToolExecutionResult(success=True, reply_text=f"Opened local path: {path}")

        return self._blocked(f"unsupported operation: {operation}")
