"""Permissioned, local-only computer control actions."""

from __future__ import annotations

import os
import socket
import shutil
import subprocess
import sys
import webbrowser
import ctypes
import tkinter
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


def _read_clipboard() -> str:
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        if not user32.OpenClipboard(None):
            raise OSError("could not open the clipboard")
        try:
            handle = user32.GetClipboardData(13)
            if not handle:
                return ""
            pointer = kernel32.GlobalLock(handle)
            if not pointer:
                raise OSError("could not read clipboard data")
            try:
                return ctypes.wstring_at(pointer)
            finally:
                kernel32.GlobalUnlock(handle)
        finally:
            user32.CloseClipboard()
    root = tkinter.Tk()
    root.withdraw()
    try:
        return root.clipboard_get()
    finally:
        root.destroy()


def _write_clipboard(text: str) -> None:
    if sys.platform == "win32":
        root = tkinter.Tk()
        root.withdraw()
        try:
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
        finally:
            root.destroy()
        return
    root = tkinter.Tk()
    root.withdraw()
    try:
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
    finally:
        root.destroy()


def _list_windows() -> list[str]:
    if sys.platform != "win32":
        raise OSError("window management is only supported on Windows")
    user32 = ctypes.windll.user32
    titles: list[str] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def callback(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                if buffer.value.strip():
                    titles.append(buffer.value)
        return True

    user32.EnumWindows(callback_type(callback), 0)
    return titles


def _focus_window(title: str) -> bool:
    if sys.platform != "win32":
        raise OSError("window management is only supported on Windows")
    user32 = ctypes.windll.user32
    found = None
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def callback(hwnd, _lparam):
        nonlocal found
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if user32.IsWindowVisible(hwnd) and title.casefold() in buffer.value.casefold():
            found = hwnd
            return False
        return True

    user32.EnumWindows(callback_type(callback), 0)
    if not found or not user32.SetForegroundWindow(found):
        return False
    return True


def _minimize_window(title: str) -> bool:
    if sys.platform != "win32":
        raise OSError("window management is only supported on Windows")
    user32 = ctypes.windll.user32
    found = None
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def callback(hwnd, _lparam):
        nonlocal found
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if user32.IsWindowVisible(hwnd) and title.casefold() in buffer.value.casefold():
            found = hwnd
            return False
        return True

    user32.EnumWindows(callback_type(callback), 0)
    if not found or not user32.ShowWindow(found, 6):
        return False
    return True


class LocalControlTool(Tool):
    """Perform a small, explicitly approved set of local UI actions."""

    @property
    def name(self) -> str:
        return "localControl"

    @property
    def description(self) -> str:
        return (
            "Read or change local clipboard text, copy or move files under configured roots, "
            "list or manage Windows, or open other allowlisted local resources."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "open_application", "open_url", "reveal_path",
                        "clipboard_read", "clipboard_write",
                        "copy_file", "move_file", "rename_file",
                        "list_windows", "focus_window", "minimize_window",
                    ],
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
                "source": {"type": "string", "description": "Existing source file under a configured root."},
                "destination": {"type": "string", "description": "Destination file under a configured root."},
                "text": {"type": "string", "description": "Text to place on the clipboard."},
                "title": {"type": "string", "description": "Case-insensitive window title match."},
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
        if operation == "clipboard_read":
            try:
                text = _read_clipboard()
            except (OSError, RuntimeError, ValueError, tkinter.TclError) as exc:
                return self._blocked(f"clipboard could not be read: {exc}")
            debug_log("localControl read clipboard", "local-control")
            return ToolExecutionResult(success=True, reply_text=f"Clipboard: {text}")

        if operation == "list_windows":
            try:
                titles = _list_windows()
            except (OSError, RuntimeError, ValueError, tkinter.TclError) as exc:
                return self._blocked(f"windows could not be listed: {exc}")
            return ToolExecutionResult(
                success=True,
                reply_text="Open windows:\n" + "\n".join(titles) if titles else "Open windows: none",
            )

        if operation == "clipboard_write":
            text = args.get("text")
            if not isinstance(text, str):
                return self._blocked("clipboard_write requires string 'text'.")
            approval = self._approval(context, {
                "operation": operation,
                "summary": f"Write clipboard text: {len(text)} characters",
                "risk": "Replaces text currently available to other local applications.",
                "reason": "The requested clipboard text is ready to be written.",
            })
            if approval is not None:
                return approval
            try:
                _write_clipboard(text)
            except (OSError, RuntimeError, ValueError) as exc:
                return self._blocked(f"clipboard could not be written: {exc}")
            debug_log("localControl wrote clipboard text", "local-control")
            return ToolExecutionResult(success=True, reply_text=f"Wrote {len(text)} characters to clipboard")

        if operation in {"copy_file", "move_file", "rename_file"}:
            source_arg = args.get("source")
            destination_arg = args.get("destination")
            if not isinstance(source_arg, str) or not source_arg.strip():
                return self._blocked(f"{operation} requires 'source'.")
            if not isinstance(destination_arg, str) or not destination_arg.strip():
                return self._blocked(f"{operation} requires 'destination'.")
            source = Path(os.path.expanduser(source_arg.strip())).resolve()
            destination = Path(os.path.expanduser(destination_arg.strip())).resolve()
            roots = list(_setting(context, "local_control_allowed_roots", []) or [])
            if not _is_under(source, roots) or not _is_under(destination, roots):
                return self._blocked("source and destination must be inside configured allowed roots.")
            if not source.exists() or not source.is_file():
                return self._blocked(f"source file does not exist: {source}")
            summaries = {
                "copy_file": f"Copy file: {source} -> {destination}",
                "move_file": f"Move file: {source} -> {destination}",
                "rename_file": f"Rename file: {source} -> {destination}",
            }
            risks = {
                "copy_file": "Copies a local file and may overwrite the destination.",
                "move_file": "Moves a local file and may overwrite the destination.",
                "rename_file": "Renames a local file and may overwrite the destination.",
            }
            approval = self._approval(context, {
                "operation": operation,
                "summary": summaries[operation],
                "risk": risks[operation],
                "reason": "Both file paths are inside configured allowed roots.",
            })
            if approval is not None:
                return approval
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if operation == "copy_file":
                    shutil.copy2(source, destination)
                else:
                    shutil.move(source, destination)
            except OSError as exc:
                return self._blocked(f"{operation} failed: {exc}")
            debug_log(f"localControl {operation}: {source} -> {destination}", "local-control")
            return ToolExecutionResult(success=True, reply_text=f"{operation.replace('_', ' ').capitalize()} completed: {destination}")

        if operation in {"focus_window", "minimize_window"}:
            title = args.get("title")
            if not isinstance(title, str) or not title.strip():
                return self._blocked(f"{operation} requires 'title'.")
            if sys.platform != "win32":
                return self._blocked("window management is only supported on Windows.")
            verb = "Focus" if operation == "focus_window" else "Minimize"
            approval = self._approval(context, {
                "operation": operation,
                "summary": f"{verb} window: {title.strip()}",
                "risk": (
                    "Changes the active window in the local desktop."
                    if operation == "focus_window"
                    else "Changes window state in the local desktop."
                ),
                "reason": "The requested window title will be matched case-insensitively.",
            })
            if approval is not None:
                return approval
            try:
                changed = _focus_window(title.strip()) if operation == "focus_window" else _minimize_window(title.strip())
            except OSError as exc:
                return self._blocked(str(exc))
            if not changed:
                return self._blocked(f"no visible window matched: {title.strip()}")
            debug_log(f"localControl {operation}: {title.strip()}", "local-control")
            return ToolExecutionResult(success=True, reply_text=f"{verb}d window: {title.strip()}")

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
                port = parsed_url.port
                socket.getaddrinfo(parsed_url.hostname, port, type=socket.SOCK_STREAM)
            except socket.gaierror as exc:
                return self._blocked(
                    f"DNS resolution failed for URL host '{parsed_url.hostname}': {exc}"
                )
            except (OSError, ValueError) as exc:
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
