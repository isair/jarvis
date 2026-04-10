"""macOS automation tool for Alfred-style desktop control."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any, Dict, Optional

from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


def _escape_applescript_string(value: str) -> str:
    """Escape user text for safe insertion into AppleScript string literals."""
    text = str(value or "")
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def _run_applescript(script: str, timeout: int = 10) -> str:
    """Execute AppleScript and return trimmed stdout."""
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "Unknown AppleScript error"
        raise RuntimeError(stderr)
    return (result.stdout or "").strip()


def _get_calendar_events() -> str:
    return _run_applescript(
        '''
        tell application "Calendar"
            set collectedEvents to {}
            set nowDate to current date
            set startOfDay to nowDate - (time of nowDate)
            set endOfDay to startOfDay + 86399

            repeat with cal in calendars
                set todayEvents to (every event of cal whose start date >= startOfDay and start date <= endOfDay)
                repeat with evt in todayEvents
                    set end of collectedEvents to ((summary of evt) & " at " & ((start date of evt) as text))
                end repeat
            end repeat

            if collectedEvents is {} then
                return "No events today, sir."
            end if
            return collectedEvents as text
        end tell
        '''
    )


def _create_calendar_event(title: str, start_date: str, calendar_name: str = "Home") -> str:
    safe_title = _escape_applescript_string(title)
    safe_start = _escape_applescript_string(start_date)
    safe_calendar = _escape_applescript_string(calendar_name or "Home")
    return _run_applescript(
        f'''
        tell application "Calendar"
            if (count of (every calendar whose name is "{safe_calendar}")) > 0 then
                set targetCalendar to first calendar whose name is "{safe_calendar}"
            else
                set targetCalendar to first calendar
            end if

            set startDateValue to date "{safe_start}"
            set newEvent to make new event at end of events of targetCalendar with properties {{summary:"{safe_title}", start date:startDateValue, end date:startDateValue + (60 * 60)}}
            return "Event created: {safe_title}"
        end tell
        '''
    )


def _create_reminder(task: str, due_date: Optional[str] = None) -> str:
    safe_task = _escape_applescript_string(task)
    if due_date:
        safe_due = _escape_applescript_string(due_date)
        script = f'''
        tell application "Reminders"
            make new reminder with properties {{name:"{safe_task}", due date:(date "{safe_due}")}}
            return "Reminder set: {safe_task}"
        end tell
        '''
    else:
        script = f'''
        tell application "Reminders"
            make new reminder with properties {{name:"{safe_task}"}}
            return "Reminder set: {safe_task}"
        end tell
        '''
    return _run_applescript(script)


def _get_reminders() -> str:
    return _run_applescript(
        '''
        tell application "Reminders"
            set pendingReminders to {}
            repeat with r in (every reminder whose completed is false)
                set end of pendingReminders to (name of r)
            end repeat

            if pendingReminders is {} then
                return "No pending reminders, sir."
            end if
            return pendingReminders as text
        end tell
        '''
    )


def _get_unread_count() -> str:
    return _run_applescript(
        '''
        tell application "Mail"
            set n to unread count of inbox
            return "You have " & n & " unread messages, sir."
        end tell
        '''
    )


def _get_unread_subjects(limit: int = 5) -> str:
    safe_limit = max(1, min(20, int(limit)))
    return _run_applescript(
        f'''
        tell application "Mail"
            set unreadMessages to (every message of inbox whose read status is false)
            set collectedSubjects to {{}}
            set i to 1

            repeat with msg in unreadMessages
                if i > {safe_limit} then exit repeat
                set end of collectedSubjects to ((subject of msg) & " - from: " & (sender of msg))
                set i to i + 1
            end repeat

            if collectedSubjects is {{}} then
                return "No unread messages, sir."
            end if
            return collectedSubjects as text
        end tell
        '''
    )


def _play_music(search_term: Optional[str] = None) -> str:
    if search_term:
        safe_term = _escape_applescript_string(search_term)
        script = f'''
        tell application "Music"
            set resultsList to search library playlist 1 for "{safe_term}"
            if resultsList is {{}} then
                return "No results found for: {safe_term}"
            end if

            play (item 1 of resultsList)
            return "Now playing: " & (name of current track) & " by " & (artist of current track)
        end tell
        '''
    else:
        script = '''
        tell application "Music"
            play
            return "Playback resumed, sir."
        end tell
        '''
    return _run_applescript(script)


def _pause_music() -> str:
    return _run_applescript(
        '''
        tell application "Music"
            pause
            return "Playback paused, sir."
        end tell
        '''
    )


def _get_now_playing() -> str:
    return _run_applescript(
        '''
        tell application "Music"
            if player state is playing then
                return "Now playing: " & (name of current track) & " by " & (artist of current track) & "."
            end if
            return "Nothing is currently playing, sir."
        end tell
        '''
    )


def _set_volume(level: int) -> str:
    clamped = max(0, min(100, int(level)))
    return _run_applescript(
        f'''
        set volume output volume {clamped}
        return "Volume set to {clamped}, sir."
        '''
    )


def _get_volume() -> str:
    current_volume = _run_applescript("return output volume of (get volume settings)")
    return f"Current volume is {current_volume}, sir."


def _open_app(app_name: str) -> str:
    safe_name = _escape_applescript_string(app_name)
    return _run_applescript(
        f'''
        tell application "{safe_name}"
            activate
            return "Opened {safe_name}, sir."
        end tell
        '''
    )


def _quit_app(app_name: str) -> str:
    safe_name = _escape_applescript_string(app_name)
    return _run_applescript(
        f'''
        tell application "{safe_name}"
            quit
            return "Closed {safe_name}, sir."
        end tell
        '''
    )


def _get_battery() -> str:
    result = subprocess.run(
        ["pmset", "-g", "batt"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"(\d+)%", combined)
    if match:
        return f"Battery is at {match.group(1)}%, sir."
    raise RuntimeError("Unable to read battery level")


def _run_shell(command: str) -> str:
    result = subprocess.run(
        ["/bin/zsh", "-lc", command],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=os.path.expanduser("~"),
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    text = "\n".join(part for part in [stdout, stderr] if part)
    if not text:
        text = "Command completed with no output."
    if result.returncode != 0:
        text = f"Command exited with code {result.returncode}.\n{text}"
    return text[:2000]


def _extract_params(args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    nested = args.get("args")
    if isinstance(nested, dict):
        return nested
    return {k: v for k, v in args.items() if k != "action"}


class AppleScriptTool(Tool):
    """Tool for controlling macOS apps and system settings."""

    @property
    def name(self) -> str:
        return "appleScript"

    @property
    def description(self) -> str:
        return (
            "Control macOS apps and system settings via AppleScript. "
            "Use this for Calendar, Reminders, Mail, Music, volume changes, app launch/quit, battery checks, "
            "and simple shell commands on the local Mac."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "get_calendar_events",
                        "create_calendar_event",
                        "create_reminder",
                        "get_reminders",
                        "get_unread_count",
                        "get_unread_subjects",
                        "play_music",
                        "pause_music",
                        "get_now_playing",
                        "set_volume",
                        "get_volume",
                        "open_app",
                        "quit_app",
                        "get_battery",
                        "run_shell",
                    ],
                    "description": "The macOS action to perform.",
                },
                "args": {
                    "type": "object",
                    "description": (
                        "Optional nested arguments for the action. "
                        "Examples: {title, start_date, calendar_name}, {task, due_date}, {search_term}, "
                        "{level}, {app_name}, or {command}."
                    ),
                },
            },
            "required": ["action"],
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        if not isinstance(args, dict):
            return ToolExecutionResult(success=False, reply_text="appleScript requires a JSON object with an 'action'.")

        action = str(args.get("action") or "").strip()
        params = _extract_params(args)
        context.user_print("Handling macOS action...")

        try:
            if action == "get_calendar_events":
                result = _get_calendar_events()
            elif action == "create_calendar_event":
                title = str(params.get("title") or "").strip()
                start_date = str(params.get("start_date") or "").strip()
                if not title or not start_date:
                    return ToolExecutionResult(
                        success=False,
                        reply_text="create_calendar_event requires both 'title' and 'start_date'.",
                    )
                result = _create_calendar_event(title, start_date, str(params.get("calendar_name") or "Home"))
            elif action == "create_reminder":
                task = str(params.get("task") or "").strip()
                if not task:
                    return ToolExecutionResult(success=False, reply_text="create_reminder requires 'task'.")
                due_date = params.get("due_date")
                result = _create_reminder(task, str(due_date).strip() if due_date else None)
            elif action == "get_reminders":
                result = _get_reminders()
            elif action == "get_unread_count":
                result = _get_unread_count()
            elif action == "get_unread_subjects":
                result = _get_unread_subjects(int(params.get("limit", 5)))
            elif action == "play_music":
                search_term = params.get("search_term")
                result = _play_music(str(search_term).strip() if search_term else None)
            elif action == "pause_music":
                result = _pause_music()
            elif action == "get_now_playing":
                result = _get_now_playing()
            elif action == "set_volume":
                if "level" not in params:
                    return ToolExecutionResult(success=False, reply_text="set_volume requires 'level'.")
                result = _set_volume(int(params.get("level", 50)))
            elif action == "get_volume":
                result = _get_volume()
            elif action == "open_app":
                app_name = str(params.get("app_name") or "").strip()
                if not app_name:
                    return ToolExecutionResult(success=False, reply_text="open_app requires 'app_name'.")
                result = _open_app(app_name)
            elif action == "quit_app":
                app_name = str(params.get("app_name") or "").strip()
                if not app_name:
                    return ToolExecutionResult(success=False, reply_text="quit_app requires 'app_name'.")
                result = _quit_app(app_name)
            elif action == "get_battery":
                result = _get_battery()
            elif action == "run_shell":
                command = str(params.get("command") or "").strip()
                if not command:
                    return ToolExecutionResult(success=False, reply_text="run_shell requires 'command'.")
                result = _run_shell(command)
            else:
                return ToolExecutionResult(success=False, reply_text=f"Unknown appleScript action: {action}")

            return ToolExecutionResult(success=True, reply_text=result)
        except Exception as exc:
            message = f"AppleScript error for action '{action}': {exc}"
            return ToolExecutionResult(success=False, reply_text=message, error_message=message)
