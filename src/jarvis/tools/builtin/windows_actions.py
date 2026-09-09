"""Windows control tools — a small, explicit allow-list (no arbitrary shell).

Each entry maps a canonical action name to a tiny implementation. Anything
outside the allow-list is rejected with a short, honest message. Destructive
steps stay behind confirmation only where the existing engine already prompts;
these actions themselves are idempotent and non-destructive.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional

from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


def _open_url(url: str) -> str:
    import webbrowser
    webbrowser.open(url)
    return f"Otevřeno: {url}"


def _focus_app(name: str) -> str:
    import psutil
    target = (name or "").strip().lower()
    if not target:
        return "Bez názvu apps."
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            nm = (proc.info.get("name") or "").lower()
            if nm and (nm.startswith(target) or target in nm):
                return f"Aplikace {proc.info['name']} (PID {proc.info['pid']}) je spuštěná."
        except Exception:
            continue
    return f"Aplikace '{name}' neběží."


def _launch_app(name: str) -> str:
    import shutil
    import subprocess
    candidates = {
        "notepad": "notepad.exe",
        "poznamkovy blok": "notepad.exe",
        "calc": "calc.exe",
        "kalkulacka": "calc.exe",
        "explorer": "explorer.exe",
        "pruzkumnik": "explorer.exe",
        "msedge": "msedge.exe",
        "edge": "msedge.exe",
        "terminal": "wt.exe",
    }
    exe = candidates.get((name or "").strip().lower())
    if not exe:
        # try bare name on PATH
        found = shutil.which((name or "").strip())
        exe = found or ""
    if not exe:
        return f"Aplikaci '{name}' nelze najít."
    try:
        subprocess.Popen([exe], creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0))
        return f"Spuštěno: {exe}"
    except Exception as exc:
        return f"Spuštění selhalo: {exc}"


def _set_volume(value: int) -> str:
    import ctypes
    v = max(0, min(100, int(value)))
    try:
        wm = ctypes.windll.winmm
        word = (v * 0xFFFF) // 100
        packed = word | (word << 16)
        if wm.waveOutSetVolume(0, packed) == 0:
            return f"Hlasitost nastavena na {v} %."
        return f"Hlasitost: {v} % (kód {wm.waveOutSetVolume(0, packed)})"
    except Exception as exc:
        return f"Nastavení hlasitosti selhalo: {exc}"


def _set_brightness(value: int) -> str:
    import subprocess
    v = max(1, min(100, int(value)))
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-WmiObject -Class WmiMonitorBrightness).SetBrightness({v}); {v}"],
            capture_output=True, text=True, timeout=8,
        )
        if v > 0:
            return f"Jas nastaven na {v} %."
        return "Jas: nepodařilo se nastavit."
    except Exception as exc:
        return f"Nastavení jasu selhalo: {exc}"


def _task_manager() -> str:
    import subprocess
    try:
        subprocess.Popen(["taskmgr.exe"], creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0))
        return "Správce úloh spuštěn."
    except Exception as exc:
        return f"Správce úloh: {exc}"


def _settings() -> str:
    import subprocess
    try:
        # explorer resolves the ms-settings URI directly.
        subprocess.Popen(["explorer.exe", "ms-settings:"], creationflags=(subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0))
        return "Nastavení Windows otevřeno."
    except Exception as exc:
        return f"Nastavení: {exc}"


def _hardware() -> str:
    import platform
    import subprocess
    parts = []
    try:
        parts.append(f"CPU: {platform.processor() or platform.machine()}")
    except Exception:
        pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        parts.append(f"RAM: {vm.total // (1024 * 1024)} MB")
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -Expand FirstChild"],
            capture_output=True, text=True, timeout=6,
        )
        out = (r.stdout or "").strip()
        if out:
            parts.append(f"GPU: {out}")
    except Exception:
        pass
    try:
        import psutil
        b = psutil.sensors_battery()
        if b:
            status = "nabíjí se" if b.power_plugged else "na baterii"
            parts.append(f"Baterie: {int(b.percent)} % ({status})")
    except Exception:
        pass
    return "; ".join(parts) if parts else "Hardware: bez datos."


class WindowsActionsTool(Tool):
    """Explicit allow-list of Windows control actions."""

    @property
    def name(self) -> str:
        return "windowsActions"

    @property
    def description(self) -> str:
        return (
            "Ovládání Windows z allowlistu akcí. Volat s argumenty "
            "{action, value}. Povolené akce: open_app, focus_app, open_url, "
            "set_volume, set_brightness, task_manager, settings, hardware. "
            "Žádný jiný shell."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["open_app", "focus_app", "open_url", "set_volume",
                             "set_brightness", "task_manager", "settings",
                             "hardware"],
                    "description": "Kterou akci z allowlistu provést.",
                },
                "value": {
                    "type": "string",
                    "description": "Hodnota akce (název aplikace, URL, číslo 0–100).",
                },
            },
            "required": ["action"],
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        try:
            action = str((args or {}).get("action", "")).strip().lower()
            value = (args or {}).get("value")
            context.user_print(f"🪟 Windows akce: {action}")
            if action == "open_app":
                text = _launch_app(str(value)) if value else "Chybí název aplikace."
            elif action == "focus_app":
                text = _focus_app(str(value)) if value else "Chybí název aplikace."
            elif action == "open_url":
                text = _open_url(str(value)) if value else "Chybí URL."
            elif action == "set_volume":
                text = _set_volume(int(float(value))) if value is not None else "Chybí hodnota."
            elif action == "set_brightness":
                text = _set_brightness(int(float(value))) if value is not None else "Chybí hodnota."
            elif action == "task_manager":
                text = _task_manager()
            elif action == "settings":
                text = _settings()
            elif action == "hardware":
                text = _hardware()
            else:
                return ToolExecutionResult(
                    success=False,
                    reply_text=f"Neznámá akce '{action}'. povolené: open_app, focus_app, open_url, set_volume, set_brightness, task_manager, settings, hardware.",
                )
            ok = not text.endswith("selhalo.") and "nelze" not in text
            return ToolExecutionResult(success=ok, reply_text=text if ok else f"Chyba: {text}")
        except Exception as exc:
            debug_log(f"windows tool error: {exc}", "tools")
            return ToolExecutionResult(success=False, reply_text=f"Chyba: {exc}")