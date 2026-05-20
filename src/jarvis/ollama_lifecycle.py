"""Ollama keep-alive and shutdown aligned with the Jarvis desktop session."""

from __future__ import annotations

import subprocess
from typing import Any, Optional

from jarvis.debug import debug_log

_session_keep_alive: str = "10m"
_ollama_server_process: Optional[subprocess.Popen] = None
_ollama_started_by_jarvis: bool = False


def configure_from_settings(cfg: Any) -> None:
    """Load keep-alive duration from config (refreshed on each Ollama call while active)."""
    global _session_keep_alive
    raw = str(getattr(cfg, "ollama_keep_alive", "") or "10m").strip()
    _session_keep_alive = raw or "10m"


def get_keep_alive() -> str:
    """Value sent on every Jarvis Ollama request while the session is active."""
    return _session_keep_alive


def register_ollama_server_process(
    proc: subprocess.Popen | None, *, started_by_jarvis: bool
) -> None:
    """Track ``ollama serve`` if the desktop app launched it (so we can stop it on exit)."""
    global _ollama_server_process, _ollama_started_by_jarvis
    _ollama_server_process = proc
    _ollama_started_by_jarvis = bool(started_by_jarvis and proc is not None)
    if _ollama_started_by_jarvis:
        debug_log(
            f"ollama serve registered (pid={proc.pid if proc else 'n/a'})",
            "desktop",
        )


def _models_to_unload(cfg: Any) -> list[str]:
    names: list[str] = []
    for attr in (
        "ollama_chat_model",
        "intent_judge_model",
        "tool_router_model",
        "planner_model",
        "ollama_embed_model",
        "ollama_vision_model",
    ):
        m = str(getattr(cfg, attr, "") or "").strip()
        if m and m not in names:
            names.append(m)
    return names


def unload_ollama_models(cfg: Any) -> None:
    """Drop loaded weights immediately (keep_alive=0) so GPU/RAM is freed."""
    if not getattr(cfg, "ollama_unload_on_stop", True):
        return
    try:
        import requests
    except ImportError:
        return

    base = str(getattr(cfg, "ollama_base_url", "") or "http://127.0.0.1:11434").rstrip(
        "/"
    )
    for model in _models_to_unload(cfg):
        try:
            requests.post(
                f"{base}/api/generate",
                json={
                    "model": model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0,
                    "options": {"num_predict": 1},
                },
                timeout=8,
            )
            debug_log(f"ollama unload requested for {model}", "desktop")
        except Exception as exc:
            debug_log(f"ollama unload failed for {model}: {exc}", "desktop")


def _stop_registered_ollama_server() -> None:
    global _ollama_server_process, _ollama_started_by_jarvis
    proc = _ollama_server_process
    if not proc or not _ollama_started_by_jarvis:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
            debug_log("ollama serve stopped (started by Jarvis)", "desktop")
    except Exception as exc:
        debug_log(f"ollama serve stop failed: {exc}", "desktop")
    finally:
        _ollama_server_process = None
        _ollama_started_by_jarvis = False


def ollama_server_reachable(cfg: Any, *, timeout_sec: float = 2.0) -> bool:
    """True when ``ollama serve`` responds on the configured base URL."""
    try:
        import requests
    except ImportError:
        return False
    base = str(getattr(cfg, "ollama_base_url", "") or "http://127.0.0.1:11434").rstrip(
        "/"
    )
    try:
        resp = requests.get(f"{base}/api/tags", timeout=timeout_sec)
        return resp.status_code == 200
    except Exception:
        return False


def release_ollama_session(cfg: Any | None = None, *, stop_server: bool = False) -> None:
    """Unload models; optionally stop ``ollama serve`` if Jarvis started it."""
    if cfg is None:
        from jarvis.config import load_settings

        cfg = load_settings()
    unload_ollama_models(cfg)
    if stop_server and getattr(cfg, "ollama_stop_with_jarvis", True):
        _stop_registered_ollama_server()
