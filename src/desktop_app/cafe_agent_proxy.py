"""Proxy HTTP calls to the Rust cafe-orchestrator (localhost)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from jarvis.debug import debug_log

DEFAULT_ORCHESTRATOR_URL = "http://127.0.0.1:8787"
CAFE_SAMPLE_CSV_REL = ("cafe-agent", "data", "sample_sales.csv")


def default_sample_sales_csv() -> str | None:
    """Absolute path to bundled demo sales CSV when present."""
    root = os.environ.get("JARVIS_ROOT", "").strip()
    if not root:
        try:
            from pathlib import Path

            root = str(Path(__file__).resolve().parents[2])
        except Exception:
            return None
    path = os.path.join(root, *CAFE_SAMPLE_CSV_REL)
    if os.path.isfile(path):
        return os.path.abspath(path)
    return None


def orchestrator_base_url() -> str:
    return (
        os.environ.get("CAFE_ORCHESTRATOR_URL", "").strip()
        or DEFAULT_ORCHESTRATOR_URL
    ).rstrip("/")


def fetch_health(timeout_sec: float = 2.0) -> dict[str, Any]:
    url = f"{orchestrator_base_url()}/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        debug_log(f"cafe-agent health unreachable: {exc}", "desktop")
        return {"status": "offline", "error": str(exc)}


def post_task(task: dict[str, Any], timeout_sec: float = 120.0) -> dict[str, Any]:
    url = f"{orchestrator_base_url()}/task"
    body = json.dumps({"task": task}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        debug_log(f"cafe-agent task HTTP {exc.code}: {raw[:200]}", "desktop")
        return {"ok": False, "error": raw, "status": exc.code}
    except Exception as exc:
        debug_log(f"cafe-agent task failed: {exc}", "desktop")
        return {"ok": False, "error": str(exc)}
