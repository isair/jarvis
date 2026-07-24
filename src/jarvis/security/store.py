"""Local persistence: snapshots, baseline, events (JSON/JSONL), atomic writes."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jarvis.security.paths import ensure_layout
from jarvis.security.redact_ext import scrub_obj
from jarvis.security.atomic_io import atomic_write_json, atomic_write_text


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class SecurityStore:
    def __init__(self, root: Path) -> None:
        self.paths = ensure_layout(root)
        self.root = root

    # ── baseline ──────────────────────────────────────────────
    def baseline_path(self) -> Path:
        return self.paths["baseline"] / "baseline.json"

    def load_baseline(self) -> Optional[dict[str, Any]]:
        p = self.baseline_path()
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            bak = p.with_name(p.name + ".bak")
            if bak.exists():
                try:
                    data = json.loads(bak.read_text(encoding="utf-8"))
                    return data if isinstance(data, dict) else None
                except (OSError, json.JSONDecodeError):
                    return None
            return None

    def save_baseline(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        baseline = {
            "schema_version": 1,
            "created_at_utc": utc_now_iso(),
            "host": snapshot.get("host", ""),
            "snapshot": scrub_obj(_baseline_view(snapshot)),
        }
        atomic_write_json(self.baseline_path(), baseline)
        return baseline

    # ── snapshots ─────────────────────────────────────────────
    def save_snapshot(self, snapshot: dict[str, Any], *, keep: int = 60) -> Path:
        ts = snapshot.get("collected_at_utc", utc_now_iso()).replace(":", "").replace("-", "")
        name = f"snapshot_{ts}_{uuid.uuid4().hex[:8]}.json"
        path = self.paths["snapshots"] / name
        atomic_write_json(path, scrub_obj(snapshot))
        self._rotate_snapshots(keep=keep)
        # pointer to latest
        atomic_write_json(self.paths["snapshots"] / "latest.json", {"path": name, "collected_at_utc": snapshot.get("collected_at_utc")})
        return path

    def load_latest_snapshot(self) -> Optional[dict[str, Any]]:
        latest = self.paths["snapshots"] / "latest.json"
        if not latest.exists():
            return None
        try:
            meta = json.loads(latest.read_text(encoding="utf-8"))
            path = self.paths["snapshots"] / meta["path"]
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    def list_snapshots(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.paths["snapshots"].glob("snapshot_*.json"), reverse=True):
            out.append({"name": p.name, "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()})
        return out

    def load_snapshot_file(self, name: str) -> Optional[dict[str, Any]]:
        # prevent path traversal
        safe = Path(name).name
        if not safe.startswith("snapshot_") or not safe.endswith(".json"):
            return None
        path = self.paths["snapshots"] / safe
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _rotate_snapshots(self, *, keep: int) -> None:
        files = sorted(self.paths["snapshots"].glob("snapshot_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[keep:]:
            try:
                old.unlink()
            except OSError:
                pass

    # ── events / alerts JSONL ─────────────────────────────────
    def events_file(self) -> Path:
        return self.paths["events"] / "alerts.jsonl"

    def append_alert(self, alert: dict[str, Any], *, max_bytes: int = 5_000_000) -> None:
        path = self.events_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(scrub_obj(alert), ensure_ascii=False) + "\n"
        if path.exists() and path.stat().st_size + len(line.encode("utf-8")) > max_bytes:
            rotated = path.with_name(f"alerts_{utc_now_iso().replace(':', '')}.jsonl")
            try:
                path.replace(rotated)
            except OSError:
                atomic_write_text(path, "")  # truncate fallback
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def load_alerts(self, *, limit: int = 200) -> list[dict[str, Any]]:
        path = self.events_file()
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except json.JSONDecodeError:
                continue
        return list(reversed(out))

    def update_alert_status(self, alert_id: str, status: str) -> bool:
        allowed = {"New", "Acknowledged", "Trusted", "Investigating", "Resolved"}
        if status not in allowed:
            return False
        alerts = list(reversed(self.load_alerts(limit=5000)))  # chronological
        changed = False
        for a in alerts:
            if a.get("id") == alert_id:
                a["status"] = status
                a["status_updated_utc"] = utc_now_iso()
                changed = True
                break
        if not changed:
            return False
        # rewrite file atomically
        body = "".join(json.dumps(scrub_obj(a), ensure_ascii=False) + "\n" for a in alerts)
        atomic_write_text(self.events_file(), body, backup=True)
        return True

    def save_report(self, markdown: str, *, name: Optional[str] = None) -> Path:
        fname = name or f"report_{utc_now_iso().replace(':', '')}.md"
        safe = Path(fname).name
        if not safe.endswith(".md"):
            safe += ".md"
        path = self.paths["reports"] / safe
        atomic_write_text(path, markdown)
        return path

    def list_reports(self) -> list[dict[str, Any]]:
        return [
            {"name": p.name, "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()}
            for p in sorted(self.paths["reports"].glob("*.md"), reverse=True)
        ]

    def read_report(self, name: str) -> Optional[str]:
        safe = Path(name).name
        path = self.paths["reports"] / safe
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None


def _baseline_view(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Store only metadata needed for diffs — no private content."""
    return {
        "users": [{"name": u.get("name"), "enabled": u.get("enabled"), "password_required": u.get("password_required")} for u in snapshot.get("users", [])],
        "administrators": list(snapshot.get("administrators", [])),
        "services": [
            {"name": s.get("name"), "start_mode": s.get("start_mode"), "path": s.get("path"), "state": s.get("state")}
            for s in snapshot.get("services", [])
        ],
        "scheduled_tasks": [
            {"path": t.get("path"), "name": t.get("name"), "action": t.get("action"), "state": t.get("state")}
            for t in snapshot.get("scheduled_tasks", [])
        ],
        "run_keys": snapshot.get("run_keys", {}),
        "startup_items": snapshot.get("startup_items", []),
        "listening": [
            {"address": x.get("address"), "port": x.get("port"), "process": x.get("process"), "path": x.get("path")}
            for x in snapshot.get("listening", [])
        ],
        "hosts_file": snapshot.get("hosts_file", {}),
        "remote_surface": snapshot.get("remote_surface", {}),
        "processes_indexed": [
            {
                "name": p.get("name"),
                "path": p.get("path"),
                "publisher": p.get("publisher"),
                "signature": p.get("signature"),
                "sha256": p.get("sha256"),
            }
            for p in snapshot.get("processes", [])
            if p.get("path")
        ],
        "firewall": snapshot.get("firewall", {}),
        "antivirus": snapshot.get("antivirus", {}),
        "secure_boot": snapshot.get("secure_boot", {}),
        "persistence_special": snapshot.get("persistence_special", {}),
    }
