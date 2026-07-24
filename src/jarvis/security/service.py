"""Orchestration service for audits (read-only)."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Optional

from jarvis.security.collector import WindowsReadOnlyCollector
from jarvis.security.config import SecurityConfig, load_security_config, save_security_config
from jarvis.security.detection import detect_alerts, diff_baseline
from jarvis.security.hash_cache import HashCache
from jarvis.security.paths import default_security_root, ensure_layout
from jarvis.security.reports import render_markdown_report
from jarvis.security.risk import compute_scores, protection_status
from jarvis.security.store import SecurityStore


class _CrossProcessAuditLock:
    """Best-effort exclusive lock so daemon monitor + UI cannot dual-write the store."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: Any = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = None
        try:
            fh = open(self.path, "a+b")
            fh.seek(0)
            if fh.read(1) == b"":
                fh.write(b"0")
                fh.flush()
            fh.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fh = fh
            return True
        except OSError:
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
            return False

    def release(self) -> None:
        fh = self._fh
        self._fh = None
        if fh is None:
            return
        try:
            fh.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            fh.close()
        except OSError:
            pass


_BUSY = {
    "ok": False,
    "error": "audit_in_progress",
    "snapshot": {},
    "baseline_created": False,
    "baseline": {},
    "changes": {"status": "busy", "added": [], "removed": [], "modified": []},
    "alerts": [],
    "scores": {},
    "protection": [],
    "report_path": "",
    "report_markdown": "",
}


class SecurityCenterService:
    """High-level API used by UI / monitor / tests."""

    def __init__(
        self,
        *,
        root: Optional[Path] = None,
        db_path: Optional[str] = None,
        collector: Optional[WindowsReadOnlyCollector] = None,
        config: Optional[SecurityConfig] = None,
    ) -> None:
        self.root = root or default_security_root(db_path=db_path)
        ensure_layout(self.root)
        self.store = SecurityStore(self.root)
        self.config = config or load_security_config(self.root)
        save_security_config(self.root, self.config)  # ensure file exists
        cache_path = self.root / "config" / "hash_cache.json"
        self.hash_cache = HashCache(cache_path)
        self.collector = collector or WindowsReadOnlyCollector(
            hash_cache=self.hash_cache,
            max_cmdline=self.config.max_command_line_chars,
            authorized_prefixes=list(self.config.authorized_path_prefixes),
        )
        self._audit_lock = threading.Lock()
        self._file_lock = _CrossProcessAuditLock(self.root / "config" / "audit.lock")

    def run_audit(self, *, create_baseline_if_missing: bool = True) -> dict[str, Any]:
        if not self._audit_lock.acquire(blocking=False):
            return dict(_BUSY)
        if not self._file_lock.acquire():
            self._audit_lock.release()
            return dict(_BUSY)
        try:
            return self._run_audit_unlocked(create_baseline_if_missing=create_baseline_if_missing)
        finally:
            self._file_lock.release()
            self._audit_lock.release()

    def _run_audit_unlocked(self, *, create_baseline_if_missing: bool = True) -> dict[str, Any]:
        snapshot = self.collector.collect()
        self.hash_cache.save()

        baseline = self.store.load_baseline()
        if baseline is None and create_baseline_if_missing:
            baseline = self.store.save_baseline(snapshot)
            baseline_just_created = True
        else:
            baseline_just_created = False

        changes = diff_baseline(baseline, snapshot)
        # detect_alerts also annotates process trust onto the snapshot
        alerts = detect_alerts(snapshot, self.config, baseline=baseline, changes=changes)
        # Persist AFTER trust annotations so UI/Processes tab has classifications
        self.store.save_snapshot(snapshot, keep=self.config.snapshot_retention)
        # Persist new alerts (skip duplicates by fingerprint recently)
        existing = {a.get("fingerprint") for a in self.store.load_alerts(limit=500)}
        for a in alerts:
            if a.get("fingerprint") in existing:
                continue
            self.store.append_alert(a, max_bytes=self.config.max_event_file_bytes)
            existing.add(a.get("fingerprint"))

        scores = compute_scores(snapshot, alerts, self.config, changes=changes)
        protection = protection_status(snapshot)
        report_md = render_markdown_report(
            snapshot=snapshot,
            scores=scores,
            alerts=alerts,
            changes=changes,
            protection=protection,
        )
        report_path = self.store.save_report(report_md)

        return {
            "ok": True,
            "snapshot": snapshot,
            "baseline_created": baseline_just_created,
            "baseline": {"created_at_utc": (baseline or {}).get("created_at_utc")},
            "changes": changes,
            "alerts": alerts,
            "scores": scores,
            "protection": protection,
            "report_path": str(report_path),
            "report_markdown": report_md,
        }

    def overview(self) -> dict[str, Any]:
        from copy import deepcopy

        latest = self.store.load_latest_snapshot()
        alerts = [a for a in self.store.load_alerts(limit=200) if a.get("status") in (None, "New", "Investigating", "Acknowledged")]
        baseline = self.store.load_baseline()
        changes = diff_baseline(baseline, latest) if latest else {"status": "no_snapshot"}
        scores = {}
        protection = []
        if latest:
            # Score from re-detection on latest snapshot (not historical JSONL) so old
            # acknowledged/new noise from prior buggy runs does not inflate risk cards.
            live_alerts = detect_alerts(deepcopy(latest), self.config, baseline=baseline, changes=changes)
            scores = compute_scores(latest, live_alerts, self.config, changes=changes)
            protection = protection_status(latest)
        return {
            "enabled": self.config.enabled,
            "monitoring": False,  # monitor sets this via status file optionally
            "last_audit_utc": (latest or {}).get("collected_at_utc"),
            "baseline_created_at_utc": (baseline or {}).get("created_at_utc"),
            "active_alerts": len([a for a in alerts if a.get("status") == "New"]),
            "changes_count": sum(len(changes.get(k) or []) for k in ("added", "removed", "modified")),
            "scores": scores,
            "protection": protection,
            "limitations": (latest or {}).get("limitations") or [],
        }
