"""Orchestration service for audits (read-only)."""

from __future__ import annotations

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

    def run_audit(self, *, create_baseline_if_missing: bool = True) -> dict[str, Any]:
        if not self._audit_lock.acquire(blocking=False):
            return {
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
        try:
            return self._run_audit_unlocked(create_baseline_if_missing=create_baseline_if_missing)
        finally:
            self._audit_lock.release()

    def _run_audit_unlocked(self, *, create_baseline_if_missing: bool = True) -> dict[str, Any]:
        snapshot = self.collector.collect()
        self.hash_cache.save()
        self.store.save_snapshot(snapshot, keep=self.config.snapshot_retention)

        baseline = self.store.load_baseline()
        if baseline is None and create_baseline_if_missing:
            baseline = self.store.save_baseline(snapshot)
            baseline_just_created = True
        else:
            baseline_just_created = False

        changes = diff_baseline(baseline, snapshot)
        alerts = detect_alerts(snapshot, self.config, baseline=baseline, changes=changes)
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
        latest = self.store.load_latest_snapshot()
        alerts = [a for a in self.store.load_alerts(limit=200) if a.get("status") in (None, "New", "Investigating", "Acknowledged")]
        baseline = self.store.load_baseline()
        changes = diff_baseline(baseline, latest) if latest else {"status": "no_snapshot"}
        scores = {}
        protection = []
        if latest:
            # Recompute lightly from stored snapshot without recollecting
            all_alerts = self.store.load_alerts(limit=200)
            scores = compute_scores(latest, all_alerts, self.config, changes=changes)
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
