"""Typed snapshot / alert models for Security Center."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


SEVERITIES = ("info", "low", "medium", "high", "critical")
TRUST_LEVELS = ("trusted", "authorized_project", "baseline_known", "unknown", "suspicious", "critical")
ALERT_STATUSES = ("New", "Acknowledged", "Trusted", "Investigating", "Resolved")
PROTECTION_STATES = ("Healthy", "Warning", "Critical", "Unknown", "Permission Required")
FINDING_LEVELS = ("clean", "unknown", "warning", "suspicious", "critical")


@dataclass
class ProtectionItem:
    name: str
    state: str  # PROTECTION_STATES
    detail: str = ""
    permission_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SecurityAlert:
    id: str
    timestamp_utc: str
    severity: str
    category: str
    title: str
    artifact: str = ""
    path: str = ""
    publisher: str = ""
    signature: str = ""
    sha256: str = ""
    connection: str = ""
    reason: str = ""
    evidence: list[str] = field(default_factory=list)
    recommendation: str = ""
    status: str = "New"
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreCard:
    name: str
    score: int  # 0–100 higher = more risk (except overall which we also keep as risk)
    explanation: str
    evidence: list[str] = field(default_factory=list)
    confidence: str = "medium"  # low|medium|high
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def empty_snapshot(*, collected_at_utc: str, host: str = "", limitations: Optional[list[str]] = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "collected_at_utc": collected_at_utc,
        "host": host,
        "read_only": True,
        "limitations": list(limitations or []),
        "users": [],
        "administrators": [],
        "sessions": [],
        "logon_events": [],
        "processes": [],
        "services": [],
        "scheduled_tasks": [],
        "startup_items": [],
        "run_keys": {},
        "connections": [],
        "listening": [],
        "smb_shares": [],
        "smb_sessions": [],
        "remote_surface": {},
        "hosts_file": {"path": "", "extra_lines": [], "sha256": ""},
        "dns_cache": [],
        "browser_extensions": [],
        "recent_executables": [],
        "firewall": {},
        "antivirus": {},
        "secure_boot": {},
        "tpm": {},
        "bitlocker": {},
        "windows_update": {},
        "event_logs": {},
        "sysmon": {},
        "audit_policy": {},
        "persistence_special": {},
        "remote_tools_detected": [],
    }
