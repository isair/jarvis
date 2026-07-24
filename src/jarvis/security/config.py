"""Security Center configuration + risk weight documentation.

Weights are explicit and editable via ``security/config/security_config.json``.
No personal project paths are hardcoded — use placeholders / env / allowlist files.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from jarvis.security.atomic_io import atomic_write_json


# Documented score weights (0–100 contribution caps per category).
DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "remote_access": {
        "remote_tool_present": 40.0,
        "rdp_enabled": 25.0,
        "winrm_running": 15.0,
        "ssh_listening": 15.0,
        "smb_session_active": 10.0,
        "unknown_listener_all_interfaces": 20.0,
        "unknown_outbound": 15.0,
    },
    "persistence": {
        "new_run_key": 20.0,
        "new_task": 20.0,
        "new_service": 20.0,
        "hosts_modified": 15.0,
        "wmi_nondefault": 30.0,
        "ifeo_debugger": 35.0,
        "appinit_dll": 35.0,
        "winlogon_shell_changed": 40.0,
    },
    "account": {
        "blank_password_admin": 35.0,
        "new_admin": 40.0,
        "new_user": 20.0,
        "guest_enabled": 25.0,
    },
    "network": {
        "new_listening_port": 15.0,
        "unknown_established": 20.0,
        "firewall_disabled": 30.0,
    },
    "malware_indicators": {
        "unsigned_temp_exe": 25.0,
        "trusted_name_bad_path": 30.0,
        "remote_tool_keyword": 35.0,
        "lolbin_suspicious": 20.0,
        "hash_changed": 25.0,
    },
    "hygiene": {
        "secure_boot_off": 20.0,
        "defender_and_av_off": 30.0,
        "firewall_off": 25.0,
        "bitlocker_unknown_or_off": 10.0,
        "min_password_zero": 20.0,
    },
    "forensic_visibility": {
        "security_log_unavailable": 25.0,
        "sysmon_absent": 15.0,
        "task_scheduler_op_disabled": 10.0,
        "audit_logon_unknown": 15.0,
        "prefetch_unavailable": 5.0,
    },
}

DEFAULT_OVERALL_BLEND: dict[str, float] = {
    "remote_access": 0.22,
    "persistence": 0.18,
    "account": 0.15,
    "network": 0.12,
    "malware_indicators": 0.15,
    "hygiene": 0.10,
    "forensic_visibility": 0.08,
}


@dataclass
class SecurityConfig:
    enabled: bool = False
    bind_host: str = "127.0.0.1"
    bind_port: int = 5051
    process_interval_sec: int = 45
    persistence_interval_sec: int = 300
    hardening_interval_sec: int = 21600
    daily_report: bool = True
    timezone_display: str = "Europe/Berlin"
    snapshot_retention: int = 60
    event_retention_days: int = 90
    max_event_file_bytes: int = 5_000_000
    max_command_line_chars: int = 400
    hash_cache_enabled: bool = True
    notifications_discord_enabled: bool = False
    notifications_discord_webhook: str = ""
    alert_cooldown_sec: int = 900
    # Trusted path prefixes (user-configured; empty by default)
    authorized_path_prefixes: list[str] = field(default_factory=list)
    trusted_publishers: list[str] = field(
        default_factory=lambda: [
            "Microsoft",
            "Python Software Foundation",
            "Norton",
            "Avast",
            "NVIDIA",
            "ASUS",
            "Gigabyte",
            "Corsair",
            "Logitech",
            "Google",
            "Opera",
            "Discord",
            "Anthropic",
            "Anysphere",  # Cursor
            "GitHub",
            "Node.js",
        ]
    )
    trusted_process_names: list[str] = field(
        default_factory=lambda: [
            "Cursor",
            "cursor",
            "Claude",
            "ollama",
            "python",
            "pythonw",
            "node",
            "git",
            "GitHubDesktop",
            "chrome",
            "opera",
            "Discord",
            "NortonUI",
            "NortonSvc",
            "figma_agent",
        ]
    )
    remote_tool_keywords: list[str] = field(
        default_factory=lambda: [
            "AnyDesk",
            "TeamViewer",
            "RustDesk",
            "ScreenConnect",
            "ConnectWise",
            "UltraVNC",
            "TightVNC",
            "RealVNC",
            "ngrok",
            "cloudflared",
            "Tailscale",
            "ZeroTier",
            "frpc",
            "frps",
            "PsExec",
            "psexec",
            "MeshCentral",
            "Chrome Remote Desktop",
            "remoting_host",
            "Parsec",
            "Sunshine",
            "ToDesk",
            "Supremo",
            "Remote Utilities",
        ]
    )
    lolbins: list[str] = field(
        default_factory=lambda: [
            "powershell",
            "powershell_ise",
            "pwsh",
            "cmd",
            "rundll32",
            "regsvr32",
            "mshta",
            "certutil",
            "bitsadmin",
            "wmic",
            "cscript",
            "wscript",
        ]
    )
    weights: dict[str, dict[str, float]] = field(default_factory=lambda: deepcopy(DEFAULT_WEIGHTS))
    overall_blend: dict[str, float] = field(default_factory=lambda: deepcopy(DEFAULT_OVERALL_BLEND))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecurityConfig":
        base = cls()
        for k, v in data.items():
            if hasattr(base, k):
                setattr(base, k, v)
        if not base.weights:
            base.weights = deepcopy(DEFAULT_WEIGHTS)
        if not base.overall_blend:
            base.overall_blend = deepcopy(DEFAULT_OVERALL_BLEND)
        # Force localhost-only bind — never allow 0.0.0.0 in Phase 1
        if base.bind_host in ("0.0.0.0", "::", "[::]"):
            base.bind_host = "127.0.0.1"
        return base


def config_path(security_root: Path) -> Path:
    return security_root / "config" / "security_config.json"


def load_security_config(security_root: Path) -> SecurityConfig:
    path = config_path(security_root)
    if not path.exists():
        cfg = SecurityConfig()
        save_security_config(security_root, cfg)
        return cfg
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return SecurityConfig()
        return SecurityConfig.from_dict(data)
    except (OSError, json.JSONDecodeError):
        bak = path.with_name(path.name + ".bak")
        if bak.exists():
            try:
                data = json.loads(bak.read_text(encoding="utf-8"))
                return SecurityConfig.from_dict(data)
            except (OSError, json.JSONDecodeError):
                pass
        return SecurityConfig()


def save_security_config(security_root: Path, cfg: SecurityConfig) -> None:
    atomic_write_json(config_path(security_root), cfg.to_dict())
