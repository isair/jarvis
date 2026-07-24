"""Trust / attribution engine — never trust by process name alone."""

from __future__ import annotations

import re
from typing import Any

from jarvis.security.config import SecurityConfig


def classify_process(proc: dict[str, Any], cfg: SecurityConfig, *, baseline_hashes: set[str] | None = None) -> dict[str, Any]:
    """Return trust classification with reasons."""
    name = (proc.get("name") or "").lower().replace(".exe", "")
    path = (proc.get("path") or "")
    path_l = path.lower()
    publisher = (proc.get("publisher") or "").lower()
    signature = (proc.get("signature") or "").lower()
    sha = (proc.get("sha256") or "").lower()
    cmdline = (proc.get("command_line") or "").lower()

    reasons: list[str] = []
    level = "unknown"

    name_trusted = any(name == t.lower().replace(".exe", "") or name.startswith(t.lower()) for t in cfg.trusted_process_names)
    pub_trusted = any(tp.lower() in publisher for tp in cfg.trusted_publishers if publisher)
    sig_ok = signature == "valid"

    in_authorized_path = False
    for prefix in cfg.authorized_path_prefixes:
        if prefix and prefix.lower().replace("/", "\\") in path_l.replace("/", "\\"):
            in_authorized_path = True
            break

    # Built-in safe-ish Windows / vendor install paths (no personal usernames)
    if re.search(r"\\windows\\(system32|syswow64)\\", path_l):
        in_authorized_path = True
        reasons.append("system_path")
    if re.search(r"\\program files( \(x86\))?\\", path_l):
        in_authorized_path = True
        reasons.append("program_files_path")

    # Cora/Jarvis heuristics without hardcoding username
    if "cora" in path_l or "jarvis" in path_l or "coralauncher" in path_l:
        in_authorized_path = True
        reasons.append("cora_jarvis_path")
    if "\\cursor\\" in path_l or path_l.endswith("\\cursor.exe"):
        in_authorized_path = True
        reasons.append("cursor_path")
    if "\\norton\\" in path_l or "\\avast\\" in path_l:
        in_authorized_path = True
        reasons.append("av_vendor_path")

    # Well-known system process names without path (often PID 4 / protected)
    _system_names = {
        "system",
        "svchost",
        "services",
        "lsass",
        "wininit",
        "spoolsv",
        "smss",
        "csrss",
        "winlogon",
        "registry",
        "memory compression",
    }
    if name in _system_names and not path:
        level = "trusted"
        reasons.append("system_process_name")
        return {
            "trust_level": level,
            "reasons": reasons,
            "name_trusted": True,
            "publisher_trusted": pub_trusted,
            "signature_ok": sig_ok,
            "authorized_path": True,
        }

    hash_known = bool(sha and baseline_hashes and sha in baseline_hashes)

    # Suspicious: trusted name but weird location (Temp/Downloads/AppData roaming drop)
    weird_drop = bool(re.search(r"\\(temp|downloads|appdata\\local\\temp)\\", path_l))
    if name_trusted and path and not in_authorized_path and not pub_trusted and weird_drop:
        level = "suspicious"
        reasons.append("trusted_name_untrusted_path")
    elif name_trusted and path and not in_authorized_path and not pub_trusted and not weird_drop:
        # Name matches allowlist but path is unusual — keep unknown, not auto-critical
        level = "unknown"
        reasons.append("trusted_name_unfamiliar_path")
    elif name_trusted and signature in ("notsigned", "hashmismatch", "not signed") and weird_drop:
        level = "suspicious"
        reasons.append("trusted_name_bad_signature")
    elif in_authorized_path and (pub_trusted or sig_ok or name_trusted or "program_files_path" in reasons or "av_vendor_path" in reasons):
        level = "authorized_project" if ("cora" in path_l or "jarvis" in path_l or "cursor" in path_l) else "trusted"
        reasons.append("path_and_identity_ok")
    elif pub_trusted and sig_ok:
        level = "trusted"
        reasons.append("publisher_and_signature")
    elif name_trusted and in_authorized_path:
        level = "trusted"
        reasons.append("trusted_name_authorized_path")
    elif hash_known:
        level = "baseline_known"
        reasons.append("baseline_hash")
    elif re.search(r"\\(temp|downloads)\\", path_l) and signature != "valid":
        level = "suspicious"
        reasons.append("unsigned_or_unknown_temp_path")
    elif not path:
        level = "unknown"
        reasons.append("missing_path")

    # Remote tool keyword in name/path/cmd
    for kw in cfg.remote_tool_keywords:
        if kw.lower() in name or kw.lower() in path_l or kw.lower() in cmdline:
            level = "critical"
            reasons.append(f"remote_tool_keyword:{kw}")
            break

    return {
        "trust_level": level,
        "reasons": reasons,
        "name_trusted": name_trusted,
        "publisher_trusted": pub_trusted,
        "signature_ok": sig_ok,
        "authorized_path": in_authorized_path,
    }
