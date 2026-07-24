"""Risk scoring with documented, configurable weights."""

from __future__ import annotations

from typing import Any

from jarvis.security.config import SecurityConfig
from jarvis.security.models import ScoreCard, ProtectionItem


def _clamp(n: float) -> int:
    return int(max(0, min(100, round(n))))


def compute_scores(
    snapshot: dict[str, Any],
    alerts: list[dict[str, Any]],
    cfg: SecurityConfig,
    *,
    changes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    w = cfg.weights
    cards: dict[str, ScoreCard] = {}

    # Remote access
    ra = 0.0
    ra_ev: list[str] = []
    lim: list[str] = list(snapshot.get("limitations") or [])
    if snapshot.get("remote_tools_detected"):
        ra += w["remote_access"]["remote_tool_present"]
        ra_ev.append(f"remote_tools={len(snapshot['remote_tools_detected'])}")
    rs = snapshot.get("remote_surface") or {}
    if rs.get("rdp_enabled"):
        ra += w["remote_access"]["rdp_enabled"]
        ra_ev.append("rdp_enabled")
    for svc in rs.get("services") or []:
        if str(svc.get("Name")).lower() == "winrm" and str(svc.get("Status")).lower() == "running":
            ra += w["remote_access"]["winrm_running"]
            ra_ev.append("winrm_running")
        if str(svc.get("Name")).lower() == "sshd" and str(svc.get("Status")).lower() == "running":
            ra += w["remote_access"]["ssh_listening"]
            ra_ev.append("sshd_running")
    if snapshot.get("smb_sessions"):
        ra += w["remote_access"]["smb_session_active"]
        ra_ev.append("smb_sessions")
    for a in alerts:
        if a.get("category") == "network" and "all interfaces" in str(a.get("title", "")).lower():
            ra += w["remote_access"]["unknown_listener_all_interfaces"]
            ra_ev.append("unknown_all_iface_listener")
            break
    cards["remote_access"] = ScoreCard(
        "Remote Access Risk",
        _clamp(ra),
        "Risk of remote control / remote entry surfaces",
        ra_ev,
        "medium" if lim else "high",
        lim[:5],
    )

    # Persistence
    pr = 0.0
    pr_ev: list[str] = []
    ch = changes or {}
    kinds_added = {i.get("kind") for i in ch.get("added") or []}
    if "run_key" in kinds_added:
        pr += w["persistence"]["new_run_key"]; pr_ev.append("new_run_key")
    if "task" in kinds_added:
        pr += w["persistence"]["new_task"]; pr_ev.append("new_task")
    if "service" in kinds_added:
        pr += w["persistence"]["new_service"]; pr_ev.append("new_service")
    if any(i.get("kind") == "hosts_file" for i in ch.get("modified") or []):
        pr += w["persistence"]["hosts_modified"]; pr_ev.append("hosts_modified")
    special = snapshot.get("persistence_special") or {}
    if special.get("ifeo"):
        pr += w["persistence"]["ifeo_debugger"]; pr_ev.append("ifeo")
    if special.get("AppInit_DLLs"):
        pr += w["persistence"]["appinit_dll"]; pr_ev.append("appinit")
    shell = str(special.get("Shell") or "explorer.exe")
    if shell.lower() not in ("explorer.exe", ""):
        pr += w["persistence"]["winlogon_shell_changed"]; pr_ev.append("winlogon_shell")
    for f in special.get("wmi_filters") or []:
        if f.get("Name") not in (None, "SCM Event Log Filter"):
            pr += w["persistence"]["wmi_nondefault"]; pr_ev.append("wmi_nondefault"); break
    cards["persistence"] = ScoreCard("Persistence Risk", _clamp(pr), "Persistence / autostart risk", pr_ev, "medium", lim[:3])

    # Account
    ac = 0.0
    ac_ev: list[str] = []
    for u in snapshot.get("users") or []:
        if str(u.get("name", "")).lower() == "administrator" and u.get("password_required") is False:
            ac += w["account"]["blank_password_admin"]; ac_ev.append("admin_blank_password")
        if str(u.get("name", "")).lower() in ("guest", "gast") and u.get("enabled"):
            ac += w["account"]["guest_enabled"]; ac_ev.append("guest_enabled")
    if any(i.get("kind") == "administrator" for i in ch.get("added") or []):
        ac += w["account"]["new_admin"]; ac_ev.append("new_admin")
    if any(i.get("kind") == "user" for i in ch.get("added") or []):
        ac += w["account"]["new_user"]; ac_ev.append("new_user")
    cards["account"] = ScoreCard("Account Risk", _clamp(ac), "Local account / admin risk", ac_ev, "high", [])

    # Network
    nw = 0.0
    nw_ev: list[str] = []
    if any(i.get("kind") == "listening" for i in ch.get("added") or []):
        nw += w["network"]["new_listening_port"]; nw_ev.append("new_listening")
    fw = snapshot.get("firewall") or {}
    profiles = fw.get("profiles") or []
    if profiles and all(not p.get("enabled") for p in profiles):
        nw += w["network"]["firewall_disabled"]; nw_ev.append("firewall_off")
    cards["network"] = ScoreCard("Network Risk", _clamp(nw), "Listening / firewall risk", nw_ev, "medium", lim[:3])

    # Malware indicators
    mw = 0.0
    mw_ev: list[str] = []
    for a in alerts:
        cat = a.get("category")
        title = str(a.get("title", "")).lower()
        if cat == "remote_access" and "tool" in title:
            mw += w["malware_indicators"]["remote_tool_keyword"]; mw_ev.append("remote_tool")
        if "trusted_name" in str(a.get("reason", "")):
            mw += w["malware_indicators"]["trusted_name_bad_path"]; mw_ev.append("name_spoof")
        if "temp" in title and "signature" in title:
            mw += w["malware_indicators"]["unsigned_temp_exe"]; mw_ev.append("temp_unsigned")
    cards["malware_indicators"] = ScoreCard("Malware Indicators", _clamp(mw), "Heuristic malware / tool indicators", mw_ev, "low", ["Absence of indicators is not proof of cleanliness"])

    # Hygiene
    hy = 0.0
    hy_ev: list[str] = []
    sb = snapshot.get("secure_boot") or {}
    if sb.get("UEFISecureBootEnabled") == 0:
        hy += w["hygiene"]["secure_boot_off"]; hy_ev.append("secure_boot_off")
    av = snapshot.get("antivirus") or {}
    if av.get("RealTime") is False and not av.get("running_av_processes"):
        hy += w["hygiene"]["defender_and_av_off"]; hy_ev.append("av_off")
    if profiles and all(not p.get("enabled") for p in profiles):
        hy += w["hygiene"]["firewall_off"]; hy_ev.append("fw_off")
    if any(u.get("password_required") is False and str(u.get("name","")).lower()=="administrator" for u in snapshot.get("users") or []):
        hy += w["hygiene"]["min_password_zero"]; hy_ev.append("weak_password_policy_signal")
    bl = snapshot.get("bitlocker") or {}
    if bl.get("PermissionRequired") or str(bl.get("ProtectionStatus", "")).lower() in ("off", "0", "unprotected"):
        hy += w["hygiene"]["bitlocker_unknown_or_off"] * 0.5; hy_ev.append("bitlocker_unknown_or_off")
    cards["hygiene"] = ScoreCard("Security Hygiene", _clamp(hy), "Hardening posture gaps", hy_ev, "medium", lim[:3])

    # Forensic visibility
    fv = 0.0
    fv_ev: list[str] = []
    elog = snapshot.get("event_logs") or {}
    if isinstance(elog.get("Security"), dict) and elog["Security"].get("Error"):
        fv += w["forensic_visibility"]["security_log_unavailable"]; fv_ev.append("security_log")
    if (snapshot.get("sysmon") or {}).get("Present") is False:
        fv += w["forensic_visibility"]["sysmon_absent"]; fv_ev.append("sysmon_absent")
    tso = elog.get("Microsoft-Windows-TaskScheduler/Operational") or {}
    if isinstance(tso, dict) and tso.get("Enabled") is False:
        fv += w["forensic_visibility"]["task_scheduler_op_disabled"]; fv_ev.append("tasksched_op")
    if (snapshot.get("audit_policy") or {}).get("permission_required"):
        fv += w["forensic_visibility"]["audit_logon_unknown"]; fv_ev.append("audit_unknown")
    cards["forensic_visibility"] = ScoreCard(
        "Forensic Visibility",
        _clamp(fv),
        "Gaps that reduce detection confidence",
        fv_ev,
        "high",
        ["Do not treat low malware score as 'clean' when visibility is poor"],
    )

    # Overall blend (risk)
    overall = 0.0
    blend_ev = []
    for key, weight in cfg.overall_blend.items():
        score = cards[key].score if key in cards else 0
        overall += weight * score
        blend_ev.append(f"{key}={score}×{weight}")
    cards["overall"] = ScoreCard(
        "Overall Security Score (risk)",
        _clamp(overall),
        "Weighted blend of category risks — higher means more concern",
        blend_ev,
        "medium",
        ["Not a certificate of cleanliness", *lim[:3]],
    )

    return {k: v.to_dict() for k, v in cards.items()}


def protection_status(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[ProtectionItem] = []

    def add(name: str, state: str, detail: str = "", perm: bool = False) -> None:
        items.append(ProtectionItem(name, state, detail, perm))

    av = snapshot.get("antivirus") or {}
    if av.get("running_av_processes"):
        add("Antivirus", "Healthy", f"Running: {', '.join(av.get('running_av_processes') or [])}")
    elif av.get("RealTime") is True:
        add("Antivirus", "Healthy", "Defender real-time on")
    elif av.get("PermissionRequired"):
        add("Antivirus", "Permission Required", "Cannot query Defender", True)
    else:
        add("Antivirus", "Critical", "No real-time AV detected")

    fw = snapshot.get("firewall") or {}
    profiles = fw.get("profiles") or []
    if not profiles:
        add("Firewall", "Unknown" if fw.get("permission_required") else "Unknown", "No profile data", bool(fw.get("permission_required")))
    elif all(p.get("enabled") for p in profiles):
        add("Firewall", "Healthy", "All profiles enabled")
    elif any(p.get("enabled") for p in profiles):
        add("Firewall", "Warning", "Some profiles disabled")
    else:
        add("Firewall", "Critical", "All profiles disabled")

    sb = snapshot.get("secure_boot") or {}
    if sb.get("PermissionRequired"):
        add("Secure Boot", "Permission Required", "", True)
    elif sb.get("UEFISecureBootEnabled") == 1:
        add("Secure Boot", "Healthy", "Enabled")
    elif sb.get("UEFISecureBootEnabled") == 0:
        add("Secure Boot", "Critical", "Disabled")
    else:
        add("Secure Boot", "Unknown", "")

    tpm = snapshot.get("tpm") or {}
    if tpm.get("PermissionRequired"):
        add("TPM", "Permission Required", "", True)
    elif tpm.get("Ready"):
        add("TPM", "Healthy", "Ready")
    elif tpm.get("Present"):
        add("TPM", "Warning", "Present but not ready")
    else:
        add("TPM", "Unknown", str(tpm))

    bl = snapshot.get("bitlocker") or {}
    if bl.get("PermissionRequired"):
        add("BitLocker", "Permission Required", "", True)
    else:
        add("BitLocker", "Unknown", str(bl)[:120])

    wu = snapshot.get("windows_update") or {}
    if wu.get("recent_hotfixes"):
        add("Windows Update", "Healthy", f"Recent: {wu['recent_hotfixes'][0].get('HotFixID')}")
    else:
        add("Windows Update", "Unknown", "No hotfix data")

    audit = snapshot.get("audit_policy") or {}
    if audit.get("permission_required"):
        add("Audit Logon", "Permission Required", "", True)
    else:
        add("Audit Logon", "Warning", "Raw policy captured — verify Logon success/failure")

    sysmon = snapshot.get("sysmon") or {}
    if sysmon.get("Present"):
        add("Sysmon", "Healthy", "Present")
    else:
        add("Sysmon", "Warning", "Absent")

    elog = snapshot.get("event_logs") or {}
    tso = elog.get("Microsoft-Windows-TaskScheduler/Operational") or {}
    if isinstance(tso, dict) and tso.get("Enabled") is True:
        add("Task Scheduler Operational", "Healthy", "Enabled")
    elif isinstance(tso, dict) and tso.get("Enabled") is False:
        add("Task Scheduler Operational", "Warning", "Disabled")
    else:
        add("Task Scheduler Operational", "Unknown", str(tso)[:80])

    rs = snapshot.get("remote_surface") or {}
    if rs.get("rdp_enabled") is True:
        add("RDP", "Critical", "Enabled")
    elif rs.get("rdp_enabled") is False:
        add("RDP", "Healthy", "Disabled")
    else:
        add("RDP", "Unknown", "")

    winrm = next((s for s in (rs.get("services") or []) if str(s.get("Name")).lower() == "winrm"), None)
    if winrm and str(winrm.get("Status")).lower() == "running":
        add("WinRM", "Warning", "Running")
    elif winrm:
        add("WinRM", "Healthy", str(winrm.get("Status")))
    else:
        add("WinRM", "Unknown", "")

    sshd = next((s for s in (rs.get("services") or []) if str(s.get("Name")).lower() == "sshd"), None)
    if sshd and str(sshd.get("Status")).lower() == "running":
        add("OpenSSH", "Warning", "sshd running")
    else:
        add("OpenSSH", "Healthy", "Not running / not installed")

    if snapshot.get("smb_sessions"):
        add("SMB sessions", "Warning", f"{len(snapshot['smb_sessions'])} active")
    else:
        add("SMB sessions", "Healthy", "None")

    if snapshot.get("remote_tools_detected"):
        add("Remote tools detected", "Critical", str(len(snapshot["remote_tools_detected"])))
    else:
        add("Remote tools detected", "Healthy", "None matched (not proof of safety)")

    return [i.to_dict() for i in items]
