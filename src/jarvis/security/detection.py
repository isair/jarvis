"""Detection engine + baseline diff (read-only analysis)."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Optional

from jarvis.security.config import SecurityConfig
from jarvis.security.models import SecurityAlert
from jarvis.security.store import utc_now_iso
from jarvis.security.trust import classify_process


def _fid(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:24]


def _alert(**kwargs: Any) -> dict[str, Any]:
    a = SecurityAlert(
        id=kwargs.pop("id", uuid.uuid4().hex),
        timestamp_utc=kwargs.pop("timestamp_utc", utc_now_iso()),
        **kwargs,
    )
    return a.to_dict()


def diff_baseline(baseline: Optional[dict[str, Any]], current: dict[str, Any]) -> dict[str, Any]:
    """Compare current snapshot to baseline metadata view."""
    if not baseline or "snapshot" not in baseline:
        return {"status": "no_baseline", "added": [], "removed": [], "modified": []}

    b = baseline["snapshot"]
    changes: dict[str, list[dict[str, Any]]] = {"added": [], "removed": [], "modified": []}

    def set_of(items: list[dict[str, Any]], key: str) -> set[str]:
        return {str(i.get(key) or "") for i in items if i.get(key)}

    # Users
    bu, cu = set_of(b.get("users", []), "name"), set_of(current.get("users", []), "name")
    for n in sorted(cu - bu):
        changes["added"].append({"kind": "user", "name": n})
    for n in sorted(bu - cu):
        changes["removed"].append({"kind": "user", "name": n})

    # Admins
    ba, ca = set(b.get("administrators") or []), set(current.get("administrators") or [])
    for n in sorted(ca - ba):
        changes["added"].append({"kind": "administrator", "name": n})
    for n in sorted(ba - ca):
        changes["removed"].append({"kind": "administrator", "name": n})

    # Services by name
    bs = {s.get("name"): s for s in b.get("services", []) if s.get("name")}
    cs = {s.get("name"): s for s in current.get("services", []) if s.get("name")}
    for n in sorted(set(cs) - set(bs)):
        changes["added"].append({"kind": "service", "name": n, "path": cs[n].get("path")})
    for n in sorted(set(bs) - set(cs)):
        changes["removed"].append({"kind": "service", "name": n})
    for n in sorted(set(bs) & set(cs)):
        if (bs[n].get("path") or "") != (cs[n].get("path") or "") or (bs[n].get("start_mode") or "") != (cs[n].get("start_mode") or ""):
            changes["modified"].append({"kind": "service", "name": n})

    # Tasks
    def task_key(t: dict[str, Any]) -> str:
        return f"{t.get('path') or ''}|{t.get('name') or ''}"

    bt = {task_key(t): t for t in b.get("scheduled_tasks", [])}
    ct = {task_key(t): t for t in current.get("scheduled_tasks", [])}
    for k in sorted(set(ct) - set(bt)):
        changes["added"].append({"kind": "task", "key": k, "action": (ct[k].get("action") or "")[:120]})
    for k in sorted(set(bt) - set(ct)):
        changes["removed"].append({"kind": "task", "key": k})
    for k in sorted(set(bt) & set(ct)):
        if (bt[k].get("action") or "") != (ct[k].get("action") or ""):
            changes["modified"].append({"kind": "task", "key": k})

    # Run keys
    br, cr = b.get("run_keys") or {}, current.get("run_keys") or {}
    for hive in set(br) | set(cr):
        bm, cm = br.get(hive) or {}, cr.get(hive) or {}
        if not isinstance(bm, dict):
            bm = {}
        if not isinstance(cm, dict):
            cm = {}
        for name in sorted(set(cm) - set(bm)):
            changes["added"].append({"kind": "run_key", "hive": hive, "name": name, "value": cm[name]})
        for name in sorted(set(bm) - set(cm)):
            changes["removed"].append({"kind": "run_key", "hive": hive, "name": name})
        for name in sorted(set(bm) & set(cm)):
            if bm[name] != cm[name]:
                changes["modified"].append({"kind": "run_key", "hive": hive, "name": name})

    # Hosts
    bh = (b.get("hosts_file") or {}).get("sha256")
    ch = (current.get("hosts_file") or {}).get("sha256")
    if bh and ch and bh != ch:
        changes["modified"].append({"kind": "hosts_file", "old_sha": bh, "new_sha": ch})

    # Listening ports
    bl = {f"{x.get('address')}:{x.get('port')}" for x in b.get("listening", [])}
    cl = {f"{x.get('address')}:{x.get('port')}" for x in current.get("listening", [])}
    for p in sorted(cl - bl):
        changes["added"].append({"kind": "listening", "endpoint": p})
    for p in sorted(bl - cl):
        changes["removed"].append({"kind": "listening", "endpoint": p})

    return {"status": "ok", "baseline_created_at_utc": baseline.get("created_at_utc"), **changes}


def detect_alerts(
    snapshot: dict[str, Any],
    cfg: SecurityConfig,
    *,
    baseline: Optional[dict[str, Any]] = None,
    changes: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Produce alert dicts with evidence. Does not mutate the OS."""
    alerts: list[dict[str, Any]] = []
    baseline_hashes: set[str] = set()
    if baseline and isinstance(baseline.get("snapshot"), dict):
        for p in baseline["snapshot"].get("processes_indexed") or []:
            if p.get("sha256"):
                baseline_hashes.add(str(p["sha256"]).lower())

    # Annotate processes with trust
    for proc in snapshot.get("processes", []):
        trust = classify_process(proc, cfg, baseline_hashes=baseline_hashes)
        proc["trust"] = trust
        if trust["trust_level"] in ("suspicious", "critical"):
            sev = "critical" if trust["trust_level"] == "critical" else "high"
            fp = _fid("proc", str(proc.get("path")), str(proc.get("sha256")), trust["trust_level"])
            alerts.append(
                _alert(
                    fingerprint=fp,
                    severity=sev,
                    category="process",
                    title="Process classification raised",
                    artifact=str(proc.get("name") or ""),
                    path=str(proc.get("path") or ""),
                    publisher=str(proc.get("publisher") or ""),
                    signature=str(proc.get("signature") or ""),
                    sha256=str(proc.get("sha256") or ""),
                    reason="; ".join(trust["reasons"]),
                    evidence=[f"trust={trust['trust_level']}", f"pid={proc.get('pid')}"],
                    recommendation="Manual investigation — read-only phase (no auto action)",
                )
            )

    # Remote tools
    for rt in snapshot.get("remote_tools_detected") or []:
        fp = _fid("remote", str(rt.get("keyword")), str(rt.get("evidence")))
        alerts.append(
            _alert(
                fingerprint=fp,
                severity="critical",
                category="remote_access",
                title=f"Remote access tool indicator: {rt.get('keyword')}",
                artifact=str(rt.get("keyword")),
                reason="Keyword match in process/service/task metadata",
                evidence=[str(rt.get("evidence") or "")],
                recommendation="Confirm whether this tool is authorized; investigate if unknown",
            )
        )

    # Account hygiene
    for u in snapshot.get("users") or []:
        if str(u.get("name", "")).lower() == "administrator" and u.get("enabled") and u.get("password_required") is False:
            alerts.append(
                _alert(
                    fingerprint=_fid("acct", "admin_blank"),
                    severity="critical",
                    category="account",
                    title="Administrator account does not require a password",
                    artifact="Administrator",
                    reason="PasswordRequired=False",
                    evidence=["Local user policy indicates blank-password allowance for Administrator"],
                    recommendation="Set a strong password and require passwords for local admins",
                )
            )

    # RDP enabled
    rs = snapshot.get("remote_surface") or {}
    if rs.get("rdp_enabled") is True:
        alerts.append(
            _alert(
                fingerprint=_fid("rdp", "enabled"),
                severity="high",
                category="remote_access",
                title="Remote Desktop appears enabled",
                reason="fDenyTSConnections=0",
                evidence=[str(rs)],
                recommendation="Disable RDP if not required; restrict with firewall + NLA",
            )
        )

    # Firewall all off
    fw = snapshot.get("firewall") or {}
    profiles = fw.get("profiles") or []
    if profiles and all(not p.get("enabled") for p in profiles):
        alerts.append(
            _alert(
                fingerprint=_fid("fw", "all_off"),
                severity="critical",
                category="network",
                title="Windows Firewall disabled on all profiles",
                reason="All firewall profiles Enabled=False",
                evidence=[str(profiles)],
                recommendation="Re-enable firewall profiles",
            )
        )

    # Antivirus off + no third-party AV procs
    av = snapshot.get("antivirus") or {}
    if av.get("RealTime") is False and not av.get("running_av_processes"):
        alerts.append(
            _alert(
                fingerprint=_fid("av", "off"),
                severity="high",
                category="hygiene",
                title="No real-time antivirus protection detected",
                reason="Defender RealTime=False and no known AV processes",
                evidence=[str(av)],
                recommendation="Enable Norton/Defender real-time protection",
            )
        )

    # Secure Boot off
    sb = snapshot.get("secure_boot") or {}
    if sb.get("UEFISecureBootEnabled") == 0:
        alerts.append(
            _alert(
                fingerprint=_fid("sb", "off"),
                severity="medium",
                category="hygiene",
                title="Secure Boot is disabled",
                reason="UEFISecureBootEnabled=0",
                evidence=[str(sb)],
                recommendation="Enable Secure Boot in firmware if hardware supports it",
            )
        )

    # Hosts extras
    hosts = snapshot.get("hosts_file") or {}
    extras = hosts.get("extra_lines") or []
    if extras:
        alerts.append(
            _alert(
                fingerprint=_fid("hosts", hosts.get("sha256") or "x"),
                severity="medium",
                category="persistence",
                title="Non-default hosts file entries present",
                reason="Custom DNS overrides detected",
                evidence=list(extras)[:10],
                recommendation="Verify each hosts entry is intentional",
            )
        )

    # Persistence special
    special = snapshot.get("persistence_special") or {}
    if special.get("ifeo"):
        alerts.append(
            _alert(
                fingerprint=_fid("ifeo", str(special.get("ifeo"))),
                severity="critical",
                category="persistence",
                title="IFEO Debugger entries present",
                evidence=[str(special.get("ifeo"))[:500]],
                recommendation="Investigate Image File Execution Options debuggers",
            )
        )
    if special.get("AppInit_DLLs"):
        alerts.append(
            _alert(
                fingerprint=_fid("appinit", str(special.get("AppInit_DLLs"))),
                severity="critical",
                category="persistence",
                title="AppInit_DLLs is non-empty",
                evidence=[str(special.get("AppInit_DLLs"))],
                recommendation="Investigate AppInit DLL injection",
            )
        )
    shell = str(special.get("Shell") or "")
    if shell and shell.lower() not in ("explorer.exe", ""):
        alerts.append(
            _alert(
                fingerprint=_fid("winlogon", shell),
                severity="critical",
                category="persistence",
                title="Winlogon Shell is not explorer.exe",
                evidence=[shell],
                recommendation="Restore Winlogon Shell to explorer.exe if unexpected",
            )
        )
    for f in special.get("wmi_filters") or []:
        name = str(f.get("Name") or "")
        if name and name != "SCM Event Log Filter":
            alerts.append(
                _alert(
                    fingerprint=_fid("wmi", name),
                    severity="high",
                    category="persistence",
                    title=f"Non-default WMI event filter: {name}",
                    evidence=[str(f.get("Query") or "")[:300]],
                    recommendation="Review WMI subscription persistence",
                )
            )

    # Unsigned temp/downloads executables
    for exe in snapshot.get("recent_executables") or []:
        sig = str(exe.get("signature") or "").lower()
        path = str(exe.get("path") or "")
        if "temp" in path.lower() and sig not in ("valid",):
            alerts.append(
                _alert(
                    fingerprint=_fid("temp_exe", path, str(exe.get("sha256"))),
                    severity="medium",
                    category="malware_indicators",
                    title="Executable in Temp without valid signature",
                    path=path,
                    signature=str(exe.get("signature") or ""),
                    sha256=str(exe.get("sha256") or ""),
                    recommendation="Quarantine only after manual review (not automated in Phase 1)",
                )
            )

    # Listening on all interfaces from unknown process
    for lst in snapshot.get("listening") or []:
        addr = str(lst.get("address") or "")
        if addr in ("0.0.0.0", "::"):
            # find process trust if present in snapshot processes
            path = str(lst.get("path") or "")
            proc = {"name": lst.get("process"), "path": path, "publisher": "", "signature": "", "sha256": "", "command_line": ""}
            trust = classify_process(proc, cfg, baseline_hashes=baseline_hashes)
            if trust["trust_level"] in ("unknown", "suspicious", "critical"):
                alerts.append(
                    _alert(
                        fingerprint=_fid("listen", addr, str(lst.get("port")), path),
                        severity="high",
                        category="network",
                        title=f"Unknown process listening on all interfaces :{lst.get('port')}",
                        artifact=str(lst.get("process") or ""),
                        path=path,
                        connection=f"{addr}:{lst.get('port')}",
                        reason="; ".join(trust["reasons"]) or "untrusted listener",
                        evidence=[str(lst)],
                        recommendation="Identify the service; bind to localhost if possible",
                    )
                )

    # Diff-driven alerts
    ch = changes or {}
    for item in ch.get("added") or []:
        kind = item.get("kind")
        if kind == "administrator":
            alerts.append(
                _alert(
                    fingerprint=_fid("new_admin", str(item.get("name"))),
                    severity="critical",
                    category="account",
                    title=f"New local administrator: {item.get('name')}",
                    evidence=[str(item)],
                    recommendation="Verify this admin change was intentional",
                )
            )
        elif kind == "user":
            alerts.append(
                _alert(
                    fingerprint=_fid("new_user", str(item.get("name"))),
                    severity="high",
                    category="account",
                    title=f"New local user: {item.get('name')}",
                    evidence=[str(item)],
                    recommendation="Verify new account",
                )
            )
        elif kind in ("service", "task", "run_key", "listening"):
            alerts.append(
                _alert(
                    fingerprint=_fid("added", kind, str(item)),
                    severity="medium",
                    category="persistence" if kind != "listening" else "network",
                    title=f"Baseline addition: {kind}",
                    evidence=[str(item)],
                    recommendation="Compare with expected software installs",
                )
            )
    for item in ch.get("modified") or []:
        if item.get("kind") == "hosts_file":
            alerts.append(
                _alert(
                    fingerprint=_fid("hosts_mod", str(item.get("new_sha"))),
                    severity="high",
                    category="persistence",
                    title="Hosts file changed since baseline",
                    evidence=[str(item)],
                    recommendation="Review hosts extras",
                )
            )

    # Sysmon / forensic visibility soft alerts
    sysmon = snapshot.get("sysmon") or {}
    if sysmon.get("Present") is False:
        alerts.append(
            _alert(
                fingerprint=_fid("sysmon", "absent"),
                severity="low",
                category="forensic_visibility",
                title="Sysmon is not installed",
                evidence=[str(sysmon)],
                recommendation="Consider installing Sysmon for better telemetry (manual)",
            )
        )
    elog = snapshot.get("event_logs") or {}
    sec = elog.get("Security") or {}
    if isinstance(sec, dict) and sec.get("Error"):
        alerts.append(
            _alert(
                fingerprint=_fid("seclog", "unavailable"),
                severity="medium",
                category="forensic_visibility",
                title="Security event log unavailable or inaccessible",
                evidence=[str(sec)],
                recommendation="Run elevated and enable logon auditing",
            )
        )

    # Deduplicate by fingerprint (keep highest severity)
    sev_rank = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    by_fp: dict[str, dict[str, Any]] = {}
    for a in alerts:
        fp = a.get("fingerprint") or a.get("id")
        prev = by_fp.get(fp)
        if not prev or sev_rank.get(a.get("severity", ""), 0) >= sev_rank.get(prev.get("severity", ""), 0):
            by_fp[fp] = a
    return list(by_fp.values())
