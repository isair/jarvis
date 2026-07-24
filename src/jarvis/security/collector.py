"""Read-only system collector (Windows-first). Never modifies the host."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from jarvis.security.hash_cache import HashCache
from jarvis.security.models import empty_snapshot
from jarvis.security.redact_ext import redact_command_line, redact_path
from jarvis.security.store import utc_now_iso

PowerShellRunner = Callable[[str, float], tuple[int, str, str]]


def _default_ps_runner(script: str, timeout: float = 30.0) -> tuple[int, str, str]:
    """Run a PowerShell snippet with shell=False, timeout, capped output."""
    # Validate: refuse obvious destructive verbs in our own scripts
    banned = re.compile(r"\b(Remove-Item|Stop-Process|Set-ItemProperty|New-Service|Disable-WindowsOptionalFeature|Format-Volume)\b", re.I)
    if banned.search(script):
        return 1, "", "refused_destructive_pattern"
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        script,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
        out = (proc.stdout or "")[:500_000]
        err = (proc.stderr or "")[:50_000]
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except OSError as exc:
        return 1, "", str(exc)


class WindowsReadOnlyCollector:
    def __init__(
        self,
        *,
        ps_runner: Optional[PowerShellRunner] = None,
        hash_cache: Optional[HashCache] = None,
        max_cmdline: int = 400,
        authorized_prefixes: Optional[list[str]] = None,
    ) -> None:
        self.ps = ps_runner or _default_ps_runner
        self.hash_cache = hash_cache or HashCache()
        self.max_cmdline = max_cmdline
        self.authorized_prefixes = authorized_prefixes or []

    def collect(self) -> dict[str, Any]:
        limitations: list[str] = []
        snap = empty_snapshot(collected_at_utc=utc_now_iso(), host=socket.gethostname(), limitations=limitations)

        if sys.platform != "win32":
            limitations.append("Non-Windows host: collector returns minimal stub")
            return snap

        snap["users"] = self._users(limitations)
        snap["administrators"] = self._administrators(limitations)
        snap["sessions"] = self._sessions(limitations)
        snap["processes"] = self._processes(limitations)
        snap["services"] = self._services(limitations)
        snap["scheduled_tasks"] = self._tasks(limitations)
        snap["startup_items"] = self._startup(limitations)
        snap["run_keys"] = self._run_keys(limitations)
        snap["listening"], snap["connections"] = self._network(limitations)
        snap["smb_shares"], snap["smb_sessions"] = self._smb(limitations)
        snap["remote_surface"] = self._remote_surface(limitations)
        snap["hosts_file"] = self._hosts(limitations)
        snap["dns_cache"] = self._dns(limitations)
        snap["browser_extensions"] = self._browser_extensions(limitations)
        snap["recent_executables"] = self._recent_exes(limitations)
        snap["firewall"] = self._firewall(limitations)
        snap["antivirus"] = self._antivirus(
            limitations,
            process_names=[str(p.get("name") or "") for p in (snap.get("processes") or [])],
        )
        snap["secure_boot"] = self._secure_boot(limitations)
        snap["tpm"] = self._tpm(limitations)
        snap["bitlocker"] = self._bitlocker(limitations)
        snap["windows_update"] = self._windows_update(limitations)
        snap["event_logs"] = self._event_logs(limitations)
        snap["sysmon"] = self._sysmon(limitations)
        snap["audit_policy"] = self._audit_policy(limitations)
        snap["persistence_special"] = self._persistence_special(limitations)
        snap["remote_tools_detected"] = self._remote_tools(snap)
        snap["logon_events"] = self._logon_events(limitations)
        snap["limitations"] = limitations
        return snap

    # ── helpers ───────────────────────────────────────────────
    def _ps_json(self, script: str, limitations: list[str], label: str, timeout: float = 45.0) -> Any:
        code, out, err = self.ps(script, timeout)
        if code != 0:
            limitations.append(f"{label}: permission_or_error ({err.strip()[:120] or code})")
            return None
        text = out.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            limitations.append(f"{label}: invalid_json")
            return None

    def _file_meta(self, path: str) -> dict[str, Any]:
        info: dict[str, Any] = {
            "path": redact_path(path),
            "publisher": "",
            "signature": "Unknown",
            "sha256": "",
        }
        try:
            p = Path(path)
            if not p.is_file():
                return info
            st = p.stat()
            info["sha256"] = self.hash_cache.get_or_compute(p, size=st.st_size, mtime=st.st_mtime)
        except OSError:
            pass
        # Authenticode via PowerShell (best-effort)
        try:
            code, out, _ = self.ps(
                f"$s=Get-AuthenticodeSignature -FilePath {json.dumps(path)}; "
                f"@{{Status=$s.Status.ToString(); Subject=($s.SignerCertificate.Subject)}} | ConvertTo-Json -Compress",
                15.0,
            )
            if code == 0 and out.strip():
                data = json.loads(out)
                info["signature"] = str(data.get("Status") or "Unknown")
                subj = str(data.get("Subject") or "")
                # keep CN only
                m = re.search(r"CN=([^,]+)", subj)
                info["publisher"] = m.group(1) if m else subj[:80]
        except Exception:
            pass
        return info

    # ── sections ──────────────────────────────────────────────
    def _users(self, lim: list[str]) -> list[dict[str, Any]]:
        data = self._ps_json(
            "Get-LocalUser | Select-Object Name,Enabled,PasswordRequired,"
            "@{N='PasswordLastSet';E={$_.PasswordLastSet}},"
            "@{N='LastLogon';E={$_.LastLogon}} | ConvertTo-Json -Compress",
            lim,
            "users",
        )
        if data is None:
            return []
        if isinstance(data, dict):
            data = [data]
        out = []
        for u in data:
            out.append(
                {
                    "name": u.get("Name"),
                    "enabled": bool(u.get("Enabled")),
                    "password_required": bool(u.get("PasswordRequired")),
                    "password_last_set": str(u.get("PasswordLastSet") or ""),
                    "last_logon": str(u.get("LastLogon") or ""),
                }
            )
        return out

    def _administrators(self, lim: list[str]) -> list[str]:
        # Language-independent well-known SID for Administrators
        data = self._ps_json(
            "$ErrorActionPreference='Stop'; "
            "try { Get-LocalGroupMember -SID 'S-1-5-32-544' | Select-Object -ExpandProperty Name | ConvertTo-Json -Compress } "
            "catch { '[]' }",
            lim,
            "admins_sid",
        )
        if data:
            if isinstance(data, str):
                return [data]
            if isinstance(data, list):
                return [str(x) for x in data]
        # Fallback localized display names (EN/DE/RO/FR)
        for group in ("Administrators", "Administratoren", "Administratori", "Administrateurs"):
            data = self._ps_json(
                "$ErrorActionPreference='Stop'; "
                f"try {{ Get-LocalGroupMember -Group {json.dumps(group)} | Select-Object -ExpandProperty Name | ConvertTo-Json -Compress }} "
                "catch { '[]' }",
                lim,
                f"admins_{group}",
            )
            if data:
                if isinstance(data, str):
                    return [data]
                if isinstance(data, list):
                    return [str(x) for x in data]
        return []

    def _sessions(self, lim: list[str]) -> list[dict[str, Any]]:
        code, out, err = self.ps("quser", 10.0)
        if code != 0 and not out:
            lim.append(f"sessions: {err[:80]}")
            return []
        rows = []
        for line in out.splitlines():
            if "USERNAME" in line or not line.strip() or line.strip().startswith("---"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                rows.append({"raw": " ".join(parts[:6]), "user": parts[0].lstrip(">")})
        return rows

    def _processes(self, lim: list[str]) -> list[dict[str, Any]]:
        data = self._ps_json(
            "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine | "
            "ConvertTo-Json -Compress -Depth 3",
            lim,
            "processes",
            timeout=60.0,
        )
        if not data:
            return []
        if isinstance(data, dict):
            data = [data]
        out = []
        # Cap process enrichment for performance
        enrich_budget = 40
        for p in data:
            path = p.get("ExecutablePath") or ""
            item = {
                "pid": p.get("ProcessId"),
                "ppid": p.get("ParentProcessId"),
                "name": p.get("Name"),
                "path": redact_path(path),
                "command_line": redact_command_line(p.get("CommandLine") or "", self.max_cmdline),
                "publisher": "",
                "signature": "Unknown",
                "sha256": "",
            }
            if path and enrich_budget > 0 and (
                re.search(r"\\(Temp|Downloads|AppData)\\", path, re.I)
                or path.lower().endswith((".exe", ".dll"))
            ):
                meta = self._file_meta(path)
                item.update({k: meta[k] for k in ("publisher", "signature", "sha256") if k in meta})
                enrich_budget -= 1
            out.append(item)
        return out

    def _services(self, lim: list[str]) -> list[dict[str, Any]]:
        data = self._ps_json(
            "Get-CimInstance Win32_Service | Select-Object Name,DisplayName,State,StartMode,PathName | ConvertTo-Json -Compress",
            lim,
            "services",
            timeout=45.0,
        )
        if not data:
            return []
        if isinstance(data, dict):
            data = [data]
        return [
            {
                "name": s.get("Name"),
                "display": s.get("DisplayName"),
                "state": s.get("State"),
                "start_mode": s.get("StartMode"),
                "path": redact_path(s.get("PathName") or ""),
            }
            for s in data
        ]

    def _tasks(self, lim: list[str]) -> list[dict[str, Any]]:
        data = self._ps_json(
            "$tasks=Get-ScheduledTask -EA SilentlyContinue | ForEach-Object {"
            "$a=($_.Actions | ForEach-Object { $_.Execute + ' ' + $_.Arguments }) -join '; ';"
            "[PSCustomObject]@{TaskPath=$_.TaskPath;TaskName=$_.TaskName;State=$_.State.ToString();Author=$_.Author;Action=$a}"
            "}; $tasks | ConvertTo-Json -Compress",
            lim,
            "tasks",
            timeout=60.0,
        )
        if not data:
            return []
        if isinstance(data, dict):
            data = [data]
        return [
            {
                "path": t.get("TaskPath"),
                "name": t.get("TaskName"),
                "state": t.get("State"),
                "author": t.get("Author") or "",
                "action": redact_command_line(t.get("Action") or "", 300),
            }
            for t in data
        ]

    def _startup(self, lim: list[str]) -> list[dict[str, Any]]:
        folders = [
            os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs\Startup"),
            os.path.join(os.environ.get("ProgramData", ""), r"Microsoft\Windows\Start Menu\Programs\Startup"),
        ]
        items = []
        for folder in folders:
            try:
                for p in Path(folder).glob("*"):
                    if p.name.lower() == "desktop.ini":
                        continue
                    items.append({"folder": redact_path(folder), "name": p.name, "path": redact_path(str(p))})
            except OSError as exc:
                lim.append(f"startup:{exc}")
        return items

    def _run_keys(self, lim: list[str]) -> dict[str, dict[str, str]]:
        script = r"""
$keys=@(
 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run',
 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce',
 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Run',
 'HKLM:\Software\Microsoft\Windows\CurrentVersion\RunOnce',
 'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Run'
)
$out=@{}
foreach($k in $keys){
  if(Test-Path $k){
    $props=Get-ItemProperty $k -EA SilentlyContinue
    $map=@{}
    $props.PSObject.Properties | Where-Object { $_.Name -notmatch '^PS' } | ForEach-Object { $map[$_.Name]=[string]$_.Value }
    $out[$k]=$map
  }
}
$out | ConvertTo-Json -Compress -Depth 4
"""
        data = self._ps_json(script, lim, "run_keys")
        if not isinstance(data, dict):
            return {}
        # redact values
        cleaned: dict[str, dict[str, str]] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                cleaned[k] = {str(nk): redact_command_line(str(nv), 300) for nk, nv in v.items()}
        return cleaned

    def _network(self, lim: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        data = self._ps_json(
            "$listen=Get-NetTCPConnection -State Listen -EA SilentlyContinue | ForEach-Object {"
            "$p=Get-Process -Id $_.OwningProcess -EA SilentlyContinue;"
            "[PSCustomObject]@{Address=$_.LocalAddress;Port=$_.LocalPort;PID=$_.OwningProcess;Process=$p.ProcessName;Path=$p.Path}"
            "};"
            "$est=Get-NetTCPConnection -State Established -EA SilentlyContinue | Select-Object -First 80 | ForEach-Object {"
            "$p=Get-Process -Id $_.OwningProcess -EA SilentlyContinue;"
            "[PSCustomObject]@{LAddress=$_.LocalAddress;LPort=$_.LocalPort;RAddress=$_.RemoteAddress;RPort=$_.RemotePort;PID=$_.OwningProcess;Process=$p.ProcessName;Path=$p.Path}"
            "};"
            "@{listening=@($listen);established=@($est)} | ConvertTo-Json -Compress -Depth 4",
            lim,
            "network",
            timeout=45.0,
        )
        if not isinstance(data, dict):
            return [], []
        listening = []
        for x in data.get("listening") or []:
            if isinstance(x, dict):
                listening.append(
                    {
                        "address": x.get("Address"),
                        "port": x.get("Port"),
                        "pid": x.get("PID"),
                        "process": x.get("Process"),
                        "path": redact_path(x.get("Path") or ""),
                    }
                )
        established = []
        for x in data.get("established") or []:
            if isinstance(x, dict):
                established.append(
                    {
                        "local_address": x.get("LAddress"),
                        "local_port": x.get("LPort"),
                        "remote_address": x.get("RAddress"),
                        "remote_port": x.get("RPort"),
                        "pid": x.get("PID"),
                        "process": x.get("Process"),
                        "path": redact_path(x.get("Path") or ""),
                    }
                )
        return listening, established

    def _smb(self, lim: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        shares = self._ps_json(
            "Get-SmbShare -EA SilentlyContinue | Select-Object Name,Path,CurrentUsers | ConvertTo-Json -Compress",
            lim,
            "smb_shares",
        )
        sessions = self._ps_json(
            "Get-SmbSession -EA SilentlyContinue | Select-Object ClientComputerName,ClientUserName,NumOpens | ConvertTo-Json -Compress",
            lim,
            "smb_sessions",
        )
        def norm(d):
            if d is None:
                return []
            if isinstance(d, dict):
                return [d]
            return d if isinstance(d, list) else []
        return [
            {"name": s.get("Name"), "path": redact_path(s.get("Path") or ""), "current_users": s.get("CurrentUsers")}
            for s in norm(shares)
        ], [
            {
                "client": s.get("ClientComputerName"),
                "user": s.get("ClientUserName"),
                "opens": s.get("NumOpens"),
            }
            for s in norm(sessions)
        ]

    def _remote_surface(self, lim: list[str]) -> dict[str, Any]:
        data = self._ps_json(
            "$rdp=(Get-ItemProperty 'HKLM:\\System\\CurrentControlSet\\Control\\Terminal Server' -Name fDenyTSConnections -EA SilentlyContinue).fDenyTSConnections;"
            "$svcs=@(); Get-Service TermService,WinRM,sshd,RemoteRegistry,RemoteAccess,UmRdpService -EA SilentlyContinue | ForEach-Object {"
            "  $svcs += @{Name=$_.Name; Status=$_.Status.ToString(); StartType=$_.StartType.ToString()} "
            "}; @{fDenyTSConnections=$rdp; services=$svcs} | ConvertTo-Json -Compress -Depth 5",
            lim,
            "remote_surface",
        )
        if not isinstance(data, dict):
            return {"fDenyTSConnections": None, "services": [], "permission_required": True}
        services = data.get("services")
        if isinstance(services, dict):
            # PowerShell sometimes wraps arrays
            if "value" in services and isinstance(services["value"], list):
                services = services["value"]
            else:
                services = [services]
        if not isinstance(services, list):
            services = []
        return {
            "fDenyTSConnections": data.get("fDenyTSConnections"),
            "rdp_enabled": data.get("fDenyTSConnections") == 0,
            "services": services,
        }

    def _hosts(self, lim: list[str]) -> dict[str, Any]:
        path = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "drivers" / "etc" / "hosts"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            extras = []
            for line in text.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if re.match(r"^(127\.0\.0\.1|::1)\s+localhost\b", s, re.I):
                    continue
                extras.append(s)
            digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
            return {"path": str(path), "extra_lines": extras, "sha256": digest}
        except OSError as exc:
            lim.append(f"hosts:{exc}")
            return {"path": str(path), "extra_lines": [], "sha256": "", "error": str(exc)}

    def _dns(self, lim: list[str]) -> list[dict[str, Any]]:
        data = self._ps_json(
            "Get-DnsClientCache -EA SilentlyContinue | Select-Object -First 40 Entry,Data | ConvertTo-Json -Compress",
            lim,
            "dns",
        )
        if not data:
            return []
        if isinstance(data, dict):
            data = [data]
        return [{"entry": d.get("Entry"), "data": d.get("Data")} for d in data]

    def _browser_extensions(self, lim: list[str]) -> list[dict[str, Any]]:
        # Lightweight: list extension IDs only (no private data)
        roots = []
        local = os.environ.get("LOCALAPPDATA", "")
        roaming = os.environ.get("APPDATA", "")
        roots.append(("Chrome", Path(local) / "Google" / "Chrome" / "User Data" / "Default" / "Extensions"))
        roots.append(("Edge", Path(local) / "Microsoft" / "Edge" / "User Data" / "Default" / "Extensions"))
        opera = Path(roaming) / "Opera Software"
        if opera.is_dir():
            for child in opera.iterdir():
                ext = child / "Default" / "Extensions"
                if ext.is_dir():
                    roots.append(("Opera", ext))
                    break
        out = []
        for browser, root in roots:
            try:
                if not root.is_dir():
                    continue
                for ext_id in root.iterdir():
                    if ext_id.is_dir():
                        out.append({"browser": browser, "extension_id": ext_id.name})
            except OSError as exc:
                lim.append(f"extensions_{browser}:{exc}")
        return out

    def _recent_exes(self, lim: list[str]) -> list[dict[str, Any]]:
        candidates: list[Path] = []
        home = Path.home()
        for folder in (
            home / "Downloads",
            Path(os.environ.get("TEMP", "")),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Temp",
        ):
            try:
                if folder.is_dir():
                    for p in folder.glob("*.exe"):
                        candidates.append(p)
            except OSError:
                continue
        # sort by mtime, take 25
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)[:25]
        out = []
        for p in candidates:
            try:
                st = p.stat()
                meta = self._file_meta(str(p))
                out.append(
                    {
                        "path": redact_path(str(p)),
                        "size": st.st_size,
                        "mtime": st.st_mtime,
                        "signature": meta.get("signature"),
                        "publisher": meta.get("publisher"),
                        "sha256": meta.get("sha256"),
                    }
                )
            except OSError:
                continue
        return out

    def _firewall(self, lim: list[str]) -> dict[str, Any]:
        data = self._ps_json(
            "Get-NetFirewallProfile | Select-Object Name,Enabled | ConvertTo-Json -Compress",
            lim,
            "firewall",
        )
        if not data:
            return {"profiles": [], "permission_required": True}
        if isinstance(data, dict):
            data = [data]
        return {"profiles": [{"name": p.get("Name"), "enabled": bool(p.get("Enabled"))} for p in data]}

    def _antivirus(self, lim: list[str], process_names: Optional[list[str]] = None) -> dict[str, Any]:
        data = self._ps_json(
            "try { $s=Get-MpComputerStatus; @{AntivirusEnabled=$s.AntivirusEnabled;RealTime=$s.RealTimeProtectionEnabled;PermissionRequired=$false} | ConvertTo-Json -Compress } "
            "catch { @{AntivirusEnabled=$null;RealTime=$null;PermissionRequired=$true;Error=$_.Exception.Message} | ConvertTo-Json -Compress }",
            lim,
            "antivirus",
        )
        # Prefer already-collected process names (avoids Get-Process multi-name exit quirks)
        av_markers = ("nortonsvc", "nortonui", "nortonsecurity", "aswengsrv", "avastsvc", "msmpeng", "securityhealthservice")
        procs: list[str] = []
        for raw in process_names or []:
            base = str(raw or "").lower().replace(".exe", "")
            if base in av_markers:
                procs.append(str(raw))
        if not procs:
            code, out, _ = self.ps(
                "Get-Process | Where-Object { $_.ProcessName -match 'Norton|aswEng|MsMpEng|Avast' } | "
                "Select-Object -ExpandProperty ProcessName -Unique | ConvertTo-Json -Compress",
                15.0,
            )
            if code == 0 and out.strip():
                try:
                    parsed = json.loads(out)
                    procs = [parsed] if isinstance(parsed, str) else [str(x) for x in (parsed or [])]
                except json.JSONDecodeError:
                    pass
        result = data if isinstance(data, dict) else {"permission_required": True}
        result["running_av_processes"] = sorted(set(procs))
        return result

    def _secure_boot(self, lim: list[str]) -> dict[str, Any]:
        data = self._ps_json(
            "try { @{UEFISecureBootEnabled=(Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\SecureBoot\\State' -EA Stop).UEFISecureBootEnabled;PermissionRequired=$false} | ConvertTo-Json -Compress } "
            "catch { @{UEFISecureBootEnabled=$null;PermissionRequired=$true} | ConvertTo-Json -Compress }",
            lim,
            "secure_boot",
        )
        return data if isinstance(data, dict) else {"permission_required": True}

    def _tpm(self, lim: list[str]) -> dict[str, Any]:
        data = self._ps_json(
            "try { $t=Get-Tpm; @{Present=$t.TpmPresent;Ready=$t.TpmReady;PermissionRequired=$false} | ConvertTo-Json -Compress } "
            "catch { @{Present=$null;Ready=$null;PermissionRequired=$true} | ConvertTo-Json -Compress }",
            lim,
            "tpm",
        )
        return data if isinstance(data, dict) else {"permission_required": True}

    def _bitlocker(self, lim: list[str]) -> dict[str, Any]:
        data = self._ps_json(
            "try { Get-BitLockerVolume -MountPoint C: | Select-Object MountPoint,VolumeStatus,ProtectionStatus | ConvertTo-Json -Compress } "
            "catch { @{PermissionRequired=$true;Error=$_.Exception.Message} | ConvertTo-Json -Compress }",
            lim,
            "bitlocker",
        )
        return data if isinstance(data, dict) else {"permission_required": True}

    def _windows_update(self, lim: list[str]) -> dict[str, Any]:
        data = self._ps_json(
            "Get-HotFix | Sort-Object InstalledOn -Descending | Select-Object -First 5 HotFixID,Description,@{N='InstalledOn';E={$_.InstalledOn}} | ConvertTo-Json -Compress",
            lim,
            "windows_update",
        )
        if not data:
            return {"recent_hotfixes": [], "permission_required": True}
        if isinstance(data, dict):
            data = [data]
        return {"recent_hotfixes": data}

    def _event_logs(self, lim: list[str]) -> dict[str, Any]:
        data = self._ps_json(
            "$names=@('Security','System','Application','Microsoft-Windows-TaskScheduler/Operational','Microsoft-Windows-Windows Defender/Operational');"
            "$out=@{}; foreach($n in $names){ try { $l=Get-WinEvent -ListLog $n -EA Stop; $out[$n]=@{Enabled=$l.IsEnabled;Records=$l.RecordCount} } catch { $out[$n]=@{Enabled=$false;Error=$_.Exception.Message} } };"
            "$out | ConvertTo-Json -Compress",
            lim,
            "event_logs",
        )
        return data if isinstance(data, dict) else {}

    def _sysmon(self, lim: list[str]) -> dict[str, Any]:
        data = self._ps_json(
            "try { Get-WinEvent -ListLog 'Microsoft-Windows-Sysmon/Operational' -EA Stop | ForEach-Object { @{Present=$true;Enabled=$_.IsEnabled;Records=$_.RecordCount} } | ConvertTo-Json -Compress } "
            "catch { @{Present=$false;Error=$_.Exception.Message} | ConvertTo-Json -Compress }",
            lim,
            "sysmon",
        )
        return data if isinstance(data, dict) else {"Present": False}

    def _audit_policy(self, lim: list[str]) -> dict[str, Any]:
        code, out, err = self.ps("auditpol /get /category:*", 15.0)
        if code != 0:
            lim.append("audit_policy: permission_required")
            return {"permission_required": True, "raw_excerpt": ""}
        return {"permission_required": False, "raw_excerpt": out[:2000]}

    def _persistence_special(self, lim: list[str]) -> dict[str, Any]:
        data = self._ps_json(
            "$ifeo=@(); Get-ChildItem 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Image File Execution Options' -EA SilentlyContinue | ForEach-Object {"
            "$d=(Get-ItemProperty $_.PSPath -Name Debugger -EA SilentlyContinue).Debugger; if($d){ $ifeo += @{Name=$_.PSChildName;Debugger=$d} }"
            "};"
            "$app=Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Windows' -Name AppInit_DLLs,LoadAppInit_DLLs -EA SilentlyContinue;"
            "$wl=Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Winlogon' -Name Shell,Userinit -EA SilentlyContinue;"
            "$wmi=@(); try { Get-CimInstance -Namespace root\\subscription -ClassName __EventFilter -EA Stop | ForEach-Object { $wmi += @{Name=$_.Name;Query=$_.Query} } } catch {};"
            "@{ifeo=$ifeo;AppInit_DLLs=$app.AppInit_DLLs;LoadAppInit_DLLs=$app.LoadAppInit_DLLs;Shell=$wl.Shell;Userinit=$wl.Userinit;wmi_filters=$wmi} | ConvertTo-Json -Compress -Depth 5",
            lim,
            "persistence_special",
        )
        return data if isinstance(data, dict) else {}

    def _logon_events(self, lim: list[str]) -> list[dict[str, Any]]:
        data = self._ps_json(
            "try { Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-TerminalServices-LocalSessionManager/Operational'; Id=21,23; StartTime=(Get-Date).AddDays(-7)} -MaxEvents 20 -EA Stop | "
            "ForEach-Object { [PSCustomObject]@{Time=$_.TimeCreated.ToString('o');Id=$_.Id;Message=($_.Message -replace '\\s+',' ').Substring(0,[Math]::Min(160,$_.Message.Length))} } | ConvertTo-Json -Compress } "
            "catch { '[]' }",
            lim,
            "logon_events",
        )
        if not data:
            return []
        if isinstance(data, dict):
            return [data]
        return data if isinstance(data, list) else []

    def _remote_tools(self, snap: dict[str, Any]) -> list[dict[str, Any]]:
        keywords = [
            "AnyDesk", "TeamViewer", "RustDesk", "ScreenConnect", "ConnectWise", "UltraVNC",
            "TightVNC", "ngrok", "cloudflared", "Tailscale", "ZeroTier", "frpc", "PsExec",
            "MeshCentral", "remoting_host", "Parsec", "Sunshine", "ToDesk",
        ]
        found = []
        blob_sources = []
        for p in snap.get("processes", []):
            blob_sources.append((p.get("name") or "") + " " + (p.get("path") or ""))
        for s in snap.get("services", []):
            blob_sources.append((s.get("name") or "") + " " + (s.get("path") or ""))
        for t in snap.get("scheduled_tasks", []):
            blob_sources.append((t.get("name") or "") + " " + (t.get("action") or ""))
        for kw in keywords:
            for blob in blob_sources:
                if kw.lower() in blob.lower():
                    found.append({"keyword": kw, "evidence": redact_path(blob)[:200]})
                    break
        return found
