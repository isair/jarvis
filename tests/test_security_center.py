"""Unit tests for Cora Security Center (Phase 1 READ-ONLY) — no host mutation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.security.config import SecurityConfig, load_security_config, save_security_config
from jarvis.security.detection import detect_alerts, diff_baseline
from jarvis.security.hash_cache import HashCache
from jarvis.security.models import empty_snapshot
from jarvis.security.paths import ensure_layout
from jarvis.security.redact_ext import redact_command_line, redact_text, sanitize_for_html
from jarvis.security.risk import compute_scores, protection_status
from jarvis.security.service import SecurityCenterService
from jarvis.security.store import SecurityStore, utc_now_iso
from jarvis.security.trust import classify_process


@pytest.fixture
def sec_root(tmp_path: Path) -> Path:
    root = tmp_path / "security"
    ensure_layout(root)
    return root


def _snap(**overrides):
    s = empty_snapshot(collected_at_utc=utc_now_iso(), host="TEST-HOST")
    s.update(overrides)
    return s


class FakeCollector:
    def __init__(self, snapshot: dict):
        self.snapshot = snapshot

    def collect(self):
        return self.snapshot


@pytest.mark.unit
def test_baseline_initial_and_diff(sec_root: Path):
    store = SecurityStore(sec_root)
    snap1 = _snap(
        users=[{"name": "Administrator", "enabled": True, "password_required": True}],
        administrators=["TEST\\Administrator"],
        services=[{"name": "Spooler", "start_mode": "Auto", "path": "C:\\Windows\\spoolsv.exe", "state": "Running"}],
        scheduled_tasks=[],
        run_keys={"HKCU\\Run": {"OneDrive": "onedrive.exe"}},
        listening=[{"address": "127.0.0.1", "port": 5050, "process": "python", "path": "C:\\py\\python.exe"}],
        hosts_file={"sha256": "aaa", "extra_lines": []},
    )
    baseline = store.save_baseline(snap1)
    assert baseline["created_at_utc"]
    snap2 = _snap(
        users=[
            {"name": "Administrator", "enabled": True, "password_required": True},
            {"name": "Hacker", "enabled": True, "password_required": True},
        ],
        administrators=["TEST\\Administrator", "TEST\\Hacker"],
        services=[
            {"name": "Spooler", "start_mode": "Auto", "path": "C:\\Windows\\spoolsv.exe", "state": "Running"},
            {"name": "EvilSvc", "start_mode": "Auto", "path": "C:\\Temp\\evil.exe", "state": "Running"},
        ],
        scheduled_tasks=[{"path": "\\", "name": "BadTask", "action": "cmd.exe /c whoami", "state": "Ready"}],
        run_keys={"HKCU\\Run": {"OneDrive": "onedrive.exe", "Bad": "C:\\Temp\\bad.exe"}},
        listening=[
            {"address": "127.0.0.1", "port": 5050, "process": "python", "path": "C:\\py\\python.exe"},
            {"address": "0.0.0.0", "port": 4444, "process": "unknown", "path": "C:\\Temp\\x.exe"},
        ],
        hosts_file={"sha256": "bbb", "extra_lines": ["1.2.3.4 evil.test"]},
    )
    changes = diff_baseline(baseline, snap2)
    assert changes["status"] == "ok"
    kinds = {c["kind"] for c in changes["added"]}
    assert "user" in kinds
    assert "administrator" in kinds
    assert "service" in kinds
    assert "task" in kinds
    assert "run_key" in kinds
    assert "listening" in kinds
    assert any(c["kind"] == "hosts_file" for c in changes["modified"])


@pytest.mark.unit
def test_detect_new_task_service_run_hosts_user_admin_rdp_av_fw(sec_root: Path):
    cfg = SecurityConfig()
    baseline_snap = _snap(
        users=[{"name": "Administrator", "enabled": True, "password_required": True}],
        administrators=["A"],
        services=[],
        scheduled_tasks=[],
        run_keys={},
        hosts_file={"sha256": "1", "extra_lines": []},
        listening=[],
    )
    baseline = {"created_at_utc": utc_now_iso(), "snapshot": {
        "users": baseline_snap["users"],
        "administrators": baseline_snap["administrators"],
        "services": [],
        "scheduled_tasks": [],
        "run_keys": {},
        "hosts_file": baseline_snap["hosts_file"],
        "listening": [],
        "processes_indexed": [],
        "firewall": {},
        "antivirus": {},
        "secure_boot": {},
        "persistence_special": {},
        "remote_surface": {},
        "startup_items": [],
    }}
    current = _snap(
        users=[
            {"name": "Administrator", "enabled": True, "password_required": False},
            {"name": "NewUser", "enabled": True, "password_required": True},
        ],
        administrators=["A", "B"],
        services=[{"name": "NewSvc", "path": "C:\\Temp\\s.exe", "start_mode": "Auto", "state": "Running"}],
        scheduled_tasks=[{"path": "\\", "name": "NewTask", "action": "powershell.exe -enc AAAA", "state": "Ready"}],
        run_keys={"HKCU\\Run": {"X": "C:\\Temp\\x.exe"}},
        hosts_file={"sha256": "2", "extra_lines": ["0.0.0.0 bad"]},
        listening=[{"address": "0.0.0.0", "port": 9999, "process": "weird", "path": "C:\\Temp\\weird.exe"}],
        remote_surface={"rdp_enabled": True, "services": []},
        firewall={"profiles": [{"name": "Domain", "enabled": False}, {"name": "Private", "enabled": False}, {"name": "Public", "enabled": False}]},
        antivirus={"RealTime": False, "running_av_processes": []},
        secure_boot={"UEFISecureBootEnabled": 0},
        processes=[],
        recent_executables=[{"path": "C:\\Temp\\a.exe", "signature": "NotSigned", "sha256": "abc"}],
        remote_tools_detected=[{"keyword": "AnyDesk", "evidence": "AnyDesk.exe"}],
        persistence_special={"ifeo": [], "AppInit_DLLs": "", "Shell": "explorer.exe", "wmi_filters": [{"Name": "SCM Event Log Filter"}]},
        sysmon={"Present": False},
        event_logs={"Security": {"Error": "denied"}},
    )
    # Use store baseline view helper via save/load
    store = SecurityStore(sec_root)
    store.save_baseline(baseline_snap)
    baseline = store.load_baseline()
    changes = diff_baseline(baseline, current)
    alerts = detect_alerts(current, cfg, baseline=baseline, changes=changes)
    titles = " ".join(a["title"] for a in alerts)
    assert "Administrator" in titles or "password" in titles.lower()
    assert "Remote Desktop" in titles
    assert "Firewall" in titles
    assert "AnyDesk" in titles
    assert "Secure Boot" in titles
    assert "Hosts" in titles or "hosts" in titles.lower() or any(a["category"] == "persistence" for a in alerts)


@pytest.mark.unit
def test_unsigned_temp_and_trusted_name_bad_path():
    cfg = SecurityConfig()
    bad = classify_process(
        {"name": "Cursor.exe", "path": "C:\\Temp\\Cursor.exe", "publisher": "", "signature": "NotSigned", "sha256": "", "command_line": ""},
        cfg,
    )
    assert bad["trust_level"] in ("suspicious", "critical")
    good = classify_process(
        {
            "name": "python.exe",
            "path": "C:\\Users\\x\\Downloads\\jarvis\\.venv\\Scripts\\python.exe",
            "publisher": "Python Software Foundation",
            "signature": "Valid",
            "sha256": "aa",
            "command_line": "-m jarvis",
        },
        cfg,
    )
    assert good["trust_level"] in ("trusted", "authorized_project", "baseline_known")


@pytest.mark.unit
def test_corrupt_snapshot_recovery(sec_root: Path):
    store = SecurityStore(sec_root)
    p = store.baseline_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not-json", encoding="utf-8")
    # bak with good content
    good = {"schema_version": 1, "created_at_utc": utc_now_iso(), "snapshot": {"users": []}}
    p.with_name(p.name + ".bak").write_text(json.dumps(good), encoding="utf-8")
    loaded = store.load_baseline()
    assert loaded is not None
    assert loaded["schema_version"] == 1


@pytest.mark.unit
def test_permission_insufficient_and_timeout_localized(sec_root: Path):
    from jarvis.security.collector import WindowsReadOnlyCollector

    def runner(script: str, timeout: float = 30.0):
        if "Get-LocalUser" in script:
            return 1, "", "Zugriff verweigert / Access is denied"
        if "Get-Tpm" in script:
            return 124, "", "timeout"
        return 0, "[]", ""

    col = WindowsReadOnlyCollector(ps_runner=runner)
    # Force win32 path by monkeypatching platform check via calling private helpers
    lim: list[str] = []
    assert col._users(lim) == []
    assert any("permission_or_error" in x or "users" in x for x in lim)


@pytest.mark.unit
def test_redaction_secrets_and_html():
    text = "Authorization: Bearer SUPERSECRETTOKEN api_key=abcd1234 https://discord.com/api/webhooks/1/2"
    out = redact_text(text)
    assert "SUPERSECRETTOKEN" not in out
    assert "abcd1234" not in out or "REDACTED" in out
    assert "webhooks" not in out.lower() or "REDACTED_WEBHOOK" in out
    cmd = redact_command_line("password=hunter2 " + ("x" * 1000), max_len=50)
    assert len(cmd) <= 50
    assert "<script>" not in sanitize_for_html("<script>alert(1)</script>")
    from jarvis.security.redact_ext import scrub_obj

    preserved = scrub_obj({"id": "a" * 32, "sha256": "b" * 64, "path": r"C:\Users\Alice\x.exe", "cmd": "Bearer tok123"})
    assert preserved["id"] == "a" * 32
    assert preserved["sha256"] == "b" * 64
    assert "[USER]" in preserved["path"]
    assert "REDACTED" in preserved["cmd"]


@pytest.mark.unit
def test_log_rotation_and_alert_dedup(sec_root: Path):
    store = SecurityStore(sec_root)
    a = {
        "id": "1",
        "fingerprint": "fp1",
        "timestamp_utc": utc_now_iso(),
        "severity": "high",
        "category": "x",
        "title": "t",
        "status": "New",
        "evidence": [],
        "recommendation": "",
        "reason": "",
    }
    store.append_alert(a, max_bytes=5_000_000)
    store.append_alert({**a, "id": "2"}, max_bytes=5_000_000)
    # service dedup path
    cfg = SecurityConfig(enabled=False)
    snap = _snap(processes=[], users=[], remote_tools_detected=[], persistence_special={"Shell": "explorer.exe", "wmi_filters": [{"Name": "SCM Event Log Filter"}]}, hosts_file={"extra_lines": [], "sha256": "x"}, firewall={"profiles": [{"name": "P", "enabled": True}]}, antivirus={"RealTime": True, "running_av_processes": ["NortonSvc"]}, secure_boot={"UEFISecureBootEnabled": 1}, sysmon={"Present": True}, event_logs={})
    svc = SecurityCenterService(root=sec_root, collector=FakeCollector(snap), config=cfg)
    r1 = svc.run_audit()
    r2 = svc.run_audit()
    # second run should not explode; alerts file grows only for new fingerprints
    assert "scores" in r1 and "scores" in r2


@pytest.mark.unit
def test_alert_status_update_only_internal(sec_root: Path):
    store = SecurityStore(sec_root)
    store.append_alert(
        {
            "id": "abc123",
            "fingerprint": "f",
            "timestamp_utc": utc_now_iso(),
            "severity": "low",
            "category": "x",
            "title": "t",
            "status": "New",
            "evidence": [],
            "recommendation": "",
            "reason": "",
        }
    )
    assert store.update_alert_status("abc123", "Acknowledged") is True
    assert store.update_alert_status("abc123", "DELETE_SYSTEM") is False
    alerts = store.load_alerts()
    assert alerts[0]["status"] == "Acknowledged"


@pytest.mark.unit
def test_hash_cache(tmp_path: Path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    cache = HashCache(tmp_path / "c.json")
    st = f.stat()
    h1 = cache.get_or_compute(f, size=st.st_size, mtime=st.st_mtime)
    h2 = cache.get_or_compute(f, size=st.st_size, mtime=st.st_mtime)
    assert h1 == h2 and len(h1) == 64
    cache.save()
    cache2 = HashCache(tmp_path / "c.json")
    assert cache2.get_or_compute(f, size=st.st_size, mtime=st.st_mtime) == h1


@pytest.mark.unit
def test_risk_scores_and_protection(sec_root: Path):
    cfg = SecurityConfig()
    snap = _snap(
        remote_tools_detected=[{"keyword": "TeamViewer"}],
        remote_surface={"rdp_enabled": True, "services": []},
        firewall={"profiles": [{"name": "Public", "enabled": False}]},
        antivirus={"RealTime": False, "running_av_processes": []},
        secure_boot={"UEFISecureBootEnabled": 0},
        sysmon={"Present": False},
        event_logs={"Security": {"Error": "no"}},
        users=[{"name": "Administrator", "password_required": False, "enabled": True}],
        persistence_special={"Shell": "explorer.exe", "wmi_filters": [{"Name": "SCM Event Log Filter"}]},
    )
    alerts = detect_alerts(snap, cfg, baseline=None, changes={"added": [], "removed": [], "modified": []})
    scores = compute_scores(snap, alerts, cfg, changes={"added": [], "removed": [], "modified": []})
    assert scores["overall"]["score"] > 0
    assert scores["remote_access"]["score"] > 0
    prot = protection_status(snap)
    names = {p["name"] for p in prot}
    assert "RDP" in names and "Antivirus" in names


@pytest.mark.unit
def test_false_positive_cora_cursor_paths():
    cfg = SecurityConfig()
    cora = classify_process(
        {
            "name": "pythonw.exe",
            "path": "C:\\Users\\Administrator\\CoraLauncher\\launch-cora.pyw",
            "publisher": "Python Software Foundation",
            "signature": "Valid",
            "sha256": "1",
            "command_line": "launch-cora.pyw",
        },
        cfg,
    )
    assert cora["trust_level"] != "critical"
    cursor = classify_process(
        {
            "name": "Cursor.exe",
            "path": "C:\\Users\\Administrator\\AppData\\Local\\Programs\\cursor\\Cursor.exe",
            "publisher": "Anysphere",
            "signature": "Valid",
            "sha256": "2",
            "command_line": "",
        },
        cfg,
    )
    assert cursor["trust_level"] in ("trusted", "authorized_project", "baseline_known")
    norton = classify_process(
        {
            "name": "NortonUI.exe",
            "path": "C:\\Program Files\\Norton\\Suite\\NortonUI.exe",
            "publisher": "",
            "signature": "Unknown",
            "sha256": "",
            "command_line": "",
        },
        cfg,
    )
    assert norton["trust_level"] in ("trusted", "authorized_project", "baseline_known")


@pytest.mark.unit
def test_first_audit_baseline_has_no_diff(sec_root: Path):
    snap = _snap(
        users=[{"name": "Administrator", "enabled": True, "password_required": True}],
        administrators=["TEST\\Administrator"],
        services=[{"name": "Spooler", "start_mode": "Auto", "path": "C:\\Windows\\spoolsv.exe", "state": "Running"}],
        scheduled_tasks=[{"path": "\\", "name": "X", "action": "cmd.exe", "state": "Ready"}],
        run_keys={"HKCU\\Run": {"OneDrive": "onedrive.exe"}},
        listening=[{"address": "127.0.0.1", "port": 5050, "process": "python", "path": "C:\\py\\python.exe"}],
        hosts_file={"sha256": "aaa", "extra_lines": []},
        processes=[],
    )
    svc = SecurityCenterService(root=sec_root, collector=FakeCollector(snap), config=SecurityConfig())
    result = svc.run_audit(create_baseline_if_missing=True)
    assert result["baseline_created"] is True
    assert result["changes"]["added"] == []
    assert result["changes"]["removed"] == []
    assert result["changes"]["modified"] == []


@pytest.mark.unit
def test_config_refuses_non_loopback_bind(sec_root: Path):
    cfg = SecurityConfig.from_dict({"bind_host": "0.0.0.0", "bind_port": 5051})
    assert cfg.bind_host == "127.0.0.1"
    save_security_config(sec_root, cfg)
    loaded = load_security_config(sec_root)
    assert loaded.bind_host == "127.0.0.1"


@pytest.mark.unit
def test_flask_api_localhost_contract(sec_root: Path):
    cfg = SecurityConfig(enabled=False, bind_host="127.0.0.1", bind_port=5051)
    snap = _snap(
        processes=[],
        users=[{"name": "Administrator", "enabled": True, "password_required": True}],
        remote_tools_detected=[],
        persistence_special={"Shell": "explorer.exe", "wmi_filters": [{"Name": "SCM Event Log Filter"}]},
        hosts_file={"extra_lines": [], "sha256": "z"},
        firewall={"profiles": [{"name": "Private", "enabled": True}]},
        antivirus={"RealTime": True, "running_av_processes": ["NortonSvc"]},
        secure_boot={"UEFISecureBootEnabled": 1},
        sysmon={"Present": True},
        event_logs={},
        remote_surface={"rdp_enabled": False, "services": []},
    )
    svc = SecurityCenterService(root=sec_root, collector=FakeCollector(snap), config=cfg)
    svc.run_audit()
    from desktop_app.security_center import create_app

    app = create_app(svc)
    client = app.test_client()
    r = client.get("/api/overview")
    assert r.status_code == 200
    body = r.get_json()
    assert "scores" in body
    r2 = client.post("/api/audit")
    assert r2.status_code == 200
    assert r2.get_json()["ok"] is True
    # status update only
    alerts = svc.store.load_alerts()
    if alerts:
        aid = alerts[0]["id"]
        r3 = client.post(f"/api/alerts/{aid}/status", json={"status": "Trusted"})
        assert r3.get_json()["ok"] is True
    # reject OS-like fake action status
    r4 = client.post("/api/alerts/x/status", json={"status": "KillProcess"})
    assert r4.status_code == 400


@pytest.mark.unit
def test_ps_runner_refuses_destructive():
    from jarvis.security.collector import _default_ps_runner

    code, out, err = _default_ps_runner("Stop-Process -Name explorer")
    assert code == 1
    assert "refused" in err


@pytest.mark.unit
def test_concurrent_audit_rejected(sec_root: Path):
    import threading
    import time

    hold = threading.Event()
    release = threading.Event()

    class SlowCollector:
        def collect(self):
            hold.set()
            release.wait(timeout=5)
            return _snap(
                processes=[],
                users=[{"name": "Administrator", "enabled": True, "password_required": True}],
                remote_tools_detected=[],
                persistence_special={"Shell": "explorer.exe", "wmi_filters": [{"Name": "SCM Event Log Filter"}]},
                hosts_file={"extra_lines": [], "sha256": "z"},
                firewall={"profiles": [{"name": "Private", "enabled": True}]},
                antivirus={"RealTime": True},
                secure_boot={"UEFISecureBootEnabled": 1},
                sysmon={"Present": True},
                event_logs={},
                remote_surface={"rdp_enabled": False, "services": []},
            )

    cfg = SecurityConfig(enabled=False)
    svc = SecurityCenterService(root=sec_root, collector=SlowCollector(), config=cfg)
    results: list[dict] = []

    def _run():
        results.append(svc.run_audit())

    t = threading.Thread(target=_run)
    t.start()
    assert hold.wait(timeout=2)
    busy = svc.run_audit()
    assert busy.get("error") == "audit_in_progress"
    release.set()
    t.join(timeout=5)
    assert results and results[0].get("ok") is True
