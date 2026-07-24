"""
Cora Security Center — local Flask dashboard (127.0.0.1 only).

Run: python -m desktop_app.security_center
Phase 1: READ-ONLY. Alert status updates touch only local JSONL store.
"""

from __future__ import annotations

import html
from typing import Any, Optional

from flask import Flask, Response, jsonify, request

from jarvis.security.models import ALERT_STATUSES
from jarvis.security.paths import default_security_root
from jarvis.security.redact_ext import sanitize_for_html
from jarvis.security.service import SecurityCenterService

app = Flask(__name__)
_service: Optional[SecurityCenterService] = None


def get_service() -> SecurityCenterService:
    global _service
    if _service is None:
        _service = SecurityCenterService(root=default_security_root())
    return _service


def create_app(service: Optional[SecurityCenterService] = None) -> Flask:
    global _service
    if service is not None:
        _service = service
    return app


@app.after_request
def _security_headers(resp: Response) -> Response:
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def index() -> Response:
    return Response(DASHBOARD_HTML, mimetype="text/html; charset=utf-8")


@app.route("/api/overview")
def api_overview() -> Response:
    return jsonify(get_service().overview())


@app.route("/api/alerts")
def api_alerts() -> Response:
    limit = min(int(request.args.get("limit", 100)), 500)
    return jsonify({"alerts": get_service().store.load_alerts(limit=limit)})


@app.route("/api/alerts/<alert_id>/status", methods=["POST"])
def api_alert_status(alert_id: str) -> Response:
    # Only allow status mutation in our store — never OS actions
    data = request.get_json(silent=True) or {}
    status = str(data.get("status") or "")
    if status not in ALERT_STATUSES:
        return jsonify({"ok": False, "error": "invalid_status"}), 400
    # sanitize id
    safe_id = "".join(c for c in alert_id if c.isalnum())[:64]
    ok = get_service().store.update_alert_status(safe_id, status)
    return jsonify({"ok": ok})


@app.route("/api/audit", methods=["POST"])
def api_audit() -> Response:
    # Manual audit — still read-only vs OS; rejects concurrent runs
    result = get_service().run_audit(create_baseline_if_missing=True)
    if result.get("error") == "audit_in_progress":
        return jsonify({"ok": False, "error": "audit_in_progress"}), 409
    # Don't dump full snapshot in response (size); return summary
    return jsonify(
        {
            "ok": True,
            "baseline_created": result.get("baseline_created"),
            "scores": result.get("scores"),
            "alerts_count": len(result.get("alerts") or []),
            "changes": {
                "status": (result.get("changes") or {}).get("status"),
                "added": len((result.get("changes") or {}).get("added") or []),
                "removed": len((result.get("changes") or {}).get("removed") or []),
                "modified": len((result.get("changes") or {}).get("modified") or []),
            },
            "report_path": result.get("report_path"),
            "protection": result.get("protection"),
            "limitations": (result.get("snapshot") or {}).get("limitations") or [],
        }
    )


@app.route("/api/changes")
def api_changes() -> Response:
    svc = get_service()
    latest = svc.store.load_latest_snapshot()
    baseline = svc.store.load_baseline()
    from jarvis.security.detection import diff_baseline

    return jsonify(diff_baseline(baseline, latest) if latest else {"status": "no_snapshot"})


@app.route("/api/processes")
def api_processes() -> Response:
    latest = get_service().store.load_latest_snapshot() or {}
    procs = latest.get("processes") or []
    # Prefer those with trust annotation
    return jsonify({"processes": procs[:300]})


@app.route("/api/network")
def api_network() -> Response:
    latest = get_service().store.load_latest_snapshot() or {}
    return jsonify({"listening": latest.get("listening") or [], "established": latest.get("connections") or []})


@app.route("/api/reports")
def api_reports() -> Response:
    return jsonify({"reports": get_service().store.list_reports()})


@app.route("/api/reports/<name>")
def api_report(name: str) -> Response:
    text = get_service().store.read_report(name)
    if text is None:
        return jsonify({"error": "not_found"}), 404
    # Return as plain text escaped for JSON
    return jsonify({"name": html.escape(name), "markdown": text})


@app.route("/api/protection")
def api_protection() -> Response:
    ov = get_service().overview()
    return jsonify({"protection": ov.get("protection") or []})


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Cora Security Center</title>
<style>
:root { --bg:#0f1115; --card:#1a1f2b; --text:#e8eaed; --muted:#9aa0a6; --ok:#34d399; --warn:#fbbf24; --crit:#f87171; --accent:#60a5fa; }
*{box-sizing:border-box} body{margin:0;font-family:Segoe UI,system-ui,sans-serif;background:var(--bg);color:var(--text)}
header{padding:16px 24px;border-bottom:1px solid #2a3142;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{margin:0;font-size:20px} .badge{padding:4px 10px;border-radius:999px;background:#243047;font-size:12px;color:var(--muted)}
nav{display:flex;gap:8px;padding:12px 24px;flex-wrap:wrap}
nav button{background:#243047;border:0;color:var(--text);padding:8px 12px;border-radius:8px;cursor:pointer}
nav button.active{background:var(--accent);color:#081018}
main{padding:16px 24px 48px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
.card{background:var(--card);border-radius:12px;padding:14px}
.card h3{margin:0 0 6px;font-size:13px;color:var(--muted);font-weight:600}
.card .val{font-size:28px;font-weight:700}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px;border-bottom:1px solid #2a3142;text-align:left;vertical-align:top}
.sev-critical{color:var(--crit)} .sev-high{color:#fb923c} .sev-medium{color:var(--warn)} .sev-low{color:var(--ok)}
.state-Healthy{color:var(--ok)} .state-Warning{color:var(--warn)} .state-Critical{color:var(--crit)} .state-Unknown,.state-Permission\ Required{color:var(--muted)}
.row{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
button.primary{background:var(--accent);border:0;color:#081018;padding:10px 14px;border-radius:8px;font-weight:600;cursor:pointer}
pre{white-space:pre-wrap;background:#0b0d12;padding:12px;border-radius:8px;max-height:480px;overflow:auto}
.note{color:var(--muted);font-size:12px;margin-top:8px}
</style>
</head>
<body>
<header>
  <h1>🛡️ Cora Security Center</h1>
  <span class="badge">READ-ONLY Phase 1</span>
  <span class="badge" id="hostBadge">—</span>
  <span class="badge" id="auditBadge">Last audit: —</span>
</header>
<nav>
  <button class="active" data-tab="overview">Overview</button>
  <button data-tab="protection">Protection</button>
  <button data-tab="alerts">Alerts</button>
  <button data-tab="changes">Changes</button>
  <button data-tab="processes">Processes</button>
  <button data-tab="network">Network</button>
  <button data-tab="reports">Reports</button>
</nav>
<main>
  <section id="overview" class="tab">
    <div class="row">
      <button class="primary" id="runAudit">Run Manual Audit</button>
      <span class="note" id="auditStatus"></span>
    </div>
    <div class="grid" id="scoreGrid"></div>
    <p class="note">Higher scores mean more risk. Low malware score ≠ proof the system is clean.</p>
  </section>
  <section id="protection" class="tab" hidden>
    <table><thead><tr><th>Control</th><th>State</th><th>Detail</th></tr></thead><tbody id="protBody"></tbody></table>
  </section>
  <section id="alerts" class="tab" hidden>
    <table><thead><tr><th>Sev</th><th>Title</th><th>Category</th><th>Status</th><th>Actions</th></tr></thead><tbody id="alertBody"></tbody></table>
  </section>
  <section id="changes" class="tab" hidden>
    <pre id="changesPre"></pre>
  </section>
  <section id="processes" class="tab" hidden>
    <table><thead><tr><th>PID</th><th>Name</th><th>Trust</th><th>Path</th><th>Sig</th></tr></thead><tbody id="procBody"></tbody></table>
  </section>
  <section id="network" class="tab" hidden>
    <h3>Listening</h3>
    <table><thead><tr><th>Endpoint</th><th>Process</th><th>Path</th></tr></thead><tbody id="listenBody"></tbody></table>
    <h3>Established (sample)</h3>
    <table><thead><tr><th>Remote</th><th>Process</th><th>Path</th></tr></thead><tbody id="estBody"></tbody></table>
  </section>
  <section id="reports" class="tab" hidden>
    <div id="reportList"></div>
    <pre id="reportView"></pre>
  </section>
</main>
<script>
const $ = (s)=>document.querySelector(s);
const esc = (s)=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
document.querySelectorAll('nav button').forEach(btn=>{
  btn.onclick=()=>{
    document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.tab').forEach(t=>t.hidden=true);
    $('#'+btn.dataset.tab).hidden=false;
    if(btn.dataset.tab==='alerts') loadAlerts();
    if(btn.dataset.tab==='changes') loadChanges();
    if(btn.dataset.tab==='processes') loadProcs();
    if(btn.dataset.tab==='network') loadNet();
    if(btn.dataset.tab==='reports') loadReports();
    if(btn.dataset.tab==='protection') loadProt();
  };
});
async function loadOverview(){
  const o=await (await fetch('/api/overview')).json();
  $('#auditBadge').textContent='Last audit: '+(o.last_audit_utc||'never');
  const scores=o.scores||{};
  const order=['overall','remote_access','hygiene','forensic_visibility','persistence','account','network','malware_indicators'];
  $('#scoreGrid').innerHTML=order.filter(k=>scores[k]).map(k=>{
    const c=scores[k];
    return `<div class="card"><h3>${esc(c.name)}</h3><div class="val">${esc(c.score)}</div></div>`;
  }).join('') + `<div class="card"><h3>Active alerts (New)</h3><div class="val">${esc(o.active_alerts)}</div></div>
  <div class="card"><h3>Baseline diffs</h3><div class="val">${esc(o.changes_count)}</div></div>`;
}
async function loadProt(){
  const o=await (await fetch('/api/protection')).json();
  $('#protBody').innerHTML=(o.protection||[]).map(p=>`<tr><td>${esc(p.name)}</td><td class="state-${esc(p.state)}">${esc(p.state)}</td><td>${esc(p.detail)}</td></tr>`).join('');
}
async function loadAlerts(){
  const o=await (await fetch('/api/alerts')).json();
  $('#alertBody').innerHTML=(o.alerts||[]).map(a=>`<tr>
    <td class="sev-${esc(a.severity)}">${esc(a.severity)}</td>
    <td>${esc(a.title)}<div class="note">${esc(a.reason||'')}</div></td>
    <td>${esc(a.category)}</td><td>${esc(a.status)}</td>
    <td>
      <select data-id="${esc(a.id)}" class="st">
        ${['New','Acknowledged','Trusted','Investigating','Resolved'].map(s=>`<option ${s===a.status?'selected':''}>${s}</option>`).join('')}
      </select>
    </td></tr>`).join('');
  document.querySelectorAll('select.st').forEach(sel=>{
    sel.onchange=async()=>{
      await fetch('/api/alerts/'+sel.dataset.id+'/status',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({status:sel.value})});
    };
  });
}
async function loadChanges(){
  const o=await (await fetch('/api/changes')).json();
  $('#changesPre').textContent=JSON.stringify(o,null,2);
}
async function loadProcs(){
  const o=await (await fetch('/api/processes')).json();
  $('#procBody').innerHTML=(o.processes||[]).slice(0,150).map(p=>{
    const t=(p.trust&&p.trust.trust_level)||'—';
    return `<tr><td>${esc(p.pid)}</td><td>${esc(p.name)}</td><td>${esc(t)}</td><td>${esc(p.path)}</td><td>${esc(p.signature)}</td></tr>`;
  }).join('');
}
async function loadNet(){
  const o=await (await fetch('/api/network')).json();
  $('#listenBody').innerHTML=(o.listening||[]).map(x=>`<tr><td>${esc(x.address)}:${esc(x.port)}</td><td>${esc(x.process)}</td><td>${esc(x.path)}</td></tr>`).join('');
  $('#estBody').innerHTML=(o.established||[]).slice(0,80).map(x=>`<tr><td>${esc(x.remote_address)}:${esc(x.remote_port)}</td><td>${esc(x.process)}</td><td>${esc(x.path)}</td></tr>`).join('');
}
async function loadReports(){
  const o=await (await fetch('/api/reports')).json();
  $('#reportList').innerHTML=(o.reports||[]).map(r=>`<button data-name="${esc(r.name)}">${esc(r.name)}</button>`).join(' ')||'<span class="note">No reports yet. Run an audit.</span>';
  $('#reportList').querySelectorAll('button').forEach(b=>b.onclick=async()=>{
    const rr=await (await fetch('/api/reports/'+encodeURIComponent(b.dataset.name))).json();
    $('#reportView').textContent=rr.markdown||rr.error||'';
  });
}
let auditBusy=false;
$('#runAudit').onclick=async()=>{
  if(auditBusy) return;
  auditBusy=true;
  const btn=$('#runAudit');
  btn.disabled=true;
  $('#auditStatus').textContent='Running audit (read-only)…';
  try{
    const resp=await fetch('/api/audit',{method:'POST'});
    const r=await resp.json();
    if(resp.status===409 || r.error==='audit_in_progress'){
      $('#auditStatus').textContent='Audit already in progress';
      return;
    }
    $('#auditStatus').textContent=`Done. Alerts=${r.alerts_count} changes=+${r.changes.added}/-${r.changes.removed}/~${r.changes.modified}`;
    await loadOverview();
  }catch(e){ $('#auditStatus').textContent='Audit failed: '+e; }
  finally{ auditBusy=false; btn.disabled=false; }
};
loadOverview();
</script>
</body>
</html>
"""


def main() -> None:
    svc = get_service()
    host = svc.config.bind_host
    port = int(svc.config.bind_port)
    if host not in ("127.0.0.1", "localhost", "::1"):
        host = "127.0.0.1"
    print(f"Security Center on http://{host}:{port} (READ-ONLY)", flush=True)
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
