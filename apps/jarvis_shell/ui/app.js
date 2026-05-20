/** @typedef {{ id: string, label: string, path: string, icon: string, group: string, activity: string }} NavRoute */
/** @typedef {{ running: boolean, port: number, baseUrl: string, listenerActive: boolean|null, message: string }} DashboardStatus */
/** @typedef {{ isListening: boolean, processAlive: boolean, pid: number|null }} ListenerStatus */

import { createSettingsController } from "./settings.js";

const invoke = window.__TAURI__?.core?.invoke;
const openUrl = window.__TAURI__?.core?.openUrl;

const $ = (sel) => document.querySelector(sel);

const ACTIVITY_LABELS = {
  explorer: "Explorer",
  assistant: "Assistant",
  operator: "Operator",
};

let routes = [];
let activeId = "home";
let activeActivity = "explorer";
let dashboard = /** @type {DashboardStatus|null} */ (null);
let listener = /** @type {ListenerStatus|null} */ (null);

function setStatusDot() {
  const dot = $("#status-dot");
  dot.classList.remove("online", "listening");
  if (!dashboard?.running) return;
  dot.classList.add("online");
  if (listener?.isListening) dot.classList.add("listening");
}

function updateListenerButton() {
  const btn = $("#btn-toggle-listener");
  if (!btn) return;
  const on = listener?.isListening;
  btn.textContent = on ? "Stop listening" : "Start listening";
  btn.classList.toggle("on", !!on);
}

function renderNav() {
  const nav = $("#nav");
  nav.innerHTML = "";
  const filtered = routes.filter(
    (r) => r.id === "home" || r.activity === activeActivity,
  );
  let lastGroup = "";
  for (const r of filtered) {
    if (r.group && r.group !== lastGroup) {
      lastGroup = r.group;
      const g = document.createElement("div");
      g.className = "nav-section";
      g.textContent = r.group;
      nav.appendChild(g);
    }
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "nav-item" + (r.id === activeId ? " active" : "");
    btn.dataset.routeId = r.id;
    btn.innerHTML = `<span class="glyph">${r.icon}</span><span>${r.label}</span>`;
    btn.addEventListener("click", () => selectRoute(r.id));
    nav.appendChild(btn);
  }
}

function selectRoute(id) {
  const route = routes.find((r) => r.id === id);
  if (!route && id !== "settings") return;
  activeId = id;
  if (route?.activity) activeActivity = route.activity;
  try {
    sessionStorage.setItem("jarvis_shell_route", id);
  } catch (_) {}
  renderNav();
  syncActivityButtons();
  $("#side-bar-title").textContent = ACTIVITY_LABELS[activeActivity] || "Explorer";
  $("#active-tab-label").textContent = route?.label || "Settings";

  const frame = $("#content-frame");
  const home = $("#home-panel");
  const settingsPanel = $("#settings-panel");
  const openBtn = $("#btn-open-browser");

  if (id === "settings") {
    frame.hidden = true;
    home.hidden = true;
    if (settingsPanel) settingsPanel.hidden = false;
    openBtn.hidden = true;
    void settingsCtrl.showPanel();
    return;
  }

  if (settingsPanel) settingsPanel.hidden = true;

  if (id === "home") {
    frame.hidden = true;
    home.hidden = false;
    openBtn.hidden = true;
    void renderHomeMetrics();
    return;
  }

  home.hidden = true;
  frame.hidden = false;
  openBtn.hidden = false;
  openBtn.onclick = () => {
    if (openUrl) openUrl(route.path);
    else window.open(route.path, "_blank");
  };
  if (route.path) {
    const target = route.path;
    if (frame.src === target) {
      frame.src = "about:blank";
      requestAnimationFrame(() => {
        frame.src = target;
      });
    } else {
      frame.src = target;
    }
  }
}

function syncActivityButtons() {
  document.querySelectorAll(".activity-btn[data-activity]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.activity === activeActivity);
  });
}

function showHomeAlert(msg) {
  const el = $("#home-alert");
  if (!el) return;
  if (!msg) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.textContent = msg;
  el.classList.remove("hidden");
}

async function renderHomeMetrics() {
  const grid = $("#home-metrics");
  const d = dashboard;
  const l = listener;
  let cafe = { online: false, status: "—" };
  let voice = {
    pttHotkeyDisplay: "Ctrl+Shift+J",
    pttEnabled: true,
    continuousListening: true,
    whisperLazyLoad: false,
    whisperModel: "medium",
    wakeWord: "Jarvis",
  };
  if (invoke) {
    try {
      cafe = await invoke("get_cafe_agent_status");
    } catch (_) {}
    if (d?.running) {
      try {
        voice = await invoke("get_voice_config", { port: d.port });
      } catch (_) {}
    }
  }
  const pttLabel = voice.pttEnabled
    ? `Hold ${voice.pttHotkeyDisplay || "PTT"}`
    : "Disabled";
  const listenMode = voice.continuousListening
    ? `Wake word «${voice.wakeWord || "Jarvis"}»`
    : "PTT / typed only";
  const whisperNote = voice.whisperLazyLoad
    ? `${voice.whisperModel || "medium"} (lazy)`
    : voice.whisperModel || "medium";
  const tiles = [
    ["Dashboard", d?.running ? "Online" : "Offline"],
    ["URL", d?.baseUrl || "—"],
    ["Listener", l?.isListening ? "Active" : "Stopped"],
    ["PTT", pttLabel],
    ["Voice mode", listenMode],
    ["Whisper", whisperNote],
    ["Cafe agent", cafe.online ? "Online" : cafe.status],
    ["Daemon PID", l?.pid != null ? String(l.pid) : "—"],
    ["Backend", d?.message || "—"],
  ];
  grid.innerHTML = tiles
    .map(
      ([label, value]) =>
        `<div class="metric-tile"><div class="label">${label}</div><div class="value">${value}</div></div>`,
    )
    .join("");
}

async function refreshListener() {
  if (!invoke) return;
  try {
    listener = await invoke("get_listener_status");
  } catch (e) {
    console.error(e);
    listener = { isListening: false, processAlive: false, pid: null };
  }
  updateListenerButton();
  setStatusDot();
  if (activeId === "home") renderHomeMetrics();
}

async function refreshBackend() {
  if (!invoke) {
    dashboard = { running: false, message: "Open via: npm run dev" };
    setStatusDot();
    return;
  }
  try {
    dashboard = await invoke("ensure_dashboard", { port: null });
    routes = await invoke("get_nav_routes", { port: dashboard.port });
    await refreshListener();
    renderNav();
    if (activeId === "home") renderHomeMetrics();
  } catch (err) {
    console.error(err);
    dashboard = { running: false, message: String(err) };
  }
  setStatusDot();
}

async function toggleListener() {
  if (!invoke) return;
  const btn = $("#btn-toggle-listener");
  btn.disabled = true;
  showHomeAlert("");
  try {
    if (!dashboard?.running && !listener?.isListening) {
      showHomeAlert(
        "Dashboard is offline on port 5050. Use Restart dashboard, then Start listening.",
      );
      return;
    }
    if (listener?.isListening) {
      await invoke("stop_listener");
    } else {
      await invoke("start_listener");
    }
    await refreshListener();
    await refreshBackend();
  } catch (e) {
    const msg = String(e?.message || e);
    console.error(e);
    showHomeAlert(msg);
  } finally {
    btn.disabled = false;
  }
}

async function openLegacySettings() {
  if (!invoke) {
    showHomeAlert("PyQt settings require the Tauri shell.");
    return;
  }
  try {
    await invoke("open_settings");
  } catch (e) {
    console.error(e);
    showHomeAlert(String(e?.message || e));
  }
}

function openSettings() {
  selectRoute("settings");
}

const settingsCtrl = createSettingsController({
  $,
  get dashboard() {
    return dashboard;
  },
  ensureDashboard: refreshBackend,
  showHomeAlert,
  openLegacySettings,
  selectRoute,
});

function bindActivityBar() {
  document.querySelectorAll(".activity-btn[data-activity]").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeActivity = btn.dataset.activity;
      syncActivityButtons();
      $("#side-bar-title").textContent =
        ACTIVITY_LABELS[activeActivity] || "Explorer";
      const first = routes.find(
        (r) => r.activity === activeActivity && r.id !== "home",
      );
      if (first) selectRoute(first.id);
      else selectRoute("home");
      renderNav();
    });
  });
  $("#btn-settings-activity")?.addEventListener("click", openSettings);
}

async function submitHomeQuery() {
  const input = $("#home-query-input");
  const status = $("#home-query-status");
  const btn = $("#btn-home-query");
  const text = (input?.value || "").trim();
  if (!text) return;
  if (!dashboard?.running) {
    showHomeAlert("Dashboard is offline. Click Restart dashboard first.");
    return;
  }
  if (btn) btn.disabled = true;
  if (status) status.textContent = "Sending…";
  try {
    const port = dashboard.port || 5050;
    const res = await fetch(`http://127.0.0.1:${port}/api/dashboard/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || "Send failed");
    }
    if (input) input.value = "";
    if (status) {
      status.textContent =
        data.message ||
        (data.delivery === "inbox"
          ? "Queued in inbox — Start listening to get a reply."
          : "Queued for Jarvis.");
    }
  } catch (e) {
    const msg = String(e?.message || e);
    if (status) status.textContent = msg;
    showHomeAlert(msg);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function ensureCafeOrchestrator() {
  if (!invoke) return;
  showHomeAlert("");
  const btn = $("#btn-cafe-orchestrator");
  if (btn) btn.disabled = true;
  try {
    const cafe = await invoke("ensure_cafe_orchestrator");
    if (activeId === "home") await renderHomeMetrics();
    if (!cafe?.online) {
      showHomeAlert(cafe?.status || "Café agent did not come online");
    }
  } catch (e) {
    showHomeAlert(String(e?.message || e));
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function boot() {
  settingsCtrl.bind();
  bindActivityBar();
  await refreshBackend();
  let initial = "home";
  try {
    const saved = sessionStorage.getItem("jarvis_shell_route");
    if (saved && routes.some((r) => r.id === saved)) initial = saved;
  } catch (_) {}
  selectRoute(initial);
  $("#btn-refresh-backend")?.addEventListener("click", refreshBackend);
  $("#btn-toggle-listener")?.addEventListener("click", toggleListener);
  $("#btn-cafe-orchestrator")?.addEventListener("click", ensureCafeOrchestrator);
  $("#btn-home-query")?.addEventListener("click", submitHomeQuery);
  $("#home-query-input")?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") submitHomeQuery();
  });
  setInterval(refreshListener, 4000);
}

boot();
