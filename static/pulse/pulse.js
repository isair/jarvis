/**
 * Jarvis Pulse — fullscreen holographic dashboard (local-only).
 */

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const statusParts = [];

  function setStatus(el, state) {
    if (!el) return;
    el.classList.remove("live", "warn");
    if (state === "live") el.classList.add("live");
    if (state === "warn") el.classList.add("warn");
  }

  function recordStatus(id, state, text) {
    const idx = statusParts.findIndex((p) => p.id === id);
    const entry = { id, state, text };
    if (idx >= 0) statusParts[idx] = entry;
    else statusParts.push(entry);
    updateStatusBar();
  }

  function updateStatusBar() {
    const el = $("pulse-status-text");
    if (!el) return;
    const order = ["weather", "news", "social", "gmail", "comms", "stats", "clock"];
    const sorted = [...statusParts].sort(
      (a, b) => order.indexOf(a.id) - order.indexOf(b.id)
    );
    el.textContent = sorted.map((p) => p.text).join(" · ") || "Dashboard ready.";
  }

  function pad(n) {
    return String(n).padStart(2, "0");
  }

  function tickClock() {
    const now = new Date();
    $("clock-time").textContent = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
    $("clock-date").textContent = now.toLocaleDateString(undefined, {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
    recordStatus("clock", "live", `Local time ${$("clock-time").textContent}`);
  }

  async function fetchJson(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`${path} ${res.status}`);
    return res.json();
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function displayText(s) {
    const raw = String(s || "");
    const el = document.createElement("textarea");
    el.innerHTML = raw;
    return escapeHtml(el.value);
  }

  function escapeAttr(s) {
    return escapeHtml(s).replace(/'/g, "&#39;");
  }

  function renderWeather(data) {
    const body = $("weather-body");
    const status = $("weather-status");
    if (!data.ok) {
      body.innerHTML = `<p class="placeholder">${escapeHtml(data.error || "Weather unavailable")}</p>`;
      setStatus(status, "warn");
      recordStatus("weather", "warn", `Weather: unavailable`);
      return;
    }
    setStatus(status, "live");
    const c = data.current || {};
    const days = (data.daily || [])
      .map(
        (d) =>
          `<div class="weather-day"><strong>${escapeHtml(d.date || "")}</strong><br>${escapeHtml(d.description || "")}<br>${d.max_c ?? "?"}° / ${d.min_c ?? "?"}°</div>`
      )
      .join("");
    body.innerHTML = `
      <div class="weather-main">
        <span class="weather-temp">${c.temp_c ?? "?"}°C</span>
        <span class="weather-desc">${escapeHtml(c.description || "")}</span>
      </div>
      <p class="weather-meta">${escapeHtml(data.location || "Baldone")} · feels ${c.feels_c ?? c.temp_c ?? "?"}°C · wind ${c.wind_kmph ?? "?"} km/h · humidity ${c.humidity ?? "?"}%</p>
      <div class="weather-daily">${days}</div>
    `;
    recordStatus(
      "weather",
      "live",
      `Weather: ${data.location || "Baldone"}, ${c.temp_c ?? "?"}°C`
    );
  }

  function renderGmail(data) {
    const list = $("gmail-list");
    const hint = $("gmail-hint");
    const status = $("gmail-status");
    list.innerHTML = "";
    const msgs = data.messages || [];
    if (!msgs.length) {
      const statusHint = data.hint || "Gmail kešs tukšs.";
      hint.textContent = statusHint;
      list.innerHTML = `<li class="comms-status-line">${escapeHtml(statusHint)}</li>`;
      setStatus(status, "warn");
      recordStatus("gmail", "warn", `Gmail: ${statusHint.slice(0, 50)}`);
      return;
    }
    setStatus(status, "live");
    hint.textContent = data.updated_at ? `Updated ${data.updated_at}` : "";
    msgs.forEach((m) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="mail-from">${escapeHtml(m.from)}</span>
        <div class="mail-subject">${escapeHtml(m.subject)}</div>
        <div class="mail-snippet">${escapeHtml(m.snippet || "")}</div>`;
      list.appendChild(li);
    });
    recordStatus("gmail", "live", `Gmail: ${msgs.length} messages`);
  }

  const PLATFORM_EMOJI = {
    instagram: "📷",
    facebook: "📘",
    linkedin: "💼",
    tiktok: "🎵",
    x: "𝕏",
    youtube: "▶",
    telegram: "✈",
    other: "🔗",
  };

  function renderSocialFeed(data) {
    const root = $("social-feed-columns");
    const hint = $("social-hint");
    const status = $("social-status");
    const nameEl = $("social-business-name");
    root.innerHTML = "";
    const feeds = data.feeds || [];
    const biz = (data.business_name || "").trim();
    nameEl.textContent = biz || "";
    nameEl.style.display = biz ? "block" : "none";

    if (!feeds.length) {
      hint.textContent = data.hint || "Add social links in Setup Wizard.";
      setStatus(status, "warn");
      recordStatus("social", "warn", "Social: no channels configured");
      return;
    }

    let totalItems = data.item_count || 0;
    feeds.forEach((feed) => {
      const platform = (feed.platform || "other").toLowerCase();
      const emoji = PLATFORM_EMOJI[platform] || PLATFORM_EMOJI.other;
      const label = escapeHtml(feed.label || platform);
      const profileUrl = feed.url || "";
      const items = feed.items || [];
      const col = document.createElement("article");
      col.className = "social-feed-col";
      const source = feed.source && feed.source !== "none" ? feed.source : "";
      col.innerHTML = `
        <header class="social-feed-col-header">
          <span>${emoji} <a href="${escapeAttr(profileUrl)}" target="_blank" rel="noopener">${label}</a></span>
          ${source ? `<span class="social-feed-source">${escapeHtml(source)}</span>` : ""}
        </header>
        <ul class="feed-list"></ul>
      `;
      const list = col.querySelector(".feed-list");
      if (!items.length) {
        const li = document.createElement("li");
        li.innerHTML = `<span class="feed-summary">No posts yet — refresh caches every 30 min or add <code>feed_url</code> in config.</span>`;
        list.appendChild(li);
      } else {
        items.forEach((item) => {
          const li = document.createElement("li");
          const title = displayText(item.title || "Post");
          const link = item.url
            ? `<a class="feed-title" href="${escapeAttr(item.url)}" target="_blank" rel="noopener">${title}</a>`
            : `<span class="feed-title">${title}</span>`;
          const summary = item.summary
            ? `<div class="feed-summary">${displayText(item.summary)}</div>`
            : "";
          const meta = item.at ? `<div class="feed-meta">${escapeHtml(item.at)}</div>` : "";
          li.innerHTML = `${link}${summary}${meta}`;
          list.appendChild(li);
        });
      }
      root.appendChild(col);
    });

    if (!totalItems) {
      totalItems = feeds.reduce((n, f) => n + (f.items || []).length, 0);
    }
    setStatus(status, totalItems ? "live" : "warn");
    hint.textContent = data.updated_at
      ? `Feeds updated ${data.updated_at}${data.hint ? " — " + data.hint : ""}`
      : data.hint || "";
    recordStatus(
      "social",
      totalItems ? "live" : "warn",
      totalItems
        ? `Social: ${totalItems} posts, ${feeds.length} channels`
        : `Social: ${feeds.length} channels (awaiting feed data)`
    );
  }

  function renderComms(data) {
    const list = $("comms-list");
    const hint = $("comms-hint");
    const status = $("comms-status");
    list.innerHTML = "";
    const ch = data.channels || {};
    const wa = ch.whatsapp || [];
    const mx = ch.matrix || [];
    const all = [
      ...wa.map((e) => ({ ...e, channel: "whatsapp" })),
      ...mx.map((e) => ({ ...e, channel: "matrix" })),
    ].sort((a, b) => String(b.at || "").localeCompare(String(a.at || "")));

    if (!all.length) {
      const statusHint = data.hint || "Nav ziņu — Connect WhatsApp vai pārbaudi MCP.";
      hint.textContent = statusHint;
      list.innerHTML = `<li class="comms-status-line">${escapeHtml(statusHint)}</li>`;
      setStatus(status, "warn");
      recordStatus("comms", "warn", `Comms: ${statusHint.slice(0, 60)}`);
      return;
    }
    setStatus(status, "live");
    hint.textContent = data.updated_at ? `Updated ${data.updated_at}` : "";
    all.slice(0, 24).forEach((e) => {
      const li = document.createElement("li");
      const tag = e.channel === "whatsapp" ? "wa" : "";
      li.innerHTML = `<span class="channel-tag ${tag}">${e.channel}</span>
        <strong>${escapeHtml(e.from || e.room || "?")}</strong>: ${escapeHtml(e.text || e.body || "")}`;
      list.appendChild(li);
    });
    recordStatus("comms", "live", `Comms: ${all.length} messages`);
  }

  function renderNews(data) {
    const list = $("news-list");
    const hint = $("news-hint");
    const status = $("news-status");
    list.innerHTML = "";
    const items = data.items || [];
    if (!items.length) {
      hint.textContent = data.hint || "Waiting for Strategist agent output.";
      setStatus(status, "warn");
      recordStatus("news", "warn", "News: waiting for feed");
      return;
    }
    setStatus(status, "live");
    hint.textContent = data.updated_at
      ? `${data.agent || "strategist"} · ${data.updated_at}`
      : "";
    items.slice(0, 8).forEach((item) => {
      const li = document.createElement("li");
      const title = escapeHtml(item.title || "Story");
      const link = item.url
        ? `<a class="news-title" href="${escapeAttr(item.url)}" target="_blank" rel="noopener">${title}</a>`
        : `<span class="news-title">${title}</span>`;
      li.innerHTML = `${link}<div class="news-summary">${escapeHtml(item.summary || "")}</div>`;
      list.appendChild(li);
    });
    recordStatus("news", "live", `News: ${items.length} stories`);
  }

  function tryInjectLogin(iframe, creds) {
    if (!iframe || !creds?.ok) return;
    const origin = creds.target_origin;
    try {
      const doc = iframe.contentDocument || iframe.contentWindow?.document;
      if (!doc) {
        iframe.contentWindow?.postMessage(
          {
            type: "jarvis-cafe-login",
            username: creds.username,
            password: creds.password,
          },
          origin || "*"
        );
        return;
      }
      const sel = creds.selectors || {};
      const userEl = doc.querySelector(sel.username);
      const passEl = doc.querySelector(sel.password);
      const submitEl = doc.querySelector(sel.submit);
      if (userEl) userEl.value = creds.username;
      if (passEl) passEl.value = creds.password;
      if (submitEl) submitEl.click();
    } catch {
      iframe.contentWindow?.postMessage(
        {
          type: "jarvis-cafe-login",
          username: creds.username,
          password: creds.password,
        },
        origin || "*"
      );
    }
  }

  async function initStats() {
    const frame = $("stats-frame");
    const fallback = $("stats-fallback");
    const openLink = $("stats-open-link");
    const loginLink = $("stats-login-link");
    const credHint = $("stats-cred-hint");
    const status = $("stats-status");
    let config;
    try {
      config = await fetchJson("/api/pulse/cafe-stats-config");
    } catch {
      $("stats-fallback-msg").textContent = "Venuefy stats API nav pieejams.";
      setStatus(status, "warn");
      recordStatus("stats", "warn", "Venuefy stats: API error");
      return;
    }
    if (!config.ok || !config.url) {
      $("stats-fallback-msg").textContent = config.note || "Nav konfigurēts stats URL.";
      setStatus(status, "warn");
      recordStatus("stats", "warn", "Venuefy stats: URL not configured");
      return;
    }

    const statsUrl = config.url;
    const loginUrl = config.login_url || "https://miers.venuefy.lv/login";
    openLink.href = statsUrl;
    loginLink.href = loginUrl;

    if (config.embed_allowed === true) {
      fallback.classList.add("hidden");
      frame.classList.remove("stats-frame-hidden");
      frame.src = statsUrl;
      frame.addEventListener("load", async () => {
        if (!config.has_credentials) return;
        try {
          const creds = await fetchJson("/api/pulse/cafe-credentials");
          tryInjectLogin(frame, creds);
        } catch {
          /* loopback-only */
        }
      });
    } else {
      frame.classList.add("stats-frame-hidden");
      fallback.classList.remove("hidden");
      $("stats-fallback-msg").textContent =
        config.note ||
        "Venuefy neļauj rādīt lapu iekš dashboard (X-Frame-Options). Nospied Atvērt statistiku.";
    }

    if (config.has_credentials) {
      credHint.textContent =
        "Pieslēgums saglabāts configā — ielogojies Venuefy, tad atver statistiku.";
      setStatus(status, "live");
      recordStatus("stats", "live", `Venuefy: atvērt ${statsUrl} pārlūkā`);
    } else {
      credHint.textContent =
        "Pievieno pulse_cafe_user / pulse_cafe_pass configā (serveris jau ir?).";
      setStatus(status, "warn");
      recordStatus("stats", "warn", "Venuefy: vajag login");
    }
  }

  async function refreshAll() {
    tickClock();
    try {
      renderWeather(await fetchJson("/api/pulse/weather"));
    } catch {
      $("weather-body").innerHTML = '<p class="placeholder">Weather fetch failed.</p>';
      setStatus($("weather-status"), "warn");
      recordStatus("weather", "warn", "Weather: fetch failed");
    }
    try {
      renderNews(await fetchJson("/api/pulse/news"));
    } catch {
      $("news-hint").textContent = "News feed API unavailable.";
      recordStatus("news", "warn", "News: API error");
    }
    try {
      renderSocialFeed(await fetchJson("/api/pulse/social-feed"));
    } catch {
      $("social-hint").textContent = "Social feed API unavailable.";
      recordStatus("social", "warn", "Social: API error");
    }
    try {
      renderGmail(await fetchJson("/api/pulse/gmail"));
    } catch {
      $("gmail-hint").textContent = "Gmail API unavailable.";
      recordStatus("gmail", "warn", "Gmail: API error");
    }
    try {
      renderComms(await fetchJson("/api/pulse/comms"));
    } catch {
      $("comms-hint").textContent = "Comms API unavailable.";
      recordStatus("comms", "warn", "Comms: API error");
    }
  }

  tickClock();
  setInterval(tickClock, 1000);
  initStats();
  refreshAll();
  setInterval(refreshAll, 120000);
})();
