/* Sulainis — single-screen command centre */

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  let overview = null;
  let inboxItems = [];
  let selectedId = null;
  let filter = "all";
  let searchQuery = "";
  let draftPollTimer = null;
  let lastDraftAt = "";
  let pendingDraft = false;
  let selectedItem = null;
  let selectedCalendarDate = "";
  let calendarSchedule = null;
  let daemonListening = false;

  function showToast(msg) {
    const el = $("toast");
    if (!el) return;
    el.textContent = msg;
    el.classList.remove("hidden");
    setTimeout(() => el.classList.add("hidden"), 3200);
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function setPill(el, state, text) {
    if (!el) return;
    el.textContent = text;
    el.className = "pill";
    if (state === "live") el.classList.add("live");
    if (state === "warn") el.classList.add("warn");
  }

  function formatWhen(iso) {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return String(iso).slice(0, 16);
      return d.toLocaleString(undefined, {
        weekday: "short",
        day: "numeric",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return String(iso).slice(0, 16);
    }
  }

  function relativeAge(iso) {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      const mins = Math.floor((Date.now() - d.getTime()) / 60000);
      if (mins < 1) return "just now";
      if (mins < 60) return `${mins}m ago`;
      const hrs = Math.floor(mins / 60);
      if (hrs < 24) return `${hrs}h ago`;
      return `${Math.floor(hrs / 24)}d ago`;
    } catch {
      return "";
    }
  }

  async function fetchDaemonStatus() {
    try {
      const res = await fetch("/api/sulainis/status");
      const data = await res.json();
      if (data.ok) daemonListening = Boolean(data.is_listening);
    } catch {
      daemonListening = false;
    }
    const st = $("status-text");
    if (st && overview) {
      const base = st.textContent?.replace(/ · Jarvis (not listening|listening)/, "") || "Ready";
      st.textContent =
        base +
        (daemonListening ? " · Jarvis listening" : " · Start listening in tray");
    }
  }

  function actionQueuedToast(data, fallback) {
    const via = data?.delivery || "";
    if (via === "mcp") {
      showToast("Done via Google — press Sync to refresh calendar.");
      return;
    }
    if (via === "stdin" || via === "listener") {
      showToast("Sent to Jarvis — watch the reply.");
      return;
    }
    if (via === "bridge" || via === "inbox") {
      if (daemonListening) {
        showToast("Queued for Jarvis — delivering now…");
      } else {
        showToast("Start listening in the tray, then try again.");
      }
      return;
    }
    showToast(fallback || "Queued.");
  }

  async function postAction(action, payload) {
    await fetchDaemonStatus();
    const res = await fetch("/api/sulainis/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, payload: payload || {} }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "action failed");
    return data;
  }

  function waThreadId(label) {
    let h = 0;
    const s = String(label || "");
    for (let i = 0; i < s.length; i += 1) {
      h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
    }
    return `wa-${Math.abs(h).toString(16)}`;
  }

  /** Client-side grouping when API cache predates whatsapp_threads (or stale server). */
  function groupWhatsAppThreads(messages) {
    const buckets = new Map();
    (messages || []).forEach((m) => {
      if (!m || typeof m !== "object") return;
      const key = String(m.chat || m.from || "Nezināms čats").trim() || "Nezināms čats";
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(m);
    });
    const threads = [];
    buckets.forEach((msgs, chat) => {
      msgs.sort((a, b) => String(a.at || "").localeCompare(String(b.at || "")));
      const latest = msgs[msgs.length - 1] || {};
      threads.push({
        id: waThreadId(chat),
        chat,
        from: chat,
        message_count: msgs.length,
        messages: msgs,
        latest_at: latest.at || "",
        preview: String(latest.text || "").slice(0, 240),
      });
    });
    threads.sort((a, b) => String(b.latest_at || "").localeCompare(String(a.latest_at || "")));
    return threads;
  }

  function resolveWhatsAppThreads(data) {
    const fromApi = data.whatsapp_threads || [];
    if (fromApi.length) return fromApi;
    const ch = data.comms?.channels || data.comms || {};
    const wa = ch.whatsapp || data.comms?.whatsapp || [];
    if (Array.isArray(wa) && wa.length) return groupWhatsAppThreads(wa);
    return [];
  }

  function buildInbox(data) {
    const items = [];
    const gmail = data.gmail?.messages || [];
    gmail.forEach((m, i) => {
      const suggestion = m.draft_suggestion || null;
      items.push({
        id: `gmail-${i}`,
        channel: "gmail",
        from: m.from || "?",
        title: m.subject || "(no subject)",
        text: m.snippet || "",
        meta: m.from || "",
        raw: m,
        draftSuggestion: suggestion,
        hasPreparedDraft: Boolean(suggestion?.body),
        messageCount: 1,
        messages: null,
      });
    });
    const threads = resolveWhatsAppThreads(data);
    threads.forEach((t) => {
      items.push({
        id: t.id,
        channel: "whatsapp",
        from: t.chat || t.from || "?",
        title: t.chat || t.from || "?",
        text: t.preview || "",
        meta: t.latest_at ? formatWhen(t.latest_at) : "",
        messageCount: t.message_count || 0,
        messages: t.messages || [],
        raw: { from: t.chat, messages: t.messages },
      });
    });
    return items;
  }

  function matchesSearch(it) {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    let blob = `${it.from} ${it.title} ${it.text}`;
    if (it.messages?.length) {
      blob += " " + it.messages.map((m) => `${m.text || ""}`).join(" ");
    }
    return blob.toLowerCase().includes(q);
  }

  function renderInbox() {
    const list = $("inbox-list");
    const hint = $("inbox-hint");
    list.innerHTML = "";
    const filtered = inboxItems.filter(
      (it) => (filter === "all" || it.channel === filter) && matchesSearch(it)
    );
    if (!filtered.length) {
      hint.textContent =
        searchQuery
          ? "No matches."
          : filter === "all"
            ? dataHint(overview)
            : `No ${filter === "gmail" ? "email" : "WhatsApp"} in cache.`;
      return;
    }
    hint.textContent = "";
    filtered.forEach((it) => {
      const li = document.createElement("li");
      li.className =
        "inbox-item" +
        (selectedId === it.id ? " selected" : "") +
        (it.hasPreparedDraft ? " inbox-draft-ready" : "");
      li.dataset.id = it.id;
      const badge = it.channel === "gmail" ? "email" : "wa";
      li.innerHTML = `
        <div class="row-top">
          <span class="from">${escapeHtml(it.from)}</span>
          <span class="badge badge-${it.channel === "gmail" ? "gmail" : "wa"}">${badge}</span>
        </div>
        <div class="preview">${escapeHtml(it.text || it.title)}${it.channel === "whatsapp" && it.messageCount > 1 ? ` <span class="msg-count-inline">(${it.messageCount})</span>` : ""}</div>
      `;
      li.addEventListener("click", () => selectItem(it.id));
      list.appendChild(li);
    });
  }

  function dataHint(data) {
    if (!data) return "Sync to load messages.";
    const parts = [];
    if (data.gmail?.hint) parts.push(data.gmail.hint);
    if (data.comms?.hint) parts.push(data.comms.hint);
    return parts.join(" ") || "Sync to load messages.";
  }

  function isLatvian() {
    return String(overview?.reply_language || "en").toLowerCase() === "lv";
  }

  function showDraftPanel(show) {
    const panel = $("draft-panel");
    if (!panel) return;
    panel.classList.toggle("hidden", !show);
  }

  function showSuggestionPanel(show) {
    const panel = $("draft-suggestions");
    if (!panel) return;
    panel.classList.toggle("hidden", !show);
  }

  function applyDraftSuggestion(suggestion, { silent } = {}) {
    if (!suggestion) return;
    const body = String(suggestion.body || "").trim();
    if (!body) return;
    const ta = $("draft-text");
    if (ta) {
      ta.value = body;
      delete ta.dataset.userEditing;
    }
    const age = suggestion.generated_at || suggestion.at || "";
    $("draft-age").textContent = age ? relativeAge(age) : "";
    const title = $("draft-panel-title");
    if (title) {
      title.textContent = isLatvian() ? "Sagatavota atbilde" : "Prepared draft";
    }
    showDraftPanel(true);
    if (!silent) showToast(isLatvian() ? "Melnraksts ielādēts." : "Draft loaded.");
  }

  function renderSuggestionChips(suggestion) {
    const wrap = $("suggestion-chips");
    if (!wrap) return;
    wrap.innerHTML = "";
    const angles = Array.isArray(suggestion?.angles) ? suggestion.angles : [];
    const lv = isLatvian();
    if (angles.length) {
      angles.forEach((label, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "suggestion-chip" + (idx === 0 ? " primary" : "");
        btn.textContent = String(label || "").slice(0, 120);
        btn.title = lv
          ? "Izmanto sagatavoto melnrakstu (vari pielāgot zemāk)"
          : "Use the prepared draft below (you can edit it)";
        btn.addEventListener("click", () => applyDraftSuggestion(suggestion));
        wrap.appendChild(btn);
      });
    }
    const useBtn = document.createElement("button");
    useBtn.type = "button";
    useBtn.className = "suggestion-chip primary";
    useBtn.textContent = lv ? "Atvērt pilnu melnrakstu" : "Open full draft";
    useBtn.addEventListener("click", () => applyDraftSuggestion(suggestion));
    wrap.appendChild(useBtn);
    const title = $("draft-suggestions-title");
    if (title) {
      title.textContent = lv ? "Ieteiktās atbildes" : "Suggested replies";
    }
    showSuggestionPanel(true);
  }

  function clearSuggestionChips() {
    const wrap = $("suggestion-chips");
    if (wrap) wrap.innerHTML = "";
    showSuggestionPanel(false);
  }

  function applyDraft(draft) {
    const text = (draft?.text || "").trim();
    if (!text) return;
    const ta = $("draft-text");
    if (ta && !ta.dataset.userEditing) ta.value = text;
    $("draft-age").textContent = draft.at ? relativeAge(draft.at) : "";
    showDraftPanel(true);
    lastDraftAt = draft.at || "";
  }

  async function pollDraft() {
    if (!pendingDraft) return;
    try {
      const res = await fetch("/api/sulainis/draft");
      const data = await res.json();
      if (!data.ok) return;
      const draft = data.draft || {};
      if (draft.text && draft.at !== lastDraftAt) {
        applyDraft(draft);
        pendingDraft = false;
        stopDraftPoll();
        showToast("Draft ready.");
      }
    } catch {
      /* ignore */
    }
  }

  function startDraftPoll() {
    pendingDraft = true;
    stopDraftPoll();
    draftPollTimer = setInterval(pollDraft, 2000);
    pollDraft();
  }

  function stopDraftPoll() {
    if (draftPollTimer) {
      clearInterval(draftPollTimer);
      draftPollTimer = null;
    }
  }

  function renderConversation(it) {
    const box = $("detail-text");
    if (!box) return;
    box.classList.add("conversation");
    box.innerHTML = "";
    const msgs = it.messages;
    if (!msgs?.length) {
      box.classList.remove("conversation");
      box.textContent = it.text || "(empty)";
      return;
    }
    msgs.forEach((m) => {
      const row = document.createElement("div");
      row.className = "thread-msg";
      const meta = document.createElement("div");
      meta.className = "thread-meta";
      meta.textContent = m.at ? formatWhen(m.at) : "";
      const body = document.createElement("div");
      body.className = "thread-body";
      body.textContent = m.text || "";
      row.appendChild(meta);
      row.appendChild(body);
      box.appendChild(row);
    });
    box.scrollTop = box.scrollHeight;
  }

  function selectItem(id) {
    selectedId = id;
    selectedItem = inboxItems.find((x) => x.id === id) || null;
    renderInbox();
    const it = selectedItem;
    if (!it) return;
    $("detail-empty").classList.add("hidden");
    $("detail-body").classList.remove("hidden");
    $("detail-channel").textContent = it.channel === "gmail" ? "Gmail" : "WhatsApp";
    $("detail-title").textContent = it.title;
    const countLabel =
      it.channel === "whatsapp" && it.messageCount > 1
        ? `${it.messageCount} messages · `
        : "";
    $("detail-meta").textContent = countLabel + (it.meta || it.from);
    if (it.channel === "whatsapp" && it.messages?.length) {
      renderConversation(it);
    } else {
      $("detail-text").classList.remove("conversation");
      $("detail-text").textContent = it.text || "(empty)";
    }
    clearSuggestionChips();
    if (it.channel === "gmail" && it.draftSuggestion?.body) {
      renderSuggestionChips(it.draftSuggestion);
      applyDraftSuggestion(it.draftSuggestion, { silent: true });
    } else {
      const draft = overview?.assistant_draft;
      if (draft?.text) applyDraft(draft);
      else showDraftPanel(false);
    }
    const replyBtn = $("btn-reply");
    if (replyBtn) {
      replyBtn.textContent =
        it.channel === "gmail" && it.hasPreparedDraft
          ? isLatvian()
            ? "Pārstrādāt ar Jarvis"
            : "Refine with Jarvis"
          : isLatvian()
            ? "Sagatavot atbildi"
            : "Draft reply";
    }
  }

  async function draftReply(it) {
    try {
      if (it.channel === "gmail") {
        await postAction("draft_email", {
          from: it.raw.from,
          subject: it.raw.subject,
          snippet: it.raw.snippet,
        });
      } else {
        const msgs = it.messages || [];
        const last = msgs[msgs.length - 1];
        const threadPreview = msgs
          .slice(-12)
          .map((m) => `${m.at ? formatWhen(m.at) + ": " : ""}${m.text || ""}`)
          .join("\n");
        await postAction("draft_whatsapp", {
          from: it.from,
          text: last?.text || it.text,
          thread_preview: threadPreview,
        });
      }
      if (it.channel === "gmail" && it.draftSuggestion?.body) {
        applyDraftSuggestion(it.draftSuggestion);
        showToast("Using prepared draft — edit and Send, or Refine with Jarvis.");
        return;
      }
      startDraftPoll();
      actionQueuedToast({ delivery: "bridge" }, "Draft requested via Jarvis…");
    } catch (e) {
      showToast(String(e.message || e));
    }
  }

  async function sendDraft() {
    const it = selectedItem;
    const body = ($("draft-text")?.value || "").trim();
    if (!it || !body) {
      showToast("Nothing to send.");
      return;
    }
    try {
      const data = await postAction(
        it.channel === "gmail" ? "send_email" : "send_whatsapp",
        it.channel === "gmail"
          ? { from: it.raw.from, subject: it.raw.subject, body }
          : { from: it.from, chat: it.from, body }
      );
      showDraftPanel(false);
      actionQueuedToast(data, "Send queued.");
    } catch (e) {
      showToast(String(e.message || e));
    }
  }

  async function cancelDraft() {
    try {
      await postAction("cancel_draft", {});
      $("draft-text").value = "";
      delete $("draft-text")?.dataset.userEditing;
      showDraftPanel(false);
      stopDraftPoll();
      showToast("Draft cancelled.");
    } catch (e) {
      showToast(String(e.message || e));
    }
  }

  function renderHero(data) {
    $("product-title").textContent = data.product || "Sulainis";
    const op = data.operator;
    $("hero-greeting").textContent = op ? `Good day, ${op}.` : "Good day.";
    const clock = data.clock || {};
    $("hero-clock").textContent = (clock.time || "--:--").slice(0, 8);
    $("hero-date").textContent = clock.date || "—";
    const w = data.weather || {};
    if (w.ok && w.current) {
      $("hero-weather").textContent = `${w.location || ""}, ${w.current.temp_c}°C · ${w.current.description || ""}`;
    } else {
      $("hero-weather").textContent = w.error || "Weather n/a";
    }
    const pw = data.parents_weather || {};
    const pel = $("hero-parents-weather");
    if (pel) {
      if (pw.ok && pw.current) {
        const pl = pw.label || "Parents";
        pel.textContent = `${pl}: ${pw.current.temp_c}°C · ${pw.current.description || ""}`;
        pel.classList.remove("hidden");
      } else {
        pel.classList.add("hidden");
      }
    }
    const pill = $("unread-pill");
    const n = data.unread_count || 0;
    if (pill) {
      if (n > 0) {
        pill.textContent = String(n);
        pill.classList.remove("hidden");
      } else {
        pill.classList.add("hidden");
      }
    }
    const tts = $("tts-hint");
    if (tts) {
      if (data.tts_hint) {
        tts.textContent = data.tts_hint;
        tts.classList.remove("hidden");
      } else {
        tts.classList.add("hidden");
      }
    }
  }

  function renderTopbarKpis(v) {
    const row = $("topbar-kpis");
    if (!row) return;
    row.innerHTML = "";
    const kpis = v?.kpis || [];
    if (!kpis.length) return;
    kpis.slice(0, 4).forEach((k) => {
      const el = document.createElement("div");
      el.className = "topbar-kpi";
      el.innerHTML = `<span class="kpi-label">${escapeHtml(k.label)}</span><span class="kpi-value">${escapeHtml(k.value)}</span>`;
      row.appendChild(el);
    });
    if (v.kpi_age) {
      const age = document.createElement("span");
      age.className = "kpi-age";
      age.textContent = v.kpi_age;
      row.appendChild(age);
    }
  }

  function renderTickerFeed(feed) {
    const track = $("ticker-track");
    if (!track) return;
    const rows = (feed || []).filter((r) => r && r.text);
    if (!rows.length) {
      track.innerHTML = `<span class="ticker-segment"><span class="ticker-kind">status</span> Sulainis ready — Sync to refresh.</span>`;
      track.style.animation = "none";
      return;
    }
    const segments = rows
      .map(
        (row) =>
          `<span class="ticker-segment"><span class="ticker-kind">${escapeHtml(row.kind || "info")}</span> ${escapeHtml(row.text)}</span>`
      )
      .join('<span class="ticker-sep" aria-hidden="true"> ◆ </span>');
    track.innerHTML = segments + '<span class="ticker-sep" aria-hidden="true"> ◆ </span>' + segments;
    track.style.animation = "";
  }

  function renderCacheAges(ages) {
    const el = $("cache-ages");
    if (!el || !ages) return;
    const parts = [];
    if (ages.gmail) parts.push(`mail ${ages.gmail}`);
    if (ages.comms) parts.push(`chat ${ages.comms}`);
    if (ages.calendar) parts.push(`cal ${ages.calendar}`);
    el.textContent = parts.length ? parts.join(" · ") : "";
  }

  function eventStartDate(ev) {
    const start = String(ev?.start || "");
    if (!start) return "";
    return start.includes("T") ? start.split("T")[0] : start.slice(0, 10);
  }

  function buildCalendarScheduleClient(cal, horizonDays) {
    const events = (cal?.events || []).filter((e) => e && typeof e === "object");
    const daysN = Math.max(1, Math.min(30, horizonDays || cal?.horizon_days || 7));
    const byDate = new Map();
    events.forEach((ev) => {
      const d = eventStartDate(ev);
      if (!d) return;
      if (!byDate.has(d)) byDate.set(d, []);
      byDate.get(d).push(ev);
    });
    const days = [];
    const base = new Date();
    base.setHours(0, 0, 0, 0);
    for (let i = 0; i < daysN; i += 1) {
      const dt = new Date(base);
      dt.setDate(base.getDate() + i);
      const iso = todayIsoFromDate(dt);
      const list = (byDate.get(iso) || []).sort((a, b) =>
        String(a.start || "").localeCompare(String(b.start || ""))
      );
      days.push({
        date: iso,
        weekday: dt.toLocaleDateString(undefined, { weekday: "short" }),
        label: dt.toLocaleDateString(undefined, { day: "numeric", month: "short" }),
        is_today: i === 0,
        event_count: list.length,
        events: list,
      });
    }
    return { days, total_events: events.length, horizon_days: daysN, hint: cal?.hint };
  }

  function todayIsoFromDate(d) {
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${d.getFullYear()}-${m}-${day}`;
  }

  function todayIso() {
    const d = new Date();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${d.getFullYear()}-${m}-${day}`;
  }

  function initCalendarForm() {
    const dateInput = $("cal-date");
    if (dateInput && !dateInput.value) dateInput.value = selectedCalendarDate || todayIso();
  }

  function renderCalendarSchedule(schedule, cal) {
    calendarSchedule = schedule;
    const week = $("calendar-week");
    const list = $("calendar-events");
    const hint = $("calendar-hint");
    if (!week || !list) return;

    const days = schedule?.days || [];
    if (!selectedCalendarDate && days.length) {
      selectedCalendarDate = days.find((d) => d.is_today)?.date || days[0].date;
    }
    initCalendarForm();

    week.innerHTML = "";
    if (!days.length) {
      list.innerHTML = "";
      hint.textContent = cal?.hint || schedule?.hint || "Sync after linking Google.";
      setPill($("calendar-status"), "warn", "—");
      return;
    }

    const total = schedule?.total_events ?? cal?.events?.length ?? 0;
    setPill($("calendar-status"), total > 0 ? "live" : "warn", String(total));
    hint.textContent = cal?.hint || schedule?.hint || "";

    days.forEach((day) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "calendar-day-btn";
      if (day.is_today) btn.classList.add("today");
      if (day.event_count > 0) btn.classList.add("has-events");
      if (day.date === selectedCalendarDate) btn.classList.add("selected");
      btn.dataset.date = day.date;
      btn.innerHTML = `
        <span class="cal-dow">${escapeHtml(day.weekday || "")}</span>
        <span class="cal-dom">${escapeHtml((day.label || "").split(" ")[0] || "")}</span>
        ${day.event_count > 0 ? `<span class="cal-count">${day.event_count}</span>` : ""}
      `;
      btn.addEventListener("click", () => {
        selectedCalendarDate = day.date;
        const dateInput = $("cal-date");
        if (dateInput) dateInput.value = day.date;
        renderCalendarSchedule(schedule, cal);
      });
      week.appendChild(btn);
    });

    const active = days.find((d) => d.date === selectedCalendarDate) || days[0];
    list.innerHTML = "";
    const events = active?.events || [];
    if (!events.length) {
      const li = document.createElement("li");
      li.className = "calendar-empty";
      li.textContent = "No events this day.";
      list.appendChild(li);
      return;
    }
    events.forEach((ev) => {
      const li = document.createElement("li");
      const when = formatWhen(ev.start);
      const loc = ev.location ? ` · ${ev.location}` : "";
      li.innerHTML = `
        <strong>${escapeHtml(ev.title || "Event")}</strong>
        <span class="cal-ev-time">${escapeHtml(when)}${escapeHtml(loc)}</span>
      `;
      list.appendChild(li);
    });
  }

  async function submitCalendarEvent(ev) {
    ev.preventDefault();
    const title = ($("cal-title")?.value || "").trim();
    const dateVal = ($("cal-date")?.value || "").trim();
    const startT = ($("cal-start")?.value || "09:00").trim();
    const endT = ($("cal-end")?.value || "10:00").trim();
    const notes = ($("cal-notes")?.value || "").trim();
    const location = ($("cal-location")?.value || "").trim();
    if (!title || !dateVal) {
      showToast("Title and date required.");
      return;
    }
    const start = `${dateVal}T${startT}:00`;
    const end = `${dateVal}T${endT}:00`;
    try {
      const data = await postAction("add_calendar_event", {
        title,
        start,
        end,
        description: notes,
        location,
      });
      actionQueuedToast(data, "Calendar request queued.");
      if (data.delivery === "mcp") {
        await refresh();
      }
      $("cal-title").value = "";
      $("cal-notes").value = "";
    } catch (e) {
      showToast(String(e.message || e));
    }
  }

  async function planCalendarDay() {
    const dateVal = ($("cal-date")?.value || selectedCalendarDate || todayIso()).trim();
    const notes = ($("cal-notes")?.value || "").trim();
    const title = ($("cal-title")?.value || "").trim();
    const request =
      notes ||
      title ||
      window.prompt("What should Jarvis plan for this day?", "Meetings, tasks, reminders…") ||
      "";
    if (!request) return;
    try {
      const data = await postAction("plan_calendar", { date: dateVal, request });
      actionQueuedToast(data, "Planning queued.");
    } catch (e) {
      showToast(String(e.message || e));
    }
  }

  function renderCafeAgent(panel) {
    const pill = $("cafe-agent-pill");
    const hint = $("cafe-agent-hint");
    const result = $("cafe-agent-result");
    if (!pill) return;
    const online = Boolean(panel?.online);
    setPill(pill, online ? "live" : "warn", online ? "online" : "offline");
    if (hint) {
      const svc = panel?.health?.service || "cafe-orchestrator";
      const sales = panel?.health?.sales_rows;
      const extra =
        typeof sales === "number" ? ` · ${sales} sales row(s)` : "";
      hint.textContent = online
        ? `${svc} ready${extra}`
        : "Offline — Jarvis shell «Start café agent» or run_cafe_orchestrator.ps1";
    }
    if (result && result.classList.contains("hidden") && !result.textContent) {
      result.textContent = "";
    }
  }

  function formatMoney(n) {
    const v = Number(n);
    if (Number.isNaN(v)) return "—";
    return `${v.toFixed(2)} €`;
  }

  function formatCafeAgentHtml(summary, payload, taskType) {
    const parts = [`<p class="cafe-agent-summary">${escapeHtml(summary)}</p>`];

    if (taskType === "payroll_calc") {
      const rows = payload?.employees;
      if (Array.isArray(rows) && rows.length) {
        let totalNet = 0;
        const body = rows
          .map((r) => {
            totalNet += Number(r.net_eur) || 0;
            return `<tr>
              <td>${escapeHtml(r.name || "")}</td>
              <td>${Number(r.hours || 0).toFixed(1)}</td>
              <td>${formatMoney(r.gross_eur)}</td>
              <td>${formatMoney(r.iin_eur)}</td>
              <td>${formatMoney(r.vsaoi_employee_eur)}</td>
              <td>${formatMoney(r.net_eur)}</td>
              <td>${formatMoney(r.total_employer_cost_eur)}</td>
            </tr>`;
          })
          .join("");
        parts.push(
          `<table class="cafe-data-table"><thead><tr>
            <th>Name</th><th>Hrs</th><th>Gross</th><th>IIN</th><th>VSAOI</th><th>Net</th><th>Employer</th>
          </tr></thead><tbody>${body}</tbody></table>
          <p class="hint">Total net: ${formatMoney(totalNet)} · month ${escapeHtml(String(payload.month || ""))}</p>`
        );
        return parts.join("");
      }
    }

    if (taskType === "sales_analysis") {
      const top = payload?.top;
      if (Array.isArray(top) && top.length) {
        const body = top
          .map(
            (r) =>
              `<tr><td>${escapeHtml(String(r.date || "").slice(5))}</td>
              <td>${escapeHtml(r.product || "")}</td>
              <td>${r.quantity ?? ""}</td>
              <td>${formatMoney(r.amount)}</td></tr>`
          )
          .join("");
        parts.push(
          `<table class="cafe-data-table"><thead><tr>
            <th>Date</th><th>Product</th><th>Qty</th><th>Amount</th>
          </tr></thead><tbody>${body}</tbody></table>`
        );
        return parts.join("");
      }
    }

    if (taskType === "schedule_plan") {
      const days = payload?.days;
      if (Array.isArray(days) && days.length) {
        days.forEach((d) => {
          parts.push(
            `<div class="cafe-schedule-day">${escapeHtml(d.weekday || "")} ${escapeHtml(String(d.date || "").slice(5))}</div>`
          );
          const shifts = d.shifts || [];
          if (shifts.length) {
            const body = shifts
              .map(
                (s) =>
                  `<tr><td>${escapeHtml(s.name || "")}</td>
                  <td>${escapeHtml(s.start || "")}–${escapeHtml(s.end || "")}</td>
                  <td>${Number(s.hours || 0).toFixed(1)} h</td></tr>`
              )
              .join("");
            parts.push(
              `<table class="cafe-data-table"><thead><tr>
                <th>Staff</th><th>Time</th><th>Hours</th>
              </tr></thead><tbody>${body}</tbody></table>`
            );
          }
        });
        if (payload.persisted) {
          parts.push(
            `<p class="hint">${payload.shifts_written ?? 0} shift(s) saved to database.</p>`
          );
        }
        if (payload.planner) {
          parts.push(
            `<p class="hint">Planner: ${escapeHtml(String(payload.planner))}</p>`
          );
        }
        return parts.join("");
      }
    }

    if (
      (taskType === "email" || taskType === "whatsapp") &&
      payload?.queued
    ) {
      parts.push(
        `<p class="hint">Delivery: ${escapeHtml(payload.delivery || "—")}. ${escapeHtml(payload.note || "")}</p>`
      );
      return parts.join("");
    }

    if (taskType === "weather_check" && Array.isArray(payload?.days)) {
      const body = payload.days
        .slice(0, 7)
        .map(
          (d) =>
            `<tr><td>${escapeHtml(String(d.date || "").slice(5))}</td>
            <td>${d.temp_max_c != null ? `${d.temp_max_c}°` : "—"}</td>
            <td>${d.temp_min_c != null ? `${d.temp_min_c}°` : "—"}</td>
            <td>${d.precipitation_mm != null ? `${d.precipitation_mm} mm` : "—"}</td></tr>`
        )
        .join("");
      parts.push(
        `<table class="cafe-data-table"><thead><tr>
          <th>Day</th><th>Max</th><th>Min</th><th>Rain</th>
        </tr></thead><tbody>${body}</tbody></table>`
      );
      return parts.join("");
    }

    parts.push(
      `<pre style="margin:0;white-space:pre-wrap">${escapeHtml(JSON.stringify(payload, null, 2))}</pre>`
    );
    return parts.join("");
  }

  async function postCafeTask(task, label) {
    const resultEl = $("cafe-agent-result");
    if (resultEl) {
      resultEl.classList.remove("hidden");
      resultEl.textContent = `${label}…`;
    }
    $("status-text").textContent = `${label}…`;
    try {
      const res = await fetch("/api/cafe-agent/task", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || data.message || `HTTP ${res.status}`);
      }
      const summary = data.result?.summary || data.summary || "Done";
      const payload = data.result?.data ?? data.data ?? {};
      const taskType = task?.type || "";
      if (resultEl) {
        resultEl.innerHTML = formatCafeAgentHtml(summary, payload, taskType);
      }
      showToast(summary.slice(0, 120));
      $("status-text").textContent = `Café: ${summary.slice(0, 80)}`;
    } catch (e) {
      const msg = String(e.message || e);
      if (resultEl) resultEl.textContent = msg;
      showToast(msg);
      $("status-text").textContent = `Café error: ${msg}`;
    }
  }

  function renderBeachOps(beach) {
    const summary = $("beach-strategic");
    const list = $("beach-plan-list");
    const hint = $("beach-ops-hint");
    const pill = $("beach-ops-pill");
    if (!list) return;
    list.innerHTML = "";
    if (beach?.enabled === false) {
      if (hint) hint.textContent = "Beach forecast disabled in config.";
      return;
    }
    if (summary) summary.textContent = beach?.strategic_summary || "";
    const rows = beach?.analysis || [];
    const openish = rows.filter((r) =>
      ["likely_open", "likely_open_partial"].includes(r.verdict)
    );
    if (pill) setPill(pill, openish.length ? "live" : "warn", String(openish.length));
    if (!rows.length) {
      if (hint) hint.textContent = beach?.hint || "Sync to refresh beach forecast.";
      return;
    }
    if (hint) hint.textContent = beach?.rules_note || "";
    rows.forEach((row) => {
      const li = document.createElement("li");
      const open = ["likely_open", "likely_open_partial"].includes(row.verdict);
      li.className = open ? "verdict-open" : "verdict-closed";
      const when = row.open_from ? ` · from ${row.open_from}` : "";
      li.innerHTML = `<span class="beach-date">${escapeHtml(row.weekday || "")} ${escapeHtml(String(row.date || "").slice(5))}</span>${escapeHtml(row.label || "")}${escapeHtml(when)}`;
      list.appendChild(li);
    });
  }

  function renderIntegrationsBar(integ) {
    const row = $("integrations-row");
    if (!row) return;
    const servers = integ?.servers || [];
    row.innerHTML = "";
    const ready = integ?.ready_count || 0;
    const total = integ?.server_count || servers.length;
    setPill($("integrations-pill"), ready === total && total > 0 ? "live" : "warn", `${ready}/${total}`);
    servers.forEach((s) => {
      const chip = document.createElement("span");
      const st = s.state || "unknown";
      chip.className = `integration-chip state-${st === "ready" ? "ready" : st === "error" ? "error" : "warn"}`;
      chip.title = s.action_hint || s.detail || "";
      chip.innerHTML = `<span class="chip-name">${escapeHtml(s.name)}</span><span class="chip-state">${escapeHtml(st)}</span>`;
      row.appendChild(chip);
    });
  }

  async function fetchWorkQueuePanel() {
    const res = await fetch("/api/sulainis/work-queue");
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || `work queue HTTP ${res.status}`);
    }
    return data;
  }

  async function postWorkQueue(operation, payload) {
    const res = await fetch("/api/sulainis/work-queue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ operation, ...payload }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || "work queue failed");
    return data;
  }

  function renderTaskQueue(wq) {
    const list = $("task-queue-list");
    const hint = $("task-queue-hint");
    const pill = $("task-queue-pill");
    if (!list) return;
    list.innerHTML = "";
    if (!wq?.enabled) {
      if (hint) hint.textContent = "Work queue disabled in settings.";
      if (pill) setPill(pill, "warn", "off");
      return;
    }
    const items = wq.items || [];
    const active = items.filter((i) => i.status !== "done" && i.status !== "cancelled");
    if (pill) setPill(pill, active.length ? "live" : "warn", String(active.length));
    if (hint) hint.textContent = active.length ? "" : "No open tasks — add one below.";
    if (!items.length) {
      const li = document.createElement("li");
      li.className = "task-queue-empty";
      li.textContent = "Queue empty.";
      list.appendChild(li);
      return;
    }
    items.forEach((item) => {
      const li = document.createElement("li");
      if (item.status === "done") li.classList.add("done-row");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = item.status === "done";
      cb.title = "Mark done";
      cb.addEventListener("change", async () => {
        try {
          await postWorkQueue("update", {
            item_id: item.id,
            status: cb.checked ? "done" : "open",
          });
          await refresh();
        } catch (e) {
          showToast(String(e.message || e));
          cb.checked = !cb.checked;
        }
      });
      const title = document.createElement("span");
      title.className = "task-queue-title";
      const pri = item.priority === "high" ? " ⚡" : "";
      title.textContent = `${item.title || "Task"}${pri}`;
      li.appendChild(cb);
      li.appendChild(title);
      list.appendChild(li);
    });
  }

  async function runTaskQueueViaJarvis() {
    try {
      const data = await postAction("run_task_queue", {});
      actionQueuedToast(data, "Task queue queued.");
    } catch (e) {
      showToast(String(e.message || e));
    }
  }

  async function submitTaskQueue(ev) {
    ev.preventDefault();
    const title = ($("tq-title")?.value || "").trim();
    if (!title) return;
    try {
      await postWorkQueue("add", { title });
      $("tq-title").value = "";
      await refresh();
      showToast("Task added.");
    } catch (e) {
      showToast(String(e.message || e));
    }
  }

  function openVenuefyOverlay() {
    const v = overview?.venuefy || {};
    const overlay = $("venuefy-overlay");
    const frame = $("venuefy-frame");
    const fallback = $("venuefy-fallback");
    const note = $("venuefy-note");
    overlay.classList.remove("hidden");

    const kpiRow = document.getElementById("venuefy-kpis");
    if (kpiRow) kpiRow.remove();
    if (v.kpis?.length) {
      const row = document.createElement("div");
      row.id = "venuefy-kpis";
      row.className = "kpi-row";
      v.kpis.forEach((k) => {
        const card = document.createElement("div");
        card.className = "kpi";
        card.innerHTML = `<div class="kpi-label">${escapeHtml(k.label)}</div><div class="kpi-value">${escapeHtml(k.value)}</div>`;
        row.appendChild(card);
      });
      overlay.querySelector(".overlay-panel").insertBefore(row, frame);
    }

    note.textContent = v.note || "";
    if (v.embed_allowed) {
      frame.classList.remove("hidden");
      frame.src = v.url || "";
      fallback.classList.add("hidden");
    } else {
      frame.classList.add("hidden");
      frame.removeAttribute("src");
      fallback.classList.remove("hidden");
      fallback.innerHTML = `
        <p>Venuefy cannot be embedded here (site security). Open in this browser view:</p>
        <p><button type="button" class="btn btn-primary" id="venuefy-same-tab">Open Venuefy stats</button></p>
      `;
      document.getElementById("venuefy-same-tab")?.addEventListener("click", () => {
        window.location.href = v.url || v.login_url;
      });
    }
  }

  async function refresh() {
    $("status-text").textContent = "Updating…";
    try {
      const res = await fetch("/api/sulainis/overview");
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "overview failed");
      overview = data;
      await fetchDaemonStatus();
      inboxItems = buildInbox(data);
      if (selectedId && !inboxItems.find((x) => x.id === selectedId)) {
        selectedId = null;
        selectedItem = null;
        $("detail-body").classList.add("hidden");
        $("detail-empty").classList.remove("hidden");
      } else if (selectedId) {
        selectedItem = inboxItems.find((x) => x.id === selectedId) || null;
      }
      renderHero(data);
      renderTopbarKpis(data.venuefy);
      renderTickerFeed(data.ticker_feed);
      renderCacheAges(data.cache_ages);
      renderInbox();
      let sched = data.calendar_schedule;
      if (!sched?.days?.length) {
        sched = buildCalendarScheduleClient(
          data.calendar,
          data.calendar?.horizon_days || 7
        );
      }
      renderCalendarSchedule(sched, data.calendar);
      let wqPanel = data.work_queue;
      if (!wqPanel || typeof wqPanel.enabled === "undefined") {
        try {
          wqPanel = await fetchWorkQueuePanel();
        } catch {
          wqPanel = { enabled: false, items: [] };
        }
      }
      renderTaskQueue(wqPanel);
      renderCafeAgent(data.cafe_agent);
      renderBeachOps(data.beach_ops);
      renderIntegrationsBar(data.integrations);
      if (data.assistant_draft?.text) applyDraft(data.assistant_draft);
      $("status-text").textContent = `Updated ${new Date().toLocaleTimeString()}`;
    } catch (e) {
      $("status-text").textContent = `Error: ${e.message || e}`;
    }
  }

  document.querySelectorAll(".inbox-tabs .tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".inbox-tabs .tab").forEach((t) => t.classList.remove("active"));
      btn.classList.add("active");
      filter = btn.dataset.filter || "all";
      renderInbox();
    });
  });

  $("inbox-search")?.addEventListener("input", (e) => {
    searchQuery = (e.target.value || "").trim();
    renderInbox();
  });

  $("btn-sync")?.addEventListener("click", async () => {
    $("status-text").textContent = "Syncing…";
    await fetch("/api/sulainis/sync");
    await refresh();
    showToast("Synced.");
  });

  $("btn-briefing")?.addEventListener("click", async () => {
    try {
      await postAction("briefing", { force: true });
      showToast("Briefing queued.");
    } catch (e) {
      showToast(String(e.message || e));
    }
  });

  $("btn-ask")?.addEventListener("click", async () => {
    const it = selectedItem;
    if (!it) return;
    const q = `About this ${it.channel} from ${it.from}: ${it.text.slice(0, 200)} — what should I do?`;
    try {
      await postAction("ask", { question: `${overview?.wake_word || "Jarvis"}, ${q}` });
      showToast("Question sent.");
    } catch (e) {
      showToast(String(e.message || e));
    }
  });

  $("btn-reply")?.addEventListener("click", () => {
    if (selectedItem) draftReply(selectedItem);
  });

  $("btn-send")?.addEventListener("click", sendDraft);
  $("btn-cancel-draft")?.addEventListener("click", cancelDraft);
  $("btn-edit-draft")?.addEventListener("click", () => {
    const ta = $("draft-text");
    if (ta) {
      ta.dataset.userEditing = "1";
      ta.focus();
    }
    showToast("Edit the draft, then Send.");
  });

  $("draft-text")?.addEventListener("input", () => {
    const ta = $("draft-text");
    if (ta) ta.dataset.userEditing = "1";
  });

  $("btn-venuefy")?.addEventListener("click", openVenuefyOverlay);
  $("venuefy-close")?.addEventListener("click", () => $("venuefy-overlay").classList.add("hidden"));

  $("calendar-form")?.addEventListener("submit", submitCalendarEvent);
  $("btn-cal-plan")?.addEventListener("click", planCalendarDay);
  $("task-queue-form")?.addEventListener("submit", submitTaskQueue);
  $("btn-run-queue")?.addEventListener("click", runTaskQueueViaJarvis);

  $("btn-cafe-weather")?.addEventListener("click", () =>
    postCafeTask({ type: "weather_check", days: 5 }, "Weather")
  );
  $("btn-cafe-sales")?.addEventListener("click", () => {
    const task = { type: "sales_analysis", days: 14 };
    const sample = overview?.cafe_agent?.sample_csv_path;
    if (sample) task.csv_path = sample;
    postCafeTask(task, "Sales");
  });
  $("btn-cafe-schedule")?.addEventListener("click", async () => {
    const ok = window.confirm(
      "Save this week's draft shifts to the database? Existing shifts in that week will be replaced."
    );
    if (!ok) return;
    await postCafeTask(
      { type: "schedule_plan", persist: true },
      "Schedule"
    );
  });
  $("btn-cafe-payroll")?.addEventListener("click", () =>
    postCafeTask({ type: "payroll_calc" }, "Payroll")
  );
  $("btn-cafe-email")?.addEventListener("click", () =>
    postCafeTask({ type: "email", action: "sync" }, "Email")
  );
  $("btn-cafe-whatsapp")?.addEventListener("click", () =>
    postCafeTask({ type: "whatsapp", action: "sync" }, "WhatsApp")
  );

  refresh();
  setInterval(refresh, 120000);
})();
