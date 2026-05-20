/** Native settings panel — metadata from GET /api/settings/metadata */

export function createSettingsController(deps) {
  const {
    $,
    dashboard,
    ensureDashboard,
    showHomeAlert,
    openLegacySettings,
    selectRoute,
  } = deps;

  let bundle = null;
  let values = {};
  let activeCategory = "llm";

  function apiBase() {
    const port = dashboard?.port || 5050;
    return `http://127.0.0.1:${port}`;
  }

  async function loadBundle() {
    const base = apiBase();
    const [metaRes, cfgRes] = await Promise.all([
      fetch(`${base}/api/settings/metadata`),
      fetch(`${base}/api/settings/config`),
    ]);
    const meta = await metaRes.json();
    const cfg = await cfgRes.json();
    if (!meta.ok || !cfg.ok) {
      throw new Error(meta.error || cfg.error || "Failed to load settings");
    }
    bundle = meta;
    values = { ...cfg.values };
    if (!activeCategory && bundle.categories?.length) {
      activeCategory = bundle.categories[0].id;
    }
  }

  function fieldsForCategory(catId) {
    return (bundle?.fields || []).filter((f) => f.category === catId);
  }

  function renderCategories() {
    const nav = $("#settings-categories");
    if (!nav || !bundle) return;
    nav.innerHTML = "";
    for (const cat of bundle.categories) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className =
        "settings-cat-btn" + (cat.id === activeCategory ? " active" : "");
      btn.textContent = cat.label;
      btn.addEventListener("click", () => {
        activeCategory = cat.id;
        renderCategories();
        renderFields();
      });
      nav.appendChild(btn);
    }
  }

  function fieldRow(fm) {
    const row = document.createElement("div");
    row.className = "settings-field";
    const label = document.createElement("label");
    label.htmlFor = `sf-${fm.key}`;
    label.innerHTML = `<span class="settings-label">${escapeHtml(fm.label)}</span>`;
    if (fm.description) {
      label.title = fm.description;
    }
    row.appendChild(label);

    let input;
    const id = `sf-${fm.key}`;
    const v = values[fm.key];

    if (fm.fieldType === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.id = id;
      input.checked = Boolean(v);
      input.addEventListener("change", () => {
        values[fm.key] = input.checked;
      });
    } else if (fm.fieldType === "choice" || fm.fieldType === "device") {
      input = document.createElement("select");
      input.id = id;
      for (const ch of fm.choices || []) {
        const opt = document.createElement("option");
        opt.value = ch.value;
        opt.textContent = ch.label;
        input.appendChild(opt);
      }
      const cur = v === null || v === undefined ? "" : String(v);
      input.value = cur;
      input.addEventListener("change", () => {
        values[fm.key] = input.value === "" && fm.nullable ? null : input.value;
      });
    } else if (fm.fieldType === "int" || fm.fieldType === "float") {
      input = document.createElement("input");
      input.type = "number";
      input.id = id;
      if (fm.min != null) input.min = String(fm.min);
      if (fm.max != null) input.max = String(fm.max);
      if (fm.step != null) input.step = String(fm.step);
      input.value = v === null || v === undefined ? "" : String(v);
      input.addEventListener("input", () => {
        const raw = input.value;
        if (raw === "" && fm.nullable) {
          values[fm.key] = null;
          return;
        }
        values[fm.key] =
          fm.fieldType === "int" ? parseInt(raw, 10) : parseFloat(raw);
      });
    } else if (fm.fieldType === "list") {
      input = document.createElement("textarea");
      input.id = id;
      input.rows = 3;
      input.placeholder = "One item per line or JSON array";
      input.value = Array.isArray(v)
        ? JSON.stringify(v, null, 2)
        : String(v || "");
      input.addEventListener("input", () => {
        values[fm.key] = input.value;
      });
    } else {
      input = document.createElement("input");
      input.type = fm.key.includes("api_key") || fm.key.includes("password") ? "password" : "text";
      input.id = id;
      input.value = v === null || v === undefined ? "" : String(v);
      input.placeholder = fm.nullable ? "Default" : "";
      input.addEventListener("input", () => {
        const t = input.value.trim();
        values[fm.key] = t === "" && fm.nullable ? null : input.value;
      });
    }

    if (fm.suffix) {
      const wrap = document.createElement("div");
      wrap.className = "settings-input-wrap";
      wrap.appendChild(input);
      const suf = document.createElement("span");
      suf.className = "settings-suffix";
      suf.textContent = fm.suffix;
      wrap.appendChild(suf);
      row.appendChild(wrap);
    } else {
      row.appendChild(input);
    }

    if (fm.description) {
      const hint = document.createElement("p");
      hint.className = "settings-hint";
      hint.textContent = fm.description;
      row.appendChild(hint);
    }
    return row;
  }

  function renderFields() {
    const form = $("#settings-form");
    if (!form) return;
    form.innerHTML = "";
    const fields = fieldsForCategory(activeCategory);
    if (!fields.length) {
      form.innerHTML = "<p class=\"settings-hint\">No fields in this category.</p>";
      return;
    }
    for (const fm of fields) {
      form.appendChild(fieldRow(fm));
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function showPanel() {
    const panel = $("#settings-panel");
    const frame = $("#content-frame");
    const home = $("#home-panel");
    if (!panel) return;
    frame.hidden = true;
    home.hidden = true;
    panel.hidden = false;
    $("#btn-open-browser")?.setAttribute("hidden", "hidden");
    const status = $("#settings-status");
    if (status) status.textContent = "Loading settings…";
    try {
      if (!dashboard?.running) {
        await ensureDashboard();
      }
      await loadBundle();
      renderCategories();
      renderFields();
      const pathEl = $("#settings-config-path");
      if (pathEl && bundle?.configPath) {
        pathEl.textContent = bundle.configPath;
      }
      if (status) {
        status.textContent = bundle?.configPath
          ? `Editing ${bundle.configPath}`
          : "";
      }
    } catch (e) {
      if (status) status.textContent = String(e.message || e);
      showHomeAlert(String(e.message || e));
    }
  }

  async function resetToDefaults() {
    if (
      !window.confirm(
        "Reset all fields to defaults? Click Save to write config.json.",
      )
    ) {
      return;
    }
    const status = $("#settings-status");
    if (status) status.textContent = "Loading defaults…";
    try {
      const res = await fetch(`${apiBase()}/api/settings/defaults`);
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Reset failed");
      values = { ...data.values };
      renderFields();
      if (status) {
        status.textContent = "Defaults loaded — click Save to persist.";
      }
    } catch (e) {
      if (status) status.textContent = String(e.message || e);
      showHomeAlert(String(e.message || e));
    }
  }

  async function save() {
    const status = $("#settings-status");
    if (status) status.textContent = "Saving…";
    try {
      const res = await fetch(`${apiBase()}/api/settings/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ values }),
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Save failed");
      if (status) status.textContent = data.message || "Saved.";
      showHomeAlert(data.message || "Settings saved.");
    } catch (e) {
      if (status) status.textContent = String(e.message || e);
      showHomeAlert(String(e.message || e));
    }
  }

  function bind() {
    $("#btn-settings-save")?.addEventListener("click", () => save());
    $("#btn-settings-reset")?.addEventListener("click", () => resetToDefaults());
    $("#btn-settings-legacy")?.addEventListener("click", () => openLegacySettings());
    $("#btn-settings-back")?.addEventListener("click", () => selectRoute("home"));
  }

  return { showPanel, bind };
}
