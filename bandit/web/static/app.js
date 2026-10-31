"use strict";

// Конструктор эксперимента: состояние живёт в браузере, сервер отвечает за
// валидацию, сборку RunConfig и запуск. Спецификация совпадает с `spec.json`,
// который CLI пишет в каталог прогона, поэтому сохранённый JSON воспроизводим.

const API = "/api";
const STORAGE_KEY = "bandit.ui.spec.v1";
const PREVIEW_DELAY = 350;

const TABLE_METRICS = [
  ["final_points", "Очки"],
  ["cumulative_arm_regret", "Arm regret"],
  ["optimal_arm_rate", "Доля лучшей руки"],
  ["points_per_pull", "Очков/вытягивание"],
  ["efficiency", "КПД"],
  ["steps_to_90pct", "Шаг 90%"],
];

const state = {
  schema: null,
  envKind: "layered",
  envParams: {},
  agents: {},        // key -> {param: value} только для выбранных агентов
  policies: {},
  exp: { title: "", runs: 30, seed: 42, steps: 3000, jobs: 1, plots: true, add_oracle: true },
  badFields: new Set(),
  job: null,
  pollTimer: null,
  previewTimer: null,
  currentRun: null,
};

function $(id) {
  return document.getElementById(id);
}

function el(tag, props, children) {
  const node = document.createElement(tag);
  Object.assign(node, props || {});
  for (const child of children || []) {
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, options) {
  const settings = Object.assign({ headers: { "Content-Type": "application/json" } }, options || {});
  const response = await fetch(API + path, settings);
  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (error) {
      data = { detail: text };
    }
  }
  if (!response.ok) {
    let detail = data && data.detail;
    if (Array.isArray(detail)) {
      detail = detail.map((item) => `${(item.loc || []).join(".")}: ${item.msg}`).join("; ");
    } else if (detail && typeof detail === "object") {
      detail = JSON.stringify(detail);
    }
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return data;
}

function fmtInt(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Math.round(value).toLocaleString("ru-RU");
}

function fmtNum(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const abs = Math.abs(value);
  if (abs !== 0 && abs < 1) return value.toFixed(3);
  if (abs < 100) return value.toFixed(2);
  return Math.round(value).toLocaleString("ru-RU");
}

function fmtDuration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 1) return `${seconds.toFixed(2)} с`;
  if (seconds < 60) return `${seconds.toFixed(1)} с`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds - minutes * 60);
  if (minutes < 60) return `${minutes} мин ${rest} с`;
  const hours = Math.floor(minutes / 60);
  return `${hours} ч ${minutes - hours * 60} мин`;
}

function fmtPct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(0)}%`;
}

function envSchema(kind) {
  return state.schema.environments.find((item) => item.kind === (kind || state.envKind));
}

function agentSchema(key) {
  return state.schema.agents.find((item) => item.key === key);
}

function policySchema(key) {
  return state.schema.policies.find((item) => item.key === key);
}

function defaultsFrom(fields) {
  const params = {};
  for (const field of fields || []) params[field.name] = field.default;
  return params;
}

function defaultsForEnv(kind) {
  const env = envSchema(kind);
  const base = (env.presets.find((preset) => preset.key === "default") || { params: {} }).params;
  return Object.assign(defaultsFrom(env.fields), base);
}

function agentDefaults(key) {
  return defaultsFrom(agentSchema(key).fields);
}

function policyDefaults(key) {
  return defaultsFrom(policySchema(key).fields);
}

function specFromDefaults() {
  const defaults = state.schema.defaults;
  return {
    title: "",
    env: { kind: defaults.env.kind, params: defaults.env.params },
    agents: defaults.agents.map((key) => ({ key, params: {} })),
    policies: defaults.policies.map((key) => ({ key, params: {} })),
    runs: defaults.runs,
    seed: defaults.seed,
    steps: defaults.steps,
    jobs: defaults.jobs,
    plots: defaults.plots,
    add_oracle: defaults.add_oracle,
  };
}

function setSpec(spec) {
  const kind = (spec.env && spec.env.kind) || "layered";
  state.envKind = envSchema(kind) ? kind : "layered";
  state.envParams = Object.assign(
    defaultsForEnv(state.envKind),
    (spec.env && spec.env.params) || {}
  );
  state.agents = {};
  for (const item of spec.agents || []) {
    if (!agentSchema(item.key)) continue;
    state.agents[item.key] = Object.assign(agentDefaults(item.key), item.params || {});
  }
  state.policies = {};
  for (const item of spec.policies || []) {
    if (!policySchema(item.key)) continue;
    state.policies[item.key] = Object.assign(policyDefaults(item.key), item.params || {});
  }
  const defaults = state.schema.defaults;
  state.exp = {
    title: spec.title ?? defaults.title,
    runs: Number(spec.runs ?? defaults.runs),
    seed: Number(spec.seed ?? defaults.seed),
    steps: Number(spec.steps ?? defaults.steps),
    jobs: Number(spec.jobs ?? defaults.jobs),
    plots: spec.plots ?? defaults.plots,
    add_oracle: spec.add_oracle ?? defaults.add_oracle,
  };
  state.badFields.clear();
}

function currentSpec() {
  return {
    title: state.exp.title,
    env: { kind: state.envKind, params: Object.assign({}, state.envParams) },
    agents: Object.keys(state.agents).map((key) => ({
      key,
      params: Object.assign({}, state.agents[key]),
    })),
    policies: Object.keys(state.policies).map((key) => ({
      key,
      params: Object.assign({}, state.policies[key]),
    })),
    runs: Number(state.exp.runs),
    seed: Number(state.exp.seed),
    steps: Number(state.exp.steps),
    jobs: Number(state.exp.jobs),
    plots: Boolean(state.exp.plots),
    add_oracle: Boolean(state.exp.add_oracle),
  };
}

function clearMessages() {
  $("messages").replaceChildren();
}

function showMessage(kind, text) {
  const box = el("div", { className: `message ${kind}` });
  box.textContent = text;
  const container = $("messages");
  container.append(box);
  while (container.children.length > 6) container.firstElementChild.remove();
  return box;
}

function saveLocal() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(currentSpec()));
  } catch (error) {
    /* приватный режим браузера — просто не запоминаем */
  }
}

function loadLocal() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    return null;
  }
}

function downloadSpec() {
  const blob = new Blob([JSON.stringify(currentSpec(), null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = el("a", { href: url, download: "bandit-spec.json" });
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function fieldControl(field, params, onChange, path) {
  const wrapper = el("label", { className: "field" });
  wrapper.append(el("span", { className: "label", textContent: field.label || field.name }));
  if (field.type === "bool") {
    const input = el("input", { type: "checkbox", checked: Boolean(params[field.name]) });
    input.addEventListener("change", () => {
      params[field.name] = input.checked;
      onChange();
    });
    wrapper.append(el("span", { className: "check" }, [input, "да"]));
  } else if (field.type === "json") {
    const input = el("textarea", { value: JSON.stringify(params[field.name] ?? null) });
    input.addEventListener("change", () => {
      try {
        params[field.name] = JSON.parse(input.value.trim() === "" ? "null" : input.value);
        wrapper.classList.remove("invalid");
        state.badFields.delete(path);
      } catch (error) {
        wrapper.classList.add("invalid");
        state.badFields.add(path);
        showMessage("err", `«${field.label}»: некорректный JSON — ${error.message}`);
      }
      onChange();
    });
    wrapper.append(input);
  } else {
    const step = field.step ?? (field.type === "int" ? 1 : 0.01);
    const input = el("input", {
      type: "number",
      step: String(step),
      value: params[field.name] ?? "",
    });
    if (field.min !== null && field.min !== undefined) input.min = String(field.min);
    if (field.max !== null && field.max !== undefined) input.max = String(field.max);
    input.addEventListener("change", () => {
      const raw = input.value.trim();
      let value = raw === "" ? null : Number(raw);
      if (value !== null && field.type === "int") value = Math.round(value);
      if (value !== null && Number.isNaN(value)) {
        wrapper.classList.add("invalid");
        state.badFields.add(path);
      } else {
        wrapper.classList.remove("invalid");
        state.badFields.delete(path);
        params[field.name] = value;
      }
      onChange();
    });
    wrapper.append(input);
  }
  if (field.help) wrapper.append(el("p", { className: "help", textContent: field.help }));
  return wrapper;
}

function renderEnv() {
  const env = envSchema();
  $("env-kind").replaceChildren(
    ...state.schema.environments.map((item) =>
      el("option", { value: item.kind, textContent: item.label, selected: item.kind === state.envKind })
    )
  );
  $("env-note").textContent = env.n_meta_actions
    ? `Мета-действий: ${env.n_meta_actions} — ${env.meta_actions.join(", ")}.`
    : "У среды нет мета-действий: любая политика вырождается в «без метацикла».";
  $("env-ignored").textContent = env.ignored.length
    ? `Не настраивается из UI: ${env.ignored.join(", ")} (callable, задаётся в коде).`
    : "";
  $("env-preset").replaceChildren(
    ...env.presets.map((preset) => el("option", { value: preset.key, textContent: preset.label }))
  );
  const base = $("env-fields");
  const advanced = $("env-advanced");
  base.replaceChildren();
  advanced.replaceChildren();
  for (const field of env.fields) {
    const control = fieldControl(field, state.envParams, onSpecChanged, `env:${field.name}`);
    (field.group === "advanced" ? advanced : base).append(control);
  }
  if (!advanced.children.length) advanced.append(el("p", { className: "hint", textContent: "Нет." }));
}

function renderAgents() {
  const list = $("agent-list");
  list.replaceChildren();
  for (const agent of state.schema.agents) {
    const selected = Object.prototype.hasOwnProperty.call(state.agents, agent.key);
    const item = el("div", { className: `item ${selected ? "selected" : ""}` });
    const head = el("label", { className: "item-head" });
    const checkbox = el("input", { type: "checkbox", checked: selected });
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.agents[agent.key] = agentDefaults(agent.key);
      else delete state.agents[agent.key];
      renderAgents();
      onSpecChanged();
    });
    head.append(
      checkbox,
      el("span", { className: "item-title", textContent: agent.label }),
      el("span", { className: "item-kind", textContent: agent.kind })
    );
    item.append(head);
    if (selected && agent.fields.length) {
      const grid = el("div", { className: "item-params" });
      for (const field of agent.fields) {
        grid.append(fieldControl(
          field, state.agents[agent.key], onSpecChanged, `agent:${agent.key}:${field.name}`
        ));
      }
      item.append(grid);
    }
    list.append(item);
  }
  $("agents-count").textContent = String(Object.keys(state.agents).length);
}

function renderPolicies() {
  const list = $("policy-list");
  list.replaceChildren();
  for (const policy of state.schema.policies) {
    const selected = Object.prototype.hasOwnProperty.call(state.policies, policy.key);
    const item = el("div", { className: `item ${selected ? "selected" : ""}` });
    const head = el("label", { className: "item-head" });
    const checkbox = el("input", { type: "checkbox", checked: selected });
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.policies[policy.key] = policyDefaults(policy.key);
      else delete state.policies[policy.key];
      renderPolicies();
      onSpecChanged();
    });
    head.append(
      checkbox,
      el("span", { className: "item-title", textContent: policy.label }),
      el("span", { className: "item-kind", textContent: policy.oracle ? "оракул" : policy.kind })
    );
    item.append(head);
    if (selected && policy.fields.length) {
      const grid = el("div", { className: "item-params" });
      for (const field of policy.fields) {
        grid.append(fieldControl(
          field, state.policies[policy.key], onSpecChanged, `policy:${policy.key}:${field.name}`
        ));
      }
      item.append(grid);
    }
    list.append(item);
  }
  $("policies-count").textContent = String(Object.keys(state.policies).length);
}

function renderExperiment() {
  $("exp-title").value = state.exp.title;
  $("exp-runs").value = state.exp.runs;
  $("exp-steps").value = state.exp.steps;
  $("exp-seed").value = state.exp.seed;
  $("exp-jobs").value = state.exp.jobs;
  $("exp-plots").checked = Boolean(state.exp.plots);
  $("exp-oracle").checked = Boolean(state.exp.add_oracle);
}

function renderAll() {
  renderEnv();
  renderAgents();
  renderPolicies();
  renderExperiment();
  schedulePreview();
}

function onSpecChanged() {
  saveLocal();
  schedulePreview();
}

function schedulePreview() {
  if (state.previewTimer) clearTimeout(state.previewTimer);
  state.previewTimer = setTimeout(refreshPreview, PREVIEW_DELAY);
}

async function refreshPreview() {
  state.previewTimer = null;
  if (state.badFields.size) {
    state.preview = null;
    renderSummary(null, "есть поля с ошибками — они подсвечены красным");
    return;
  }
  const spec = currentSpec();
  if (!spec.agents.length || !spec.policies.length) {
    state.preview = null;
    renderSummary(null, "выберите хотя бы одного агента и одну политику");
    return;
  }
  try {
    state.preview = await api("/preview", { method: "POST", body: JSON.stringify(spec) });
    renderSummary(state.preview);
  } catch (error) {
    state.preview = null;
    renderSummary(null, error.message);
  }
}

function renderSummary(preview, errorText) {
  const dl = $("summary");
  dl.replaceChildren();
  const rows = preview
    ? [
        ["Конфигов", String(preview.n_configs)],
        ["Эпизодов", fmtInt(preview.n_episodes)],
        ["Шагов", fmtInt(preview.total_steps)],
        ["Оценка времени", fmtDuration(preview.est_seconds)],
      ]
    : [["Состояние", errorText || "—"]];
  for (const [label, value] of rows) {
    dl.append(el("dt", { textContent: label }), el("dd", { textContent: value }));
  }
  if (preview) {
    const names = el("code", { textContent: preview.configs.join(" · ") });
    dl.append(
      el("dt", { textContent: "Имена" }),
      el("dd", {}, [el("details", {}, [
        el("summary", { textContent: `показать (${preview.configs.length})` }),
        names,
      ])])
    );
  }
}

async function validateSpec() {
  clearMessages();
  try {
    const result = await api("/validate", { method: "POST", body: JSON.stringify(currentSpec()) });
    if (result.ok) {
      showMessage("ok", `Проверка пройдена: ${result.n_configs} конфигов, параметры принимаются средой.`);
    } else {
      for (const text of result.errors.slice(0, 8)) showMessage("err", text);
      if (result.errors.length > 8) showMessage("err", `…ещё ${result.errors.length - 8} ошибок.`);
    }
  } catch (error) {
    showMessage("err", "Проверка не удалась: " + error.message);
  }
}

async function startJob() {
  clearMessages();
  if (state.badFields.size) {
    showMessage("err", "Есть поля с некорректными значениями (подсвечены красным).");
    return;
  }
  const button = $("btn-run");
  button.disabled = true;
  try {
    const job = await api("/jobs", { method: "POST", body: JSON.stringify(currentSpec()) });
    applyJob(job);
    showMessage("info", `Задание ${job.id}: ${job.n_configs} конфигов в очереди.`);
  } catch (error) {
    showMessage("err", "Запуск отклонён: " + error.message);
    button.disabled = false;
  }
}

async function cancelJob() {
  if (!state.job) return;
  try {
    const job = await api(`/jobs/${state.job.id}/cancel`, { method: "POST" });
    applyJob(job);
  } catch (error) {
    showMessage("err", "Отмена не удалась: " + error.message);
  }
}

function applyJob(job) {
  state.job = job;
  $("job-card").hidden = false;
  $("job-id").textContent = job.id;
  const chip = $("job-state");
  chip.textContent = job.state;
  chip.className = `chip ${job.state}`;
  const log = $("job-log");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
  log.textContent = job.log.join("\n");
  if (atBottom) log.scrollTop = log.scrollHeight;
  const active = job.state === "queued" || job.state === "running";
  $("btn-cancel").disabled = !active;
  $("btn-run").disabled = active;
  $("progress").hidden = false;
  $("progress-fill").style.width = `${Math.round((job.progress || 0) * 100)}%`;
  const parts = [`${job.n_done}/${job.n_configs} конфигов`, fmtDuration(job.elapsed_sec)];
  if (active) parts.push(`осталось ≈ ${fmtDuration((job.steps * job.runs * Math.max(job.n_configs - job.n_done, 0)) / 5000)}`);
  $("progress-text").textContent = parts.join(" · ");
  if (active) schedulePoll();
}

function schedulePoll() {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(pollJob, 1200);
}

async function pollJob() {
  state.pollTimer = null;
  if (!state.job) return;
  try {
    const job = await api(`/jobs/${state.job.id}`);
    applyJob(job);
    if (job.state !== "queued" && job.state !== "running") finishJob(job);
  } catch (error) {
    showMessage("err", "Состояние задания недоступно: " + error.message);
    state.pollTimer = setTimeout(pollJob, 3000);
  }
}

async function finishJob(job) {
  $("btn-run").disabled = false;
  $("btn-cancel").disabled = true;
  if (job.state === "done") {
    showMessage("ok", `Готово за ${fmtDuration(job.elapsed_sec)}. Метрики ниже.`);
    try {
      const run = await api(`/runs/${job.id}`);
      renderMetrics(run, {
        metricsCard: "metrics-card",
        table: "metrics-table",
        plots: "plots",
        hypothesesBox: "hypotheses-box",
        hypotheses: "hypotheses",
        hint: "metrics-hint",
        runLabel: "metrics-run",
      });
    } catch (error) {
      showMessage("err", "Не удалось прочитать метрики: " + error.message);
    }
    refreshRuns();
  } else if (job.state === "cancelled") {
    showMessage("info", "Задание отменено.");
  } else {
    showMessage("err", "Задание завершилось с ошибкой: " + (job.error || "см. лог"));
  }
}

function hypothesisCard(item) {
  let kind = "";
  if (item.status.startsWith("подтвержд")) kind = "ok";
  else if (item.status.startsWith("опроверг")) kind = "no";
  else if (item.status.startsWith("частично")) kind = "warn";
  const status = el("span", {
    className: `status ${kind}`.trim(),
    textContent: `${item.code}: ${item.status}`,
  });
  const head = el("div", {}, [status, el("strong", { textContent: ` ${item.title}` })]);
  return el("div", { className: "hypothesis" }, [head, el("p", { textContent: item.detail })]);
}

function buildMetricsTable(rows) {
  const table = el("table");
  const headRow = el("tr");
  headRow.append(el("th", { textContent: "Конфиг" }));
  for (const [, label] of TABLE_METRICS) headRow.append(el("th", { textContent: label }));
  const body = el("tbody");
  for (const row of rows || []) {
    const tr = el("tr");
    tr.append(el("td", { textContent: row.config }));
    for (const [key] of TABLE_METRICS) {
      const cell = row.values ? row.values[key] : null;
      const text = cell && cell.mean !== null && cell.mean !== undefined
        ? (cell.sem ? `${fmtNum(cell.mean)} ± ${fmtNum(cell.sem)}` : fmtNum(cell.mean))
        : "—";
      tr.append(el("td", { textContent: text, title: key }));
    }
    body.append(tr);
  }
  table.append(el("thead", {}, [headRow]), body);
  return table;
}

function renderMetrics(run, ids) {
  if (ids.metricsCard) $(ids.metricsCard).hidden = false;
  if (ids.runLabel) $(ids.runLabel).textContent = run.id;
  const host = $(ids.table);
  host.replaceChildren();
  if (run.metrics && run.metrics.length) {
    host.append(buildMetricsTable(run.metrics));
  } else {
    host.append(el("p", { className: "hint", textContent: "В каталоге прогона нет metrics.csv." }));
  }
  const plots = $(ids.plots);
  plots.replaceChildren();
  if (run.has_plots) {
    for (const [name, caption] of [["dashboard.png", "Динамика"], ["final_points.png", "Итоговые очки"]]) {
      const src = `${API}/runs/${run.id}/file/${name}?t=${Date.now()}`;
      plots.append(el("figure", { className: "plot" }, [
        el("img", { src, alt: caption, loading: "lazy" }),
        el("figcaption", { className: "hint", textContent: caption }),
      ]));
    }
  }
  if (ids.hypothesesBox) {
    const box = $(ids.hypothesesBox);
    const hypotheses = run.hypotheses || [];
    $(ids.hypotheses).replaceChildren(...hypotheses.map(hypothesisCard));
    box.hidden = hypotheses.length === 0;
  }
  if (ids.hint) {
    $(ids.hint).textContent = run.has_report
      ? `${run.n_runs} прогонов, сид ${run.base_seed}, ${run.n_steps} шагов; вердикты — из report.json.`
      : `${run.n_runs} прогонов, сид ${run.base_seed}, ${run.n_steps} шагов; отчёт ещё не построен.`;
  }
}

function renderRunSummary(run) {
  const dl = $("run-summary");
  dl.replaceChildren();
  const rows = [
    ["Создан", run.created || "—"],
    ["Прогонов", String(run.n_runs)],
    ["Сид", String(run.base_seed)],
    ["Шагов", String(run.n_steps)],
    ["Время", fmtDuration(run.elapsed_sec)],
    ["Конфигов", String(run.configs.length)],
  ];
  for (const [label, value] of rows) {
    dl.append(el("dt", { textContent: label }), el("dd", { textContent: value }));
  }
}

async function refreshRuns() {
  const list = $("runs-list");
  try {
    const runs = await api("/runs");
    if (!runs.length) {
      list.replaceChildren(el("p", { className: "hint", textContent: "Пока нет прогонов: запустите первый эксперимент." }));
      return;
    }
    list.replaceChildren(...runs.map((run) => {
      const item = el("div", { className: `item ${state.currentRun === run.id ? "active" : ""}` });
      const head = el("div", { className: "item-head" });
      head.append(
        el("span", { className: "item-title", textContent: run.id }),
        el("span", {
          className: "item-meta",
          textContent: `${run.n_configs} конфигов × ${run.n_runs} прогонов · ${fmtDuration(run.elapsed_sec)}`
            + (run.has_report ? " · отчёт" : ""),
        })
      );
      head.addEventListener("click", () => openRun(run.id));
      item.append(head);
      return item;
    }));
  } catch (error) {
    list.replaceChildren(el("p", { className: "message err", textContent: "Не удалось прочитать результаты: " + error.message }));
  }
}

async function openRun(runId) {
  try {
    const run = await api(`/runs/${runId}`);
    state.currentRun = run.id;
    $("run-detail-card").hidden = false;
    $("run-title").textContent = run.id;
    renderRunSummary(run);
    renderMetrics(run, {
      table: "run-metrics",
      plots: "run-plots",
      hypothesesBox: "run-hypotheses-box",
      hypotheses: "run-hypotheses",
    });
    const link = $("link-report");
    if (run.has_report) {
      link.href = `${API}/runs/${run.id}/file/report.md`;
      link.classList.remove("disabled");
    } else {
      link.href = "#";
      link.classList.add("disabled");
    }
    refreshRuns();
  } catch (error) {
    showMessage("err", "Не удалось открыть прогон: " + error.message);
  }
}

async function buildReport() {
  if (!state.currentRun) return;
  const button = $("btn-report");
  button.disabled = true;
  try {
    const run = await api(`/runs/${state.currentRun}/report`, { method: "POST" });
    $("run-hypotheses").replaceChildren(...(run.hypotheses || []).map(hypothesisCard));
    $("run-hypotheses-box").hidden = !(run.hypotheses || []).length;
    const link = $("link-report");
    link.href = `${API}/runs/${run.id}/file/report.md`;
    link.classList.remove("disabled");
    showMessage("ok", "Отчёт построен: report.md и report.json лежат в каталоге прогона.");
    refreshRuns();
  } catch (error) {
    showMessage("err", "Отчёт не построен: " + error.message);
  } finally {
    button.disabled = false;
  }
}

function bindEvents() {
  $("tabs").addEventListener("click", (event) => {
    const tab = event.target.closest(".tab");
    if (!tab) return;
    for (const item of document.querySelectorAll(".tab")) item.classList.toggle("active", item === tab);
    const name = tab.dataset.tab;
    $("tab-builder").hidden = name !== "builder";
    $("tab-results").hidden = name !== "results";
    if (name === "results") refreshRuns();
  });

  $("env-kind").addEventListener("change", (event) => {
    state.envKind = event.target.value;
    state.envParams = defaultsForEnv(state.envKind);
    state.badFields.clear();
    renderEnv();
    onSpecChanged();
  });

  $("env-preset").addEventListener("change", (event) => {
    const preset = envSchema().presets.find((item) => item.key === event.target.value);
    if (!preset) return;
    Object.assign(state.envParams, preset.params);
    renderEnv();
    onSpecChanged();
  });

  $("btn-agents-all").addEventListener("click", () => {
    for (const agent of state.schema.agents) {
      if (!state.agents[agent.key]) state.agents[agent.key] = agentDefaults(agent.key);
    }
    renderAgents();
    onSpecChanged();
  });

  $("btn-policies-reset").addEventListener("click", () => {
    state.policies = {};
    for (const key of state.schema.defaults.policies) state.policies[key] = policyDefaults(key);
    renderPolicies();
    onSpecChanged();
  });

  const bindNumber = (id, field) => {
    $(id).addEventListener("change", (event) => {
      const raw = event.target.value.trim();
      const value = raw === "" ? null : Math.round(Number(raw));
      if (value !== null && Number.isFinite(value)) state.exp[field] = value;
      saveLocal();
      schedulePreview();
    });
  };
  bindNumber("exp-runs", "runs");
  bindNumber("exp-steps", "steps");
  bindNumber("exp-seed", "seed");
  bindNumber("exp-jobs", "jobs");

  $("exp-title").addEventListener("input", (event) => {
    state.exp.title = event.target.value;
    saveLocal();
  });
  $("exp-plots").addEventListener("change", (event) => {
    state.exp.plots = event.target.checked;
    saveLocal();
    schedulePreview();
  });
  $("exp-oracle").addEventListener("change", (event) => {
    state.exp.add_oracle = event.target.checked;
    saveLocal();
    schedulePreview();
  });

  $("btn-validate").addEventListener("click", validateSpec);
  $("btn-run").addEventListener("click", startJob);
  $("btn-cancel").addEventListener("click", cancelJob);
  $("btn-save").addEventListener("click", downloadSpec);
  $("btn-refresh-runs").addEventListener("click", refreshRuns);
  $("btn-report").addEventListener("click", buildReport);

  $("file-load").addEventListener("change", async (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    try {
      const spec = JSON.parse(await file.text());
      setSpec(spec);
      renderAll();
      showMessage("ok", `Загружена конфигурация ${file.name}.`);
    } catch (error) {
      showMessage("err", "Не удалось прочитать JSON: " + error.message);
    }
    event.target.value = "";
  });
}

async function boot() {
  bindEvents();
  try {
    const [schema, health] = await Promise.all([api("/schema"), api("/health")]);
    state.schema = schema;
    $("version").textContent =
      `v${schema.version} · оценка ${fmtInt(health.steps_per_second)} шагов/с на процесс`;
    $("results-dir").textContent = health.results_dir;
    setSpec(loadLocal() || specFromDefaults());
    renderAll();
    refreshRuns();
  } catch (error) {
    showMessage("err", "Не удалось загрузить схему интерфейса: " + error.message);
  }
}

document.addEventListener("DOMContentLoaded", boot);









