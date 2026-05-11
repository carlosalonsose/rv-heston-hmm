const state = {
  snapshot: null,
  calibration: null,
  selectedTicker: null,
  refreshSeconds: 60,
  countdown: 60,
  refreshInFlight: false,
  health: null,
};

const titles = {
  quotes: "Cotizaciones Kalshi",
  calibration: "Calibracion del modelo",
  distribution: "Distribucion simulada",
  edge: "Edge neto",
};

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab-button").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    document.getElementById(button.dataset.tab).classList.add("active");
    document.getElementById("pageTitle").textContent = titles[button.dataset.tab];
    if (button.dataset.tab === "distribution") loadDistribution();
  });
});

document.getElementById("refreshNow").addEventListener("click", refreshSnapshot);
document.getElementById("refreshDistribution").addEventListener("click", loadDistribution);
document.getElementById("marketSelect").addEventListener("change", (event) => {
  state.selectedTicker = event.target.value;
  loadDistribution();
});
document.getElementById("saveCalibration").addEventListener("click", saveCalibration);
document.getElementById("recalibrateModel").addEventListener("click", recalibrateModel);

setInterval(() => {
  if (document.hidden) return;
  state.countdown -= 1;
  if (state.countdown <= 0) {
    refreshSnapshot();
  }
  document.getElementById("countdown").textContent = `${state.countdown}s`;
}, 1000);

refreshSnapshot();

async function refreshSnapshot() {
  if (state.refreshInFlight) return;
  state.refreshInFlight = true;
  setStatus("Cargando", "warn");
  state.countdown = state.refreshSeconds;
  try {
    const response = await fetch("/api/snapshot");
    state.snapshot = await response.json();
    state.calibration = structuredClone(state.snapshot.calibration);
    renderQuotes();
    renderCalibration();
    renderEdges();
    await refreshHealth();
    if (!state.selectedTicker && state.snapshot.markets.length) {
      state.selectedTicker = state.snapshot.markets[0].ticker;
    }
    setStatus(`Actualizado ${new Date().toLocaleTimeString()}`, "ok");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    state.refreshInFlight = false;
  }
}

async function refreshHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("health unavailable");
    state.health = await response.json();
    renderHealth();
  } catch (error) {
    state.health = { status: "ERROR", error: error.message };
    renderHealth();
  }
}

function renderHealth() {
  const root = document.getElementById("healthSummary");
  const health = state.health || {};
  const stats = health.snapshot || {};
  const status = health.status || "UNKNOWN";
  const statusClass = status === "OK" ? "ok" : status === "ERROR" ? "error" : "warn";
  const latency = Number.isFinite(stats.latency_ms) ? `${stats.latency_ms.toFixed(0)}ms` : "-";
  const markets = Number.isFinite(stats.markets) ? stats.markets : "-";
  const alerts = Number.isFinite(stats.alerts) ? stats.alerts : "-";
  const warnings = Array.isArray(health.calibration_warnings)
    ? health.calibration_warnings.length
    : Number.isFinite(health.calibration_warnings)
      ? health.calibration_warnings
      : 0;
  root.className = `health-summary ${statusClass}`;
  root.innerHTML = `
    <div><span>Health</span><strong>${status}</strong></div>
    <div><span>Latencia</span><strong>${latency}</strong></div>
    <div><span>Mercados</span><strong>${markets}</strong></div>
    <div><span>Alertas</span><strong>${alerts}</strong></div>
    <div><span>Warnings</span><strong>${warnings}</strong></div>
  `;
}

function renderQuotes() {
  const markets = state.snapshot.markets || [];
  const body = document.getElementById("quotesBody");
  const select = document.getElementById("marketSelect");
  body.innerHTML = "";
  select.innerHTML = "";

  markets.forEach((market) => {
    const option = document.createElement("option");
    option.value = market.ticker;
    option.textContent = market.ticker;
    select.appendChild(option);
    if (!state.selectedTicker) state.selectedTicker = market.ticker;
  });
  select.value = state.selectedTicker || "";

  markets.forEach((market) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${market.ticker}</td>
      <td>${money(market.spot)}</td>
      <td>${money(market.barrier)}</td>
      <td>${market.spot_source || "-"}</td>
      <td>${prob(market.yes_bid)}</td>
      <td>${prob(market.yes_ask)}</td>
      <td>${prob(market.spread)}</td>
      <td>${prob(market.model_probability)}</td>
      <td>${signalPill(market.signal || market.status)}</td>
    `;
    body.appendChild(row);
  });
}

function renderCalibration() {
  if (!state.calibration) return;
  renderInputs("signalInputs", state.calibration.signal, ["taker_fee_rate", "slippage", "model_buffer", "min_edge", "min_liquidity"], "signal");
  renderInputs("simulationInputs", state.calibration.simulation, ["paths", "steps_per_day", "annualization_days"], "simulation");
  renderInputs("scannerInputs", state.calibration.scanner, ["max_markets", "initial_variance", "min_liquidity_contracts", "min_edge_alert", "max_spread"], "scanner");
  renderCalibrationMeta();
  const regimeRoot = document.getElementById("regimeInputs");
  regimeRoot.innerHTML = "";
  state.calibration.regime.params.forEach((regime, index) => {
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<h4>${regime.name || `Regime ${index + 1}`}</h4><div class="form-grid"></div>`;
    regimeRoot.appendChild(panel);
    const grid = panel.querySelector(".form-grid");
    ["mu", "kappa", "theta", "vol_of_vol", "rho", "jump_intensity", "jump_mean", "jump_std"].forEach((key) => {
      grid.appendChild(inputFor(`regime.params.${index}.${key}`, key, regime[key]));
    });
  });
}

function renderCalibrationMeta() {
  const root = document.getElementById("calibrationMeta");
  const meta = state.calibration.metadata?.historical_calibration;
  if (!meta) {
    root.textContent = "Parametros manuales: todavia no hay calibracion historica guardada.";
    return;
  }
  const realized = meta.realized_vol_1h ? ` | rv1h ${pct(meta.realized_vol_1h)} | v0 ${pct(meta.initial_vol)}` : "";
  const jumps = Number.isFinite(meta.jump_intensity) ? meta.jump_intensity.toFixed(2) : "-";
  const warnings = Array.isArray(meta.warnings) && meta.warnings.length ? ` | warnings: ${meta.warnings.join("; ")}` : "";
  root.textContent = `${meta.mode || "historical"} ${meta.ticker} | ${meta.period} ${meta.interval} | obs ${meta.observations} | sigma ${pct(meta.sigma_hist)}${realized} | kappa ${num(meta.kappa_estimate)} | vov ${num(meta.vol_of_vol_estimate)} | rho ${num(meta.rho_estimate)} | jumps ${jumps}/ano | ${new Date(meta.timestamp_utc).toLocaleString()}${warnings}`;
}

function renderInputs(rootId, source, keys, prefix) {
  const root = document.getElementById(rootId);
  root.innerHTML = "";
  keys.forEach((key) => root.appendChild(inputFor(`${prefix}.${key}`, key, source[key])));
}

function inputFor(path, label, value) {
  const wrapper = document.createElement("label");
  wrapper.textContent = label;
  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  input.value = value;
  input.dataset.path = path;
  input.addEventListener("input", updateCalibrationValue);
  wrapper.appendChild(input);
  return wrapper;
}

function updateCalibrationValue(event) {
  const parts = event.target.dataset.path.split(".");
  let ref = state.calibration;
  for (let i = 0; i < parts.length - 1; i += 1) {
    ref = ref[pathPart(parts[i])];
  }
  ref[pathPart(parts[parts.length - 1])] = Number(event.target.value);
}

async function saveCalibration() {
  setStatus("Guardando calibracion", "warn");
  const response = await fetch("/api/calibration", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.calibration),
  });
  if (!response.ok) {
    setStatus("No se pudo guardar", "error");
    return;
  }
  await refreshSnapshot();
}

async function recalibrateModel() {
  setStatus("Recalibrando BTC", "warn");
  const response = await fetch("/api/recalibrate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mode: "short_dte",
      ticker: "BTC-USD",
      short_period: "7d",
      short_interval: "5m",
      long_period: "45d",
      long_interval: "1h",
    }),
  });
  if (!response.ok) {
    const text = await response.text();
    setStatus(`No se pudo recalibrar: ${text.slice(0, 120)}`, "error");
    return;
  }
  await refreshSnapshot();
}

function renderEdges() {
  renderAlerts();
  const root = document.getElementById("edgeCards");
  root.innerHTML = "";
  const markets = [...(state.snapshot.markets || [])].sort((a, b) => {
    return Math.max(b.buy_yes_edge || -1, b.sell_yes_edge || -1) - Math.max(a.buy_yes_edge || -1, a.sell_yes_edge || -1);
  });
  markets.forEach((market) => {
    const card = document.createElement("article");
    card.className = "edge-card";
    card.innerHTML = `
      <strong>${market.ticker}</strong>
      <div style="margin-top:8px">${signalPill(market.signal || market.status)}</div>
      <div class="edge-metrics">
        ${metric("Buy YES edge", pct(market.buy_yes_edge))}
        ${metric("Sell YES edge", pct(market.sell_yes_edge))}
        ${metric("Spread", pct(market.spread))}
        ${metric("Fee buy", pct(market.buy_fee))}
        ${metric("Buffer", pct(market.model_buffer))}
        ${metric("P modelo", pct(market.model_probability))}
        ${metric("Liquidez", liquidity(market))}
      </div>
    `;
    root.appendChild(card);
  });
}

function renderAlerts() {
  const root = document.getElementById("alertCards");
  root.innerHTML = "";
  const alerts = state.snapshot.alerts || [];
  if (!alerts.length) {
    const card = document.createElement("article");
    card.className = "edge-card";
    card.innerHTML = `<strong>Sin alertas accionables</strong><p class="muted" style="margin-top:8px">No hay edge neto por encima del umbral con liquidez suficiente.</p>`;
    root.appendChild(card);
    return;
  }
  alerts.forEach((market) => {
    const card = document.createElement("article");
    card.className = "edge-card";
    card.innerHTML = `
      <strong>${market.ticker}</strong>
      <div style="margin-top:8px">${signalPill(market.signal)}</div>
      <div class="edge-metrics">
        ${metric("Edge neto", pct(market.edge_net))}
        ${metric("Ask", pct(market.yes_ask))}
        ${metric("Bid", pct(market.yes_bid))}
        ${metric("Spread", pct(market.spread))}
        ${metric("Liquidez", liquidity(market))}
        ${metric("P modelo", pct(market.model_probability))}
      </div>
    `;
    root.appendChild(card);
  });
}

async function loadDistribution() {
  if (!state.selectedTicker) return;
  setStatus("Recalculando distribucion", "warn");
  try {
    const response = await fetch(`/api/distribution?ticker=${encodeURIComponent(state.selectedTicker)}`);
    const data = await response.json();
    if (data.status !== "OK") throw new Error(data.reason || "distribution error");
    drawDistribution(data);
    renderPercentiles(data);
    setStatus(`Distribucion lista ${new Date().toLocaleTimeString()}`, "ok");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

function drawDistribution(data) {
  const canvas = document.getElementById("distributionChart");
  const ctx = canvas.getContext("2d");
  const counts = data.histogram.counts;
  const edges = data.histogram.edges;
  const width = canvas.width;
  const height = canvas.height;
  const pad = 44;
  const maxCount = Math.max(...counts, 1);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#dbe3dc";
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, height - pad);
  ctx.lineTo(width - pad, height - pad);
  ctx.stroke();

  const plotWidth = width - pad * 2;
  const plotHeight = height - pad * 2;
  const barWidth = plotWidth / counts.length;
  counts.forEach((count, index) => {
    const barHeight = (count / maxCount) * plotHeight;
    ctx.fillStyle = "#0f766e";
    ctx.fillRect(pad + index * barWidth, height - pad - barHeight, Math.max(1, barWidth - 2), barHeight);
  });

  const min = edges[0];
  const max = edges[edges.length - 1];
  const barrierX = pad + ((data.barrier - min) / (max - min)) * plotWidth;
  if (barrierX >= pad && barrierX <= width - pad) {
    ctx.strokeStyle = "#9a3412";
    ctx.beginPath();
    ctx.moveTo(barrierX, pad);
    ctx.lineTo(barrierX, height - pad);
    ctx.stroke();
    ctx.fillStyle = "#9a3412";
    ctx.fillText("strike", barrierX + 6, pad + 14);
  }

  ctx.fillStyle = "#16201a";
  ctx.fillText(`P(modelo) ${pct(data.model_probability)}`, pad, 24);
  ctx.fillText(money(min), pad, height - 16);
  ctx.fillText(money(max), width - pad - 88, height - 16);
}

function renderPercentiles(data) {
  const root = document.getElementById("percentiles");
  root.innerHTML = "";
  Object.entries(data.percentiles).forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = "metric-row";
    row.innerHTML = `<span>${key}</span><strong>${money(value)}</strong>`;
    root.appendChild(row);
  });
}

function setStatus(text, mode) {
  const dot = document.getElementById("statusDot");
  dot.className = `dot ${mode === "ok" ? "ok" : mode === "error" ? "error" : ""}`;
  document.getElementById("statusText").textContent = text;
}

function signalPill(signal) {
  const cls = signal === "BUY_YES" ? "buy" : signal === "SELL_YES_OR_BUY_NO" ? "sell" : "";
  return `<span class="pill ${cls}">${signal || "-"}</span>`;
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

function prob(value) {
  return value === undefined || value === null ? "-" : value.toFixed(4);
}

function pct(value) {
  return value === undefined || value === null ? "-" : `${(value * 100).toFixed(2)}%`;
}

function num(value) {
  return value === undefined || value === null || !Number.isFinite(value) ? "-" : Number(value).toFixed(3);
}

function money(value) {
  return value === undefined || value === null ? "-" : `$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}

function pathPart(part) {
  const numeric = Number(part);
  return Number.isInteger(numeric) && String(numeric) === part ? numeric : part;
}

function liquidity(market) {
  const bidQty = market.yes_bid_qty || 0;
  const askQty = market.yes_ask_qty || 0;
  return `${Math.min(bidQty, askQty).toFixed(0)} ct`;
}
