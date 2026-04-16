const stateEls = {
  controller: document.getElementById("controller-state"),
  round: document.getElementById("round-id"),
  autopilot: document.getElementById("autopilot-state"),
  avgAccuracy: document.getElementById("avg-accuracy"),
  avgLoss: document.getElementById("avg-loss"),
  trackingError: document.getElementById("tracking-error"),
  divergence: document.getElementById("mean-divergence"),
  robotsGrid: document.getElementById("robots-grid"),
  messageList: document.getElementById("message-list"),
  digitalTwin: document.getElementById("digital-twin"),
  mpcSizeHint: document.getElementById("mpc-size-hint"),
  mpcExplainerBody: document.getElementById("mpc-explainer-body"),
};

const ROBOT_COLORS = ["#52b788", "#4dabf7", "#ffd166", "#c792ea", "#ff9f43", "#00c9a7", "#ff6b6b", "#9ab6ff"];
const colorFor = (id) => ROBOT_COLORS[(id.split("_").pop() * 1 - 1) % ROBOT_COLORS.length] || "#52b788";

// ───────────────── API helpers ─────────────────

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

async function sendCommand(command) {
  const response = await fetch("/api/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command }),
  });
  if (!response.ok) {
    throw new Error(`Command failed with ${response.status}`);
  }
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return Number(value).toFixed(digits);
}

// ───────────────── Chart.js setup ─────────────────

const CHART_COMMON = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  interaction: { intersect: false, mode: "index" },
  plugins: {
    legend: { labels: { color: "#cbd5e1", boxWidth: 12, font: { size: 11 } } },
    tooltip: { backgroundColor: "#0f1b2e", borderColor: "#1d3557", borderWidth: 1 },
  },
  scales: {
    x: { ticks: { color: "#8ca0ba", maxTicksLimit: 8 }, grid: { color: "rgba(140,160,186,0.1)" } },
    y: { ticks: { color: "#8ca0ba" }, grid: { color: "rgba(140,160,186,0.1)" } },
  },
};

function makeLineChart(canvasId, datasets, yLabel) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: "line",
    data: { labels: [], datasets },
    options: {
      ...CHART_COMMON,
      scales: {
        ...CHART_COMMON.scales,
        y: { ...CHART_COMMON.scales.y, title: { display: !!yLabel, text: yLabel, color: "#8ca0ba" } },
      },
    },
  });
}

const charts = {};
const perRobotCharts = new Map();

function initCharts() {
  charts.globalLoss = makeLineChart("chart-global-loss", [
    { label: "train loss", data: [], borderColor: "#4dabf7", backgroundColor: "rgba(77,171,247,0.12)", tension: 0.3, pointRadius: 0 },
    { label: "val loss",   data: [], borderColor: "#ff9f43", backgroundColor: "rgba(255,159,67,0.12)", borderDash: [4, 3], tension: 0.3, pointRadius: 0 },
  ], "loss");
  charts.globalAcc = makeLineChart("chart-global-acc", [
    { label: "train acc (%)", data: [], borderColor: "#52b788", backgroundColor: "rgba(82,183,136,0.12)", tension: 0.3, pointRadius: 0 },
    { label: "val acc (%)",   data: [], borderColor: "#c792ea", backgroundColor: "rgba(199,146,234,0.12)", borderDash: [4, 3], tension: 0.3, pointRadius: 0 },
  ], "accuracy %");
  charts.mpcTrack  = makeLineChart("chart-mpc-track", [], "||x - x_ref||");
  charts.mpcEffort = makeLineChart("chart-mpc-effort", [], "||u_0||");
  charts.mpcSolve  = makeLineChart("chart-mpc-solve", [], "ms");
  charts.toaRmse   = makeLineChart("chart-toa-rmse", [
    { label: "mean RMSE", data: [], borderColor: "#ff6b6b", tension: 0.3, pointRadius: 0 },
  ], "m");
  charts.toaGap    = makeLineChart("chart-toa-gap", [
    { label: "consensus gap", data: [], borderColor: "#ffd166", tension: 0.3, pointRadius: 0 },
  ], "m");
}

// ───────────────── Update helpers ─────────────────

function updateGlobalCharts(globalSeries) {
  if (!charts.globalLoss || !charts.globalAcc) return;
  const labels = globalSeries.map((p) => `r${p.round_id}`);
  charts.globalLoss.data.labels = labels;
  charts.globalLoss.data.datasets[0].data = globalSeries.map((p) => p.mean_loss);
  charts.globalLoss.data.datasets[1].data = globalSeries.map((p) => p.val_loss);
  charts.globalLoss.update("none");
  charts.globalAcc.data.labels = labels;
  charts.globalAcc.data.datasets[0].data = globalSeries.map((p) => p.mean_accuracy);
  charts.globalAcc.data.datasets[1].data = globalSeries.map((p) => p.val_accuracy);
  charts.globalAcc.update("none");
}

function groupByRobot(series) {
  const byRobot = new Map();
  for (const d of series) {
    if (!byRobot.has(d.robot_id)) byRobot.set(d.robot_id, []);
    byRobot.get(d.robot_id).push(d);
  }
  return byRobot;
}

function updateMPCCharts(mpcHistory, system) {
  if (!charts.mpcTrack || !mpcHistory) return;
  const byRobot = groupByRobot(mpcHistory);
  // Build a common tick axis (union of all ticks).
  const tickSet = new Set();
  mpcHistory.forEach((d) => tickSet.add(d.tick));
  const ticks = [...tickSet].sort((a, b) => a - b);
  const labels = ticks.map((t) => `t${t}`);

  const buildDatasets = (field) => {
    const datasets = [];
    for (const [rid, entries] of byRobot.entries()) {
      const lookup = new Map(entries.map((e) => [e.tick, e[field]]));
      datasets.push({
        label: rid,
        data: ticks.map((t) => lookup.get(t) ?? null),
        borderColor: colorFor(rid),
        backgroundColor: colorFor(rid) + "22",
        tension: 0.25,
        pointRadius: 0,
        spanGaps: true,
      });
    }
    return datasets;
  };

  charts.mpcTrack.data.labels = labels;
  charts.mpcTrack.data.datasets = buildDatasets("tracking_error");
  charts.mpcTrack.update("none");
  charts.mpcEffort.data.labels = labels;
  charts.mpcEffort.data.datasets = buildDatasets("control_effort");
  charts.mpcEffort.update("none");
  charts.mpcSolve.data.labels = labels;
  charts.mpcSolve.data.datasets = buildDatasets("qp_solve_time_ms");
  charts.mpcSolve.update("none");

  if (system && stateEls.mpcSizeHint) {
    stateEls.mpcSizeHint.textContent =
      `${system.planner_kind} — N=${system.n_robots} robots × H=${system.horizon} horizon; ` +
      `${system.n_variables} decision vars, ${system.n_constraints} box constraints, ` +
      `mean solve ${fmt(system.mean_solve_time_ms, 2)} ms`;
  }
}

function updateTOACharts(toaHistory) {
  if (!charts.toaRmse || !toaHistory) return;
  const labels = toaHistory.map((p) => `t${p.tick}`);
  charts.toaRmse.data.labels = labels;
  charts.toaRmse.data.datasets[0].data = toaHistory.map((p) => p.mean_rmse);
  charts.toaRmse.update("none");
  charts.toaGap.data.labels = labels;
  charts.toaGap.data.datasets[0].data = toaHistory.map((p) => p.consensus_gap);
  charts.toaGap.update("none");
}

// ───────────────── Robots grid (with per-robot mini chart) ─────────────────

function renderRobots(robots, robotHistories) {
  stateEls.robotsGrid.innerHTML = robots
    .map((robot) => {
      const statusClass = robot.is_training ? "robot-status" : "robot-status paused";
      const statusLabel = robot.is_training ? "Training active" : "Navigation only";
      return `
        <article class="robot-card" data-robot-id="${robot.robot_id}">
          <h3 style="color:${colorFor(robot.robot_id)}">${robot.robot_id}</h3>
          <div class="robot-meta">
            <div><span class="label">Accuracy</span><br>${fmt(robot.accuracy, 1)}%</div>
            <div><span class="label">Loss</span><br>${fmt(robot.training_loss, 3)}</div>
            <div><span class="label">Round</span><br>${robot.training_round}</div>
            <div><span class="label">Tracking</span><br>${fmt(robot.last_tracking_error, 3)}</div>
          </div>
          <div class="robot-chart-wrap"><canvas class="robot-chart"></canvas></div>
          <span class="${statusClass}">${statusLabel}</span>
        </article>
      `;
    })
    .join("");

  // Instantiate / update per-robot mini charts.
  stateEls.robotsGrid.querySelectorAll(".robot-card").forEach((card) => {
    const rid = card.dataset.robotId;
    const canvas = card.querySelector(".robot-chart");
    let chart = perRobotCharts.get(rid);
    if (!chart || chart.canvas !== canvas) {
      chart = new Chart(canvas, {
        type: "line",
        data: {
          labels: [],
          datasets: [{
            label: "loss",
            data: [],
            borderColor: colorFor(rid),
            backgroundColor: colorFor(rid) + "22",
            tension: 0.3,
            pointRadius: 0,
            fill: true,
          }],
        },
        options: {
          ...CHART_COMMON,
          plugins: { legend: { display: false }, tooltip: CHART_COMMON.plugins.tooltip },
          scales: {
            x: { display: false },
            y: { ticks: { color: "#8ca0ba", maxTicksLimit: 3, font: { size: 9 } }, grid: { color: "rgba(140,160,186,0.08)" } },
          },
        },
      });
      perRobotCharts.set(rid, chart);
    }
    const series = robotHistories[rid] || [];
    chart.data.labels = series.map((p) => p.tick);
    chart.data.datasets[0].data = series.map((p) => p.local_loss);
    chart.update("none");
  });
}

// ───────────────── Messages & Digital twin ─────────────────

function renderMessages(messages) {
  stateEls.messageList.innerHTML = [...messages]
    .reverse()
    .map((message) => {
      const time = new Date(message.timestamp * 1000).toLocaleTimeString();
      return `
        <article class="message-item">
          <header>
            <span class="message-topic">${message.topic}</span>
            <span class="message-source">${message.source}</span>
            <span>${time}</span>
          </header>
          <pre>${JSON.stringify(message.payload, null, 2)}</pre>
        </article>
      `;
    })
    .join("");
}

function twinPoint(x, y) { return `${x},${-y}`; }

function renderDigitalTwin(snapshot) {
  const leader = snapshot.system.leader_position;
  const toa = snapshot.localization && snapshot.localization.current;

  const robotMarkup = snapshot.robots
    .map((robot) => {
      const pathPoints = robot.predicted_path.map((p) => twinPoint(p.x, p.y)).join(" ");
      const color = colorFor(robot.robot_id);
      return `
        <polyline points="${pathPoints}" fill="none" stroke="${color}" stroke-opacity="0.55" stroke-width="0.03" stroke-dasharray="0.08 0.05" />
        <line x1="${robot.pose.x}" y1="${-robot.pose.y}" x2="${robot.goal.x}" y2="${-robot.goal.y}" stroke="rgba(82,183,136,0.25)" stroke-width="0.02" />
        <circle cx="${robot.goal.x}" cy="${-robot.goal.y}" r="0.09" fill="rgba(255,209,102,0.18)" />
        <circle cx="${robot.pose.x}" cy="${-robot.pose.y}" r="0.14" fill="${color}" />
        <text x="${robot.pose.x}" y="${-robot.pose.y - 0.24}" text-anchor="middle" fill="#edf2f7" font-size="0.23">${robot.robot_id}</text>
      `;
    })
    .join("");

  let toaMarkup = "";
  if (toa) {
    const estimatesMarkup = toa.estimates
      .map((e) => `<circle cx="${e.x}" cy="${-e.y}" r="0.08" fill="#c792ea" fill-opacity="0.7" stroke="#c792ea" stroke-width="0.015"/>`)
      .join("");
    // Star shape for ground truth (5-point).
    const star = (cx, cy, R, r) => {
      const pts = [];
      for (let i = 0; i < 10; i++) {
        const ang = -Math.PI / 2 + (i * Math.PI) / 5;
        const rr = i % 2 === 0 ? R : r;
        pts.push(`${cx + rr * Math.cos(ang)},${-(cy + rr * Math.sin(ang))}`);
      }
      return pts.join(" ");
    };
    toaMarkup = `
      ${estimatesMarkup}
      <polygon points="${star(toa.target.x, toa.target.y, 0.22, 0.1)}" fill="#ffd166" stroke="#fff" stroke-width="0.02"/>
      <text x="${toa.target.x}" y="${-toa.target.y - 0.32}" text-anchor="middle" fill="#ffd166" font-size="0.2">target</text>
    `;
  }

  stateEls.digitalTwin.innerHTML = `
    <defs>
      <pattern id="grid" width="0.5" height="0.5" patternUnits="userSpaceOnUse">
        <path d="M 0.5 0 L 0 0 0 0.5" fill="none" stroke="rgba(140,160,186,0.16)" stroke-width="0.01" />
      </pattern>
    </defs>
    <rect x="-4" y="-3" width="8" height="6" fill="url(#grid)" />
    <path d="M -3 0 C -1 -1.3, 1 1.3, 3 0" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="0.04" />
    <circle cx="${leader.x}" cy="${-leader.y}" r="0.18" fill="#ff6b6b" />
    <text x="${leader.x}" y="${-leader.y - 0.28}" text-anchor="middle" fill="#ff6b6b" font-size="0.23">leader</text>
    ${robotMarkup}
    ${toaMarkup}
  `;
}

// ───────────────── Explainer ─────────────────

async function loadExplainer() {
  try {
    const e = await fetchJSON("/api/mpc/explainer");
    const listConstraints = (e.constraints || []).map((c) => `<li><code>${c}</code></li>`).join("");
    const listWeights = Object.entries(e.weights || {}).map(([k, v]) => `<li><b>${k}</b> — ${v}</li>`).join("");
    stateEls.mpcExplainerBody.innerHTML = `
      <p>${e.summary}</p>
      <p><b>Decision variables:</b> <code>${e.decision_variables}</code></p>
      <p><b>Dynamics:</b> <code>${e.dynamics}</code></p>
      <p><b>Objective:</b> <code>${e.objective}</code></p>
      <p><b>Constraints:</b></p><ul>${listConstraints}</ul>
      <p><b>Weights:</b></p><ul>${listWeights}</ul>
      <p><b>Solver:</b> ${e.solver}</p>
      <p><b>Distributed aspect:</b> ${e.distributed_aspect}</p>
    `;
  } catch (err) {
    stateEls.mpcExplainerBody.innerHTML = `<p style="color:#ff6b6b">${err.message}</p>`;
  }
}

// ───────────────── Main render ─────────────────

function render(snapshot) {
  stateEls.controller.textContent = snapshot.system.controller_state;
  stateEls.round.textContent = snapshot.system.current_round;
  stateEls.autopilot.textContent = snapshot.system.autopilot ? "ON" : "OFF";
  stateEls.avgAccuracy.textContent = `${fmt(snapshot.metrics.avg_accuracy, 1)}%`;
  stateEls.avgLoss.textContent = fmt(snapshot.metrics.avg_loss, 3);
  stateEls.trackingError.textContent = fmt(snapshot.metrics.mean_tracking_error, 3);
  const divergence = snapshot.metrics.last_aggregation ? snapshot.metrics.last_aggregation.mean_divergence : 0;
  stateEls.divergence.textContent = fmt(divergence, 3);

  const history = snapshot.history || { global: [], robots: {} };
  renderRobots(snapshot.robots, history.robots || {});
  renderMessages(snapshot.messages);
  renderDigitalTwin(snapshot);
  updateGlobalCharts(history.global || []);
  if (snapshot.mpc) updateMPCCharts(snapshot.mpc.history || [], snapshot.mpc.system);
  if (snapshot.localization) updateTOACharts(snapshot.localization.history || []);
}

async function refresh() {
  try {
    const snapshot = await fetchJSON("/api/status");
    render(snapshot);
  } catch (error) {
    stateEls.messageList.innerHTML = `<article class="message-item"><pre>${error.message}</pre></article>`;
  }
}

document.querySelectorAll("[data-command]").forEach((button) => {
  button.addEventListener("click", async () => {
    try {
      await sendCommand(button.dataset.command);
      await refresh();
    } catch (error) {
      stateEls.messageList.innerHTML = `<article class="message-item"><pre>${error.message}</pre></article>`;
    }
  });
});

// Bootstrap
initCharts();
loadExplainer();
refresh();
window.setInterval(refresh, 1400);

