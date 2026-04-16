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
};

async function fetchStatus() {
  const response = await fetch("/api/status");
  if (!response.ok) {
    throw new Error(`Status request failed with ${response.status}`);
  }
  return response.json();
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
  return Number(value).toFixed(digits);
}

function renderRobots(robots) {
  stateEls.robotsGrid.innerHTML = robots
    .map((robot) => {
      const statusClass = robot.is_training ? "robot-status" : "robot-status paused";
      const statusLabel = robot.is_training ? "Training active" : "Navigation only";
      return `
        <article class="robot-card">
          <h3>${robot.robot_id}</h3>
          <div class="robot-meta">
            <div><span class="label">Accuracy</span><br>${fmt(robot.accuracy, 1)}%</div>
            <div><span class="label">Loss</span><br>${fmt(robot.training_loss, 3)}</div>
            <div><span class="label">Round</span><br>${robot.training_round}</div>
            <div><span class="label">Messages</span><br>${robot.messages_sent}</div>
            <div><span class="label">Tracking</span><br>${fmt(robot.last_tracking_error, 3)}</div>
            <div><span class="label">Plan cost</span><br>${fmt(robot.last_plan_cost, 3)}</div>
          </div>
          <span class="${statusClass}">${statusLabel}</span>
        </article>
      `;
    })
    .join("");
}

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

function twinPoint(x, y) {
  return `${x},${-y}`;
}

function renderDigitalTwin(snapshot) {
  const leader = snapshot.system.leader_position;
  const robotMarkup = snapshot.robots
    .map((robot) => {
      const pathPoints = robot.predicted_path.map((point) => twinPoint(point.x, point.y)).join(" ");
      return `
        <polyline points="${pathPoints}" fill="none" stroke="rgba(77,171,247,0.55)" stroke-width="0.03" stroke-dasharray="0.08 0.05" />
        <line x1="${robot.pose.x}" y1="${-robot.pose.y}" x2="${robot.goal.x}" y2="${-robot.goal.y}" stroke="rgba(82,183,136,0.25)" stroke-width="0.02" />
        <circle cx="${robot.goal.x}" cy="${-robot.goal.y}" r="0.09" fill="rgba(255,209,102,0.18)" />
        <circle cx="${robot.pose.x}" cy="${-robot.pose.y}" r="0.14" fill="#52b788" />
        <text x="${robot.pose.x}" y="${-robot.pose.y - 0.24}" text-anchor="middle" fill="#edf2f7" font-size="0.23">${robot.robot_id}</text>
      `;
    })
    .join("");

  stateEls.digitalTwin.innerHTML = `
    <defs>
      <pattern id="grid" width="0.5" height="0.5" patternUnits="userSpaceOnUse">
        <path d="M 0.5 0 L 0 0 0 0.5" fill="none" stroke="rgba(140,160,186,0.16)" stroke-width="0.01" />
      </pattern>
    </defs>
    <rect x="-4" y="-3" width="8" height="6" fill="url(#grid)" />
    <path d="M -3 0 C -1 -1.3, 1 1.3, 3 0" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="0.04" />
    <circle cx="${leader.x}" cy="${-leader.y}" r="0.18" fill="#ff6b6b" />
    <text x="${leader.x}" y="${-leader.y - 0.28}" text-anchor="middle" fill="#ffd166" font-size="0.23">leader</text>
    ${robotMarkup}
  `;
}

function render(snapshot) {
  stateEls.controller.textContent = snapshot.system.controller_state;
  stateEls.round.textContent = snapshot.system.current_round;
  stateEls.autopilot.textContent = snapshot.system.autopilot ? "ON" : "OFF";
  stateEls.avgAccuracy.textContent = `${fmt(snapshot.metrics.avg_accuracy, 1)}%`;
  stateEls.avgLoss.textContent = fmt(snapshot.metrics.avg_loss, 3);
  stateEls.trackingError.textContent = fmt(snapshot.metrics.mean_tracking_error, 3);
  const divergence = snapshot.metrics.last_aggregation ? snapshot.metrics.last_aggregation.mean_divergence : 0;
  stateEls.divergence.textContent = fmt(divergence, 3);
  renderRobots(snapshot.robots);
  renderMessages(snapshot.messages);
  renderDigitalTwin(snapshot);
}

async function refresh() {
  try {
    const snapshot = await fetchStatus();
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

refresh();
window.setInterval(refresh, 1400);

