/**
 * FL Robot Dashboard — Client-side Controller (MVC: View + Controller)
 *
 * Uses WebSocket (Socket.IO) for real-time push updates from the ROS2 system.
 * Falls back to HTTP polling if WebSocket is unavailable.
 */

// ── Model ──────────────────────────────────────────────────────────
const DashboardModel = {
    lossHistory: [],
    accHistory: [],
    roundLabels: [],
    robots: {},
    events: [],
    coordinator_state: '—',
    current_round: 0,
    total_aggregations: 0,
    mean_divergence: 0,
};

// ── Charts ─────────────────────────────────────────────────────────
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
    scales: {
        x: { ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } },
    },
    plugins: { legend: { labels: { color: '#ccc' } } },
};

const lossChart = new Chart(document.getElementById('lossChart'), {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, title: { display: true, text: 'Loss', color: '#888' } } } },
});

const accChart = new Chart(document.getElementById('accChart'), {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, title: { display: true, text: 'Accuracy (%)', color: '#888' }, min: 0, max: 100 } } },
});

const robotColors = ['#e94560', '#4ecca3', '#f39c12', '#3498db', '#9b59b6', '#1abc9c', '#e67e22', '#2ecc71'];

// ── WebSocket Connection ───────────────────────────────────────────
let socket = null;
let usePolling = false;

function getCookie(name) {
    const prefix = `${name}=`;
    return document.cookie
        .split(';')
        .map((part) => part.trim())
        .find((part) => part.startsWith(prefix))
        ?.slice(prefix.length) ?? '';
}

function csrfHeaders() {
    const token = getCookie('fl_robots_dashboard_csrf_token');
    return token ? { 'X-CSRF-Token': token } : {};
}

function initSocket() {
    try {
        socket = io({ transports: ['websocket', 'polling'], reconnection: true });

        socket.on('connect', () => {
            document.getElementById('conn-indicator').classList.remove('disconnected');
            document.getElementById('conn-status').textContent = 'Live (WebSocket)';
            addLogEntry('WebSocket connected');
        });

        socket.on('disconnect', () => {
            document.getElementById('conn-indicator').classList.add('disconnected');
            document.getElementById('conn-status').textContent = 'Disconnected';
        });

        socket.on('status_update', (data) => {
            updateDashboard(data);
        });

        socket.on('event', (data) => {
            addLogEntry(data.message);
        });

        socket.on('connect_error', () => {
            // Fall back to polling
            usePolling = true;
            document.getElementById('conn-status').textContent = 'Live (Polling)';
        });

    } catch (e) {
        usePolling = true;
    }
}

// ── HTTP Polling Fallback ──────────────────────────────────────────
function pollData() {
    fetch('/api/status')
        .then(r => r.json())
        .then(data => updateDashboard(data))
        .catch(() => {});
}

// ── View: Update Dashboard ─────────────────────────────────────────
function updateDashboard(data) {
    // System status
    setText('coordinator-state', data.coordinator_state || '—');
    setText('current-round', data.current_round || 0);
    setText('total-aggregations', data.total_aggregations || 0);
    setText('active-robots', data.active_robots || 0);
    setText('mean-divergence', data.mean_divergence ? data.mean_divergence.toFixed(4) : '—');

    // Training metrics
    setText('avg-loss', data.avg_loss ? data.avg_loss.toFixed(4) : '—');
    setText('avg-accuracy', data.avg_accuracy ? data.avg_accuracy.toFixed(1) + '%' : '—');
    setText('best-accuracy', data.best_accuracy ? data.best_accuracy.toFixed(1) + '%' : '—');
    setText('training-time', data.training_time ? formatTime(data.training_time) : '—');
    document.getElementById('accuracy-progress').style.width = (data.avg_accuracy || 0) + '%';

    // Update robots
    if (data.robots) updateRobots(data.robots);

    // Update charts from history
    if (data.loss_history) updateCharts(data);

    // Digital twin image (cache bust)
    const img = document.getElementById('digital-twin-img');
    const twinError = document.getElementById('twin-error');
    if (img) {
        img.src = '/api/digital-twin?' + Date.now();
        img.classList.remove('hidden');
        if (twinError) twinError.classList.add('hidden');
    }

    // Draw topology
    drawTopology(data.robots || {});
}

function updateRobots(robots) {
    const container = document.getElementById('robots-container');
    if (!Object.keys(robots).length) {
        const empty = document.createElement('p');
        empty.className = 'muted-text';
        empty.textContent = 'Waiting for robots…';
        container.replaceChildren(empty);
        return;
    }

    const cards = [];
    for (const [id, r] of Object.entries(robots)) {
        const cls = r.is_training ? 'status-training' : (r.accuracy > 60 ? 'status-complete' : 'status-idle');
        const txt = r.is_training ? 'Training' : (r.accuracy > 60 ? 'Ready' : 'Idle');
        const card = document.createElement('div');
        card.className = 'robot-card';

        const header = document.createElement('div');
        header.className = 'robot-header';

        const name = document.createElement('span');
        name.className = 'robot-name';
        name.textContent = `🤖 ${id}`;

        const status = document.createElement('span');
        status.className = `robot-status ${cls}`;
        status.textContent = txt;

        header.append(name, status);
        card.appendChild(header);

        const stats = [
            ['Loss', r.loss != null ? r.loss.toFixed(4) : '—'],
            ['Accuracy', r.accuracy != null ? `${r.accuracy.toFixed(1)}%` : '—'],
            ['Rounds', String(r.rounds || 0)],
        ];

        stats.forEach(([label, value]) => {
            const box = document.createElement('div');
            box.className = 'stat-box';
            const labelEl = document.createElement('span');
            labelEl.className = 'stat-label';
            labelEl.textContent = label;
            const valueEl = document.createElement('span');
            valueEl.textContent = value;
            box.append(labelEl, valueEl);
            card.appendChild(box);
        });

        const bar = document.createElement('div');
        bar.className = 'progress-bar';
        const fill = document.createElement('div');
        fill.className = 'progress-fill';
        fill.style.width = `${r.accuracy || 0}%`;
        bar.appendChild(fill);
        card.appendChild(bar);
        cards.push(card);
    }
    container.replaceChildren(...cards);
}

function updateCharts(data) {
    if (!data.loss_history || !data.loss_history.length) return;

    // Build per-robot datasets
    const robotIds = Object.keys(data.loss_history[0].robots || {});
    const rounds = data.loss_history.map((_, i) => i + 1);

    // Loss chart
    lossChart.data.labels = rounds;
    lossChart.data.datasets = robotIds.map((rid, idx) => ({
        label: rid,
        data: data.loss_history.map(h => h.robots[rid]?.loss ?? null),
        borderColor: robotColors[idx % robotColors.length],
        tension: 0.3,
        pointRadius: 2,
    }));
    lossChart.update('none');

    // Accuracy chart
    accChart.data.labels = rounds;
    accChart.data.datasets = robotIds.map((rid, idx) => ({
        label: rid,
        data: data.acc_history.map(h => h.robots[rid]?.accuracy ?? null),
        borderColor: robotColors[idx % robotColors.length],
        tension: 0.3,
        pointRadius: 2,
    }));
    accChart.update('none');
}

// ── Topology Canvas ────────────────────────────────────────────────
function drawTopology(robots) {
    const canvas = document.getElementById('topologyCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Aggregator at center
    const cx = W / 2, cy = H / 2;
    const ids = Object.keys(robots);
    const n = ids.length;

    // Draw connections
    ids.forEach((id, i) => {
        const angle = (i / Math.max(n, 1)) * Math.PI * 2 - Math.PI / 2;
        const rx = cx + 150 * Math.cos(angle);
        const ry = cy + 150 * Math.sin(angle);
        const r = robots[id];
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(rx, ry);
        ctx.strokeStyle = r.is_training ? '#27ae60' : 'rgba(149,165,166,0.4)';
        ctx.lineWidth = r.is_training ? 3 : 1.5;
        ctx.stroke();
    });

    // Aggregator circle
    ctx.beginPath();
    ctx.arc(cx, cy, 30, 0, Math.PI * 2);
    ctx.fillStyle = '#9b59b6';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('AGG', cx, cy);

    // Robot circles
    ids.forEach((id, i) => {
        const angle = (i / Math.max(n, 1)) * Math.PI * 2 - Math.PI / 2;
        const rx = cx + 150 * Math.cos(angle);
        const ry = cy + 150 * Math.sin(angle);
        const r = robots[id];

        // Accuracy ring
        if (r.accuracy > 0) {
            ctx.beginPath();
            ctx.arc(rx, ry, 28, -Math.PI / 2, -Math.PI / 2 + (r.accuracy / 100) * Math.PI * 2);
            ctx.strokeStyle = '#27ae60';
            ctx.lineWidth = 4;
            ctx.stroke();
        }

        ctx.beginPath();
        ctx.arc(rx, ry, 22, 0, Math.PI * 2);
        ctx.fillStyle = r.is_training ? '#e74c3c' : (r.accuracy > 60 ? '#2ecc71' : '#3498db');
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px sans-serif';
        ctx.fillText(id.replace('robot_', 'R'), rx, ry);
    });
}

// ── Controller: Commands ───────────────────────────────────────────
function sendCommand(cmd) {
    fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...csrfHeaders() },
        credentials: 'same-origin',
        body: JSON.stringify({ command: cmd }),
    })
    .then(r => r.json())
    .then(() => addLogEntry(`Command: ${cmd}`))
    .catch(e => addLogEntry(`Error: ${e}`));
}

function triggerAggregation() {
    fetch('/api/trigger-aggregation', {
        method: 'POST',
        headers: { ...csrfHeaders() },
        credentials: 'same-origin',
    })
        .then(r => r.json())
        .then(d => addLogEntry(`Aggregation: ${d.message || 'triggered'}`))
        .catch(e => addLogEntry(`Error: ${e}`));
}

function updateHyperparams() {
    const lr = parseFloat(document.getElementById('param-lr').value);
    const bs = parseInt(document.getElementById('param-bs').value);
    const ep = parseInt(document.getElementById('param-epochs').value);
    fetch('/api/update-hyperparameters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...csrfHeaders() },
        credentials: 'same-origin',
        body: JSON.stringify({ learning_rate: lr, batch_size: bs, local_epochs: ep }),
    })
    .then(r => r.json())
    .then(d => addLogEntry(`Hyperparameters updated: LR=${lr}, BS=${bs}, Epochs=${ep}`))
    .catch(e => addLogEntry(`Error: ${e}`));
}

function downloadResults() {
    window.location.href = '/api/download-results';
}

// ── Helpers ────────────────────────────────────────────────────────
function setText(id, val) { document.getElementById(id).textContent = val; }

function formatTime(s) {
    const m = Math.floor(s / 60), sec = Math.floor(s % 60);
    return `${m}m ${sec}s`;
}

function addLogEntry(msg) {
    const log = document.getElementById('event-log');
    const t = new Date().toLocaleTimeString();
    const el = document.createElement('div');
    el.className = 'log-entry';
    const timeEl = document.createElement('span');
    timeEl.className = 'log-time';
    timeEl.textContent = t;
    el.appendChild(timeEl);
    el.appendChild(document.createTextNode(msg));
    log.insertBefore(el, log.firstChild);
    while (log.children.length > 100) log.removeChild(log.lastChild);
}

// ── Init ───────────────────────────────────────────────────────────
document.querySelectorAll('[data-command]').forEach((button) => {
    button.addEventListener('click', () => sendCommand(button.dataset.command));
});
document.getElementById('trigger-aggregation-btn')?.addEventListener('click', triggerAggregation);
document.getElementById('download-results-btn')?.addEventListener('click', downloadResults);
document.getElementById('update-hyperparams-btn')?.addEventListener('click', updateHyperparams);
document.getElementById('digital-twin-img')?.addEventListener('error', () => {
    document.getElementById('digital-twin-img')?.classList.add('hidden');
    document.getElementById('twin-error')?.classList.remove('hidden');
});

initSocket();
addLogEntry('Dashboard loaded');

// Polling fallback (always runs but only useful if WS fails)
setInterval(() => {
    if (usePolling || !socket || !socket.connected) pollData();
}, 2000);

// Initial fetch
pollData();

