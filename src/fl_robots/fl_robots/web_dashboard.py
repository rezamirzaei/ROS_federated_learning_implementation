#!/usr/bin/env python3
"""
Web Dashboard for Federated Learning System

This node provides a web-based user interface for:
- Real-time monitoring of training progress
- System control (start/stop training)
- Viewing robot metrics and status
- Downloading results
- Interactive parameter adjustment

Uses Flask as the web framework with ROS2 integration.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import String

import json
import time
import threading
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque

try:
    from flask import Flask, render_template_string, jsonify, request, send_file
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask flask-cors")


# HTML Template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FL Robot Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #eee;
        }
        
        .header {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #0f3460;
        }
        
        .header h1 {
            color: #e94560;
            font-size: 2em;
            margin-bottom: 5px;
        }
        
        .header .subtitle {
            color: #888;
            font-size: 0.9em;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        
        .card h2 {
            color: #e94560;
            font-size: 1.2em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .stat-box {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        
        .stat-box:last-child {
            border-bottom: none;
        }
        
        .stat-label {
            color: #888;
        }
        
        .stat-value {
            font-size: 1.3em;
            font-weight: bold;
            color: #4ecca3;
        }
        
        .stat-value.warning {
            color: #f39c12;
        }
        
        .stat-value.danger {
            color: #e74c3c;
        }
        
        .robot-card {
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
        }
        
        .robot-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .robot-name {
            font-weight: bold;
            color: #4ecca3;
        }
        
        .robot-status {
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.8em;
        }
        
        .status-training {
            background: #e74c3c;
        }
        
        .status-idle {
            background: #3498db;
        }
        
        .status-complete {
            background: #27ae60;
        }
        
        .progress-bar {
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #e94560, #4ecca3);
            transition: width 0.3s ease;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: #e94560;
            color: white;
        }
        
        .btn-primary:hover {
            background: #ff6b6b;
            transform: translateY(-2px);
        }
        
        .btn-secondary {
            background: #0f3460;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #1a4a7a;
        }
        
        .btn-danger {
            background: #c0392b;
            color: white;
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .chart-container {
            height: 200px;
            position: relative;
        }
        
        .log-container {
            max-height: 300px;
            overflow-y: auto;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 10px;
            font-family: monospace;
            font-size: 0.85em;
        }
        
        .log-entry {
            padding: 3px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        
        .log-time {
            color: #666;
            margin-right: 10px;
        }
        
        .digital-twin-container {
            text-align: center;
        }
        
        .digital-twin-container img {
            max-width: 100%;
            border-radius: 10px;
            border: 2px solid rgba(255,255,255,0.1);
        }
        
        .refresh-notice {
            text-align: center;
            color: #666;
            font-size: 0.8em;
            margin-top: 10px;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .live-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #27ae60;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 1.5s infinite;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Federated Learning Dashboard</h1>
        <p class="subtitle"><span class="live-indicator"></span>Real-time monitoring and control</p>
    </div>
    
    <div class="container">
        <div class="grid">
            <!-- System Status -->
            <div class="card">
                <h2>📊 System Status</h2>
                <div class="stat-box">
                    <span class="stat-label">Coordinator State</span>
                    <span class="stat-value" id="coordinator-state">-</span>
                </div>
                <div class="stat-box">
                    <span class="stat-label">Current Round</span>
                    <span class="stat-value" id="current-round">0</span>
                </div>
                <div class="stat-box">
                    <span class="stat-label">Total Aggregations</span>
                    <span class="stat-value" id="total-aggregations">0</span>
                </div>
                <div class="stat-box">
                    <span class="stat-label">Active Robots</span>
                    <span class="stat-value" id="active-robots">0</span>
                </div>
                <div class="stat-box">
                    <span class="stat-label">Mean Divergence</span>
                    <span class="stat-value" id="mean-divergence">-</span>
                </div>
            </div>
            
            <!-- Training Metrics -->
            <div class="card">
                <h2>📈 Training Metrics</h2>
                <div class="stat-box">
                    <span class="stat-label">Average Loss</span>
                    <span class="stat-value" id="avg-loss">-</span>
                </div>
                <div class="stat-box">
                    <span class="stat-label">Average Accuracy</span>
                    <span class="stat-value" id="avg-accuracy">-</span>
                </div>
                <div class="stat-box">
                    <span class="stat-label">Best Accuracy</span>
                    <span class="stat-value" id="best-accuracy">-</span>
                </div>
                <div class="stat-box">
                    <span class="stat-label">Training Time</span>
                    <span class="stat-value" id="training-time">-</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="accuracy-progress" style="width: 0%"></div>
                </div>
            </div>
            
            <!-- Controls -->
            <div class="card">
                <h2>🎮 Controls</h2>
                <div class="controls">
                    <button class="btn btn-primary" onclick="sendCommand('start_training')">
                        ▶️ Start Training
                    </button>
                    <button class="btn btn-danger" onclick="sendCommand('stop_training')">
                        ⏹️ Stop Training
                    </button>
                    <button class="btn btn-secondary" onclick="refreshData()">
                        🔄 Refresh
                    </button>
                    <button class="btn btn-secondary" onclick="downloadResults()">
                        📥 Download Results
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Robots Grid -->
        <div class="card">
            <h2>🤖 Robot Agents</h2>
            <div class="grid" id="robots-container">
                <p style="color: #666;">Waiting for robots to connect...</p>
            </div>
        </div>
        
        <!-- Digital Twin -->
        <div class="card">
            <h2>🌐 Digital Twin Visualization</h2>
            <div class="digital-twin-container">
                <img id="digital-twin-img" src="/api/digital-twin" alt="Digital Twin" 
                     onerror="this.style.display='none'; document.getElementById('twin-error').style.display='block';">
                <p id="twin-error" style="display:none; color:#666;">Digital twin image not available yet</p>
                <p class="refresh-notice">Updates every 5 seconds</p>
            </div>
        </div>
        
        <!-- Event Log -->
        <div class="card">
            <h2>📋 Event Log</h2>
            <div class="log-container" id="event-log">
                <div class="log-entry">
                    <span class="log-time">--:--:--</span>
                    Waiting for events...
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Auto-refresh data
        function refreshData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(err => console.error('Error fetching data:', err));
        }
        
        function updateDashboard(data) {
            // System status
            document.getElementById('coordinator-state').textContent = data.coordinator_state || '-';
            document.getElementById('current-round').textContent = data.current_round || 0;
            document.getElementById('total-aggregations').textContent = data.total_aggregations || 0;
            document.getElementById('active-robots').textContent = data.active_robots || 0;
            document.getElementById('mean-divergence').textContent = 
                data.mean_divergence ? data.mean_divergence.toFixed(4) : '-';
            
            // Training metrics
            document.getElementById('avg-loss').textContent = 
                data.avg_loss ? data.avg_loss.toFixed(4) : '-';
            document.getElementById('avg-accuracy').textContent = 
                data.avg_accuracy ? data.avg_accuracy.toFixed(1) + '%' : '-';
            document.getElementById('best-accuracy').textContent = 
                data.best_accuracy ? data.best_accuracy.toFixed(1) + '%' : '-';
            document.getElementById('training-time').textContent = 
                data.training_time ? formatTime(data.training_time) : '-';
            
            // Progress bar
            const progress = data.avg_accuracy || 0;
            document.getElementById('accuracy-progress').style.width = progress + '%';
            
            // Update robots
            updateRobots(data.robots || {});
            
            // Update digital twin image (cache bust)
            const img = document.getElementById('digital-twin-img');
            img.src = '/api/digital-twin?' + new Date().getTime();
        }
        
        function updateRobots(robots) {
            const container = document.getElementById('robots-container');
            
            if (Object.keys(robots).length === 0) {
                container.innerHTML = '<p style="color: #666;">Waiting for robots to connect...</p>';
                return;
            }
            
            let html = '';
            for (const [id, robot] of Object.entries(robots)) {
                const statusClass = robot.is_training ? 'status-training' : 
                                   robot.accuracy > 60 ? 'status-complete' : 'status-idle';
                const statusText = robot.is_training ? 'Training' : 
                                  robot.accuracy > 60 ? 'Ready' : 'Idle';
                
                html += `
                    <div class="robot-card">
                        <div class="robot-header">
                            <span class="robot-name">🤖 ${id}</span>
                            <span class="robot-status ${statusClass}">${statusText}</span>
                        </div>
                        <div class="stat-box">
                            <span class="stat-label">Loss</span>
                            <span>${robot.loss ? robot.loss.toFixed(4) : '-'}</span>
                        </div>
                        <div class="stat-box">
                            <span class="stat-label">Accuracy</span>
                            <span>${robot.accuracy ? robot.accuracy.toFixed(1) + '%' : '-'}</span>
                        </div>
                        <div class="stat-box">
                            <span class="stat-label">Rounds</span>
                            <span>${robot.rounds || 0}</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${robot.accuracy || 0}%"></div>
                        </div>
                    </div>
                `;
            }
            container.innerHTML = html;
        }
        
        function sendCommand(cmd) {
            fetch('/api/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: cmd})
            })
            .then(response => response.json())
            .then(data => {
                addLogEntry(`Command sent: ${cmd}`);
                refreshData();
            })
            .catch(err => addLogEntry(`Error: ${err}`));
        }
        
        function downloadResults() {
            window.location.href = '/api/download-results';
        }
        
        function addLogEntry(message) {
            const log = document.getElementById('event-log');
            const time = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `<span class="log-time">${time}</span>${message}`;
            log.insertBefore(entry, log.firstChild);
            
            // Keep only last 50 entries
            while (log.children.length > 50) {
                log.removeChild(log.lastChild);
            }
        }
        
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}m ${secs}s`;
        }
        
        // Initial load and auto-refresh
        refreshData();
        setInterval(refreshData, 2000);
        
        // Log initial connection
        addLogEntry('Dashboard connected');
    </script>
</body>
</html>
"""


class WebDashboardNode(Node):
    """
    Web Dashboard Node for Federated Learning System.

    Provides a web-based UI for monitoring and controlling the FL system.
    """

    def __init__(self):
        super().__init__('web_dashboard')

        self.cb_group = ReentrantCallbackGroup()

        # Parameters
        self.declare_parameter('port', 5000)
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('output_dir', '/ros2_ws/results')

        self.port = self.get_parameter('port').value
        self.host = self.get_parameter('host').value
        self.output_dir = self.get_parameter('output_dir').value

        self.get_logger().info(f'Initializing Web Dashboard on {self.host}:{self.port}')

        # State storage
        self.state_lock = threading.Lock()
        self.robots: Dict[str, Dict] = {}
        self.coordinator_state = "IDLE"
        self.current_round = 0
        self.total_aggregations = 0
        self.mean_divergence = 0.0
        self.start_time = time.time()
        self.event_log: deque = deque(maxlen=100)

        # QoS
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Subscribers
        self.robot_status_sub = self.create_subscription(
            String, '/fl/robot_status', self.robot_status_callback,
            qos_reliable, callback_group=self.cb_group
        )

        self.aggregation_sub = self.create_subscription(
            String, '/fl/aggregation_metrics', self.aggregation_callback,
            qos_reliable, callback_group=self.cb_group
        )

        self.coordinator_sub = self.create_subscription(
            String, '/fl/coordinator_status', self.coordinator_callback,
            qos_reliable, callback_group=self.cb_group
        )

        # Publisher for commands
        self.command_publisher = self.create_publisher(
            String, '/fl/training_command', qos_reliable
        )

        # Start Flask in separate thread
        if FLASK_AVAILABLE:
            self.flask_thread = threading.Thread(target=self._run_flask, daemon=True)
            self.flask_thread.start()
        else:
            self.get_logger().error('Flask not available, web dashboard disabled')

        self.get_logger().info(f'Web Dashboard available at http://{self.host}:{self.port}')

    def robot_status_callback(self, msg: String):
        """Handle robot status updates."""
        try:
            data = json.loads(msg.data)
            robot_id = data.get('robot_id')

            if not robot_id:
                return

            with self.state_lock:
                if robot_id not in self.robots:
                    self.robots[robot_id] = {}
                    self._add_event(f'Robot {robot_id} connected')

                self.robots[robot_id].update({
                    'is_training': data.get('is_training', False),
                    'loss': data.get('last_loss'),
                    'accuracy': data.get('last_accuracy'),
                    'rounds': data.get('training_round', 0),
                    'last_seen': time.time()
                })
        except Exception as e:
            self.get_logger().error(f'Error in robot status callback: {e}')

    def aggregation_callback(self, msg: String):
        """Handle aggregation metrics."""
        try:
            data = json.loads(msg.data)
            with self.state_lock:
                self.total_aggregations = data.get('round', self.total_aggregations)
                self.mean_divergence = data.get('mean_divergence', 0.0)
                self._add_event(f'Aggregation round {self.total_aggregations} complete')
        except Exception as e:
            self.get_logger().error(f'Error in aggregation callback: {e}')

    def coordinator_callback(self, msg: String):
        """Handle coordinator status."""
        try:
            data = json.loads(msg.data)
            with self.state_lock:
                old_state = self.coordinator_state
                self.coordinator_state = data.get('state', 'UNKNOWN')
                self.current_round = data.get('current_round', 0)

                if old_state != self.coordinator_state:
                    self._add_event(f'Coordinator state: {self.coordinator_state}')
        except Exception as e:
            self.get_logger().error(f'Error in coordinator callback: {e}')

    def _add_event(self, message: str):
        """Add event to log."""
        self.event_log.append({
            'time': time.strftime('%H:%M:%S'),
            'message': message
        })

    def _get_status(self) -> Dict:
        """Get current system status."""
        with self.state_lock:
            robots_data = {}
            total_loss = 0
            total_acc = 0
            best_acc = 0
            count = 0

            for rid, robot in self.robots.items():
                robots_data[rid] = robot.copy()
                if robot.get('loss') is not None:
                    total_loss += robot['loss']
                    count += 1
                if robot.get('accuracy') is not None:
                    total_acc += robot['accuracy']
                    best_acc = max(best_acc, robot['accuracy'])

            return {
                'coordinator_state': self.coordinator_state,
                'current_round': self.current_round,
                'total_aggregations': self.total_aggregations,
                'active_robots': len(self.robots),
                'mean_divergence': self.mean_divergence,
                'avg_loss': total_loss / count if count > 0 else None,
                'avg_accuracy': total_acc / count if count > 0 else None,
                'best_accuracy': best_acc if best_acc > 0 else None,
                'training_time': time.time() - self.start_time,
                'robots': robots_data,
                'events': list(self.event_log)
            }

    def _send_command(self, command: str):
        """Send training command."""
        msg = String()
        msg.data = json.dumps({
            'command': command,
            'round': self.current_round,
            'timestamp': time.time()
        })
        self.command_publisher.publish(msg)
        self._add_event(f'Sent command: {command}')

    def _run_flask(self):
        """Run Flask web server."""
        app = Flask(__name__)
        CORS(app)

        node = self  # Reference to ROS node

        @app.route('/')
        def index():
            return render_template_string(DASHBOARD_HTML)

        @app.route('/api/status')
        def get_status():
            return jsonify(node._get_status())

        @app.route('/api/command', methods=['POST'])
        def send_command():
            data = request.get_json()
            cmd = data.get('command')
            if cmd:
                node._send_command(cmd)
                return jsonify({'success': True, 'command': cmd})
            return jsonify({'success': False, 'error': 'No command specified'})

        @app.route('/api/digital-twin')
        def get_digital_twin():
            twin_path = f'{node.output_dir}/digital_twin.png'
            if os.path.exists(twin_path):
                return send_file(twin_path, mimetype='image/png')
            return '', 404

        @app.route('/api/download-results')
        def download_results():
            import zipfile
            import io

            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                for filename in ['aggregation_history.csv', 'robot_metrics.json',
                               'training_summary.json', 'digital_twin.png']:
                    filepath = f'{node.output_dir}/{filename}'
                    if os.path.exists(filepath):
                        zf.write(filepath, filename)

            memory_file.seek(0)
            return send_file(memory_file, mimetype='application/zip',
                           as_attachment=True, download_name='fl_results.zip')

        # Run Flask
        app.run(host=node.host, port=node.port, threaded=True, use_reloader=False)


def main(args=None):
    rclpy.init(args=args)

    node = WebDashboardNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
