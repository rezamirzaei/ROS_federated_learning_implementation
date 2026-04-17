#!/usr/bin/env python3
"""
Web Dashboard Node — MVC Architecture with WebSocket Real-time Updates.

This node provides a comprehensive web-based UI for the federated learning system:

Architecture (MVC):
- Model: ROS2 subscriber callbacks collect state from all nodes
- View: Jinja2 HTML templates + Chart.js + Canvas topology
- Controller: Flask routes + Socket.IO for bidirectional real-time communication

ROS2 Concepts Demonstrated:
- Subscribers (robot status, aggregation metrics, coordinator status)
- Publishers (training commands)
- Service Clients (TriggerAggregation, UpdateHyperparameters, GetModelInfo)
- QoS Profiles (reliable + transient local)
- Multi-threaded integration with Flask/Socket.IO
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .results_artifacts import DIGITAL_TWIN_IMAGE, iter_bundle_paths
from .ros_compat import (
    DurabilityPolicy,
    HistoryPolicy,
    MultiThreadedExecutor,
    Node,
    QoSProfile,
    ReentrantCallbackGroup,
    ReliabilityPolicy,
    String,
    rclpy,
)

try:
    from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
    from flask_cors import CORS

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask flask-cors")

try:
    from flask_socketio import SocketIO, emit

    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False

# Try importing custom service interfaces for service client integration
try:
    from fl_robots_interfaces.srv import GetModelInfo, TriggerAggregation, UpdateHyperparameters

    CUSTOM_INTERFACES = True
except ImportError:
    CUSTOM_INTERFACES = False


class WebDashboardNode(Node):
    """
    Web Dashboard Node — MVC Controller integrated with ROS2.

    Model: Collects state via ROS2 subscriptions
    View: Serves HTML templates with Chart.js and Canvas
    Controller: Flask routes + Socket.IO events for bidirectional communication
    """

    def __init__(self):
        super().__init__("web_dashboard")

        self.cb_group = ReentrantCallbackGroup()
        self.cb_group_clients = ReentrantCallbackGroup()

        # Parameters
        self.declare_parameter("port", 5000)
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("output_dir", "/ros2_ws/results")

        self.port = self.get_parameter("port").value
        self.host = self.get_parameter("host").value
        self.output_dir = self.get_parameter("output_dir").value

        self.get_logger().info(f"Initializing Web Dashboard on {self.host}:{self.port}")

        # ── Model: State storage ────────────────────────────────────
        self.state_lock = threading.Lock()
        self.robots: dict[str, dict] = {}
        self.coordinator_state = "IDLE"
        self.current_round = 0
        self.total_aggregations = 0
        self.mean_divergence = 0.0
        self.start_time = time.time()
        self.event_log: deque = deque(maxlen=200)
        self.loss_history: list[dict] = []
        self.acc_history: list[dict] = []

        # QoS
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── Subscribers ─────────────────────────────────────────────
        self.robot_status_sub = self.create_subscription(
            String,
            "/fl/robot_status",
            self.robot_status_callback,
            qos_reliable,
            callback_group=self.cb_group,
        )

        self.aggregation_sub = self.create_subscription(
            String,
            "/fl/aggregation_metrics",
            self.aggregation_callback,
            qos_reliable,
            callback_group=self.cb_group,
        )

        self.coordinator_sub = self.create_subscription(
            String,
            "/fl/coordinator_status",
            self.coordinator_callback,
            qos_reliable,
            callback_group=self.cb_group,
        )

        # ── Publisher for commands ──────────────────────────────────
        self.command_publisher = self.create_publisher(String, "/fl/training_command", qos_reliable)

        # ── Service Clients (for interactive control) ───────────────
        self.trigger_agg_client = None
        self.update_hp_clients: dict[str, Any] = {}

        if CUSTOM_INTERFACES:
            self.trigger_agg_client = self.create_client(
                TriggerAggregation, "/fl/trigger_aggregation", callback_group=self.cb_group_clients
            )
            self.get_logger().info("Service client: /fl/trigger_aggregation")

        # Socket.IO reference (set in _run_flask)
        self.socketio = None

        # Start Flask/SocketIO in separate thread
        if FLASK_AVAILABLE:
            self.flask_thread = threading.Thread(target=self._run_flask, daemon=True)
            self.flask_thread.start()
        else:
            self.get_logger().error("Flask not available, web dashboard disabled")

        # Timer to push updates via WebSocket
        self.push_timer = self.create_timer(1.5, self._push_ws_update, callback_group=self.cb_group)

        self.get_logger().info(f"Web Dashboard: http://{self.host}:{self.port}")
        self.get_logger().info(
            f"WebSocket: {'ENABLED' if SOCKETIO_AVAILABLE else 'DISABLED (polling fallback)'}"
        )
        self.get_logger().info(f"Service clients: {'ENABLED' if CUSTOM_INTERFACES else 'DISABLED'}")

    # ────────────────────────────────────────────────────────────────
    # Model: ROS2 Callbacks
    # ────────────────────────────────────────────────────────────────

    def robot_status_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            robot_id = data.get("robot_id")
            if not robot_id:
                return

            with self.state_lock:
                if robot_id not in self.robots:
                    self.robots[robot_id] = {}
                    self._add_event(f"Robot {robot_id} connected")

                self.robots[robot_id].update(
                    {
                        "is_training": data.get("is_training", False),
                        "loss": data.get("last_loss"),
                        "accuracy": data.get("last_accuracy"),
                        "rounds": data.get("training_round", 0),
                        "last_seen": time.time(),
                    }
                )
        except Exception as e:
            self.get_logger().error(f"Error in robot status callback: {e}")

    def aggregation_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self.state_lock:
                self.total_aggregations = data.get("round", self.total_aggregations)
                self.mean_divergence = data.get("mean_divergence", 0.0)
                self._add_event(f"Aggregation round {self.total_aggregations} complete")

                # Record history for charts
                snapshot_loss = {}
                snapshot_acc = {}
                for rid, r in self.robots.items():
                    snapshot_loss[rid] = {"loss": r.get("loss")}
                    snapshot_acc[rid] = {"accuracy": r.get("accuracy")}
                self.loss_history.append(
                    {"round": self.total_aggregations, "robots": snapshot_loss}
                )
                self.acc_history.append({"round": self.total_aggregations, "robots": snapshot_acc})
        except Exception as e:
            self.get_logger().error(f"Error in aggregation callback: {e}")

    def coordinator_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self.state_lock:
                old_state = self.coordinator_state
                self.coordinator_state = data.get("state", "UNKNOWN")
                self.current_round = data.get("current_round", 0)
                if old_state != self.coordinator_state:
                    self._add_event(f"Coordinator: {self.coordinator_state}")
        except Exception as e:
            self.get_logger().error(f"Error in coordinator callback: {e}")

    def _add_event(self, message: str):
        self.event_log.append({"time": time.strftime("%H:%M:%S"), "message": message})

    def _push_ws_update(self):
        """Timer callback: push state to all WebSocket clients."""
        if self.socketio:
            try:
                self.socketio.emit("status_update", self._get_status(), namespace="/")
            except Exception:
                pass

    # ────────────────────────────────────────────────────────────────
    # Model: State Retrieval
    # ────────────────────────────────────────────────────────────────

    def _get_status(self) -> dict:
        with self.state_lock:
            robots_data = {}
            total_loss = 0
            total_acc = 0
            best_acc = 0
            count = 0

            for rid, robot in self.robots.items():
                robots_data[rid] = robot.copy()
                if robot.get("loss") is not None:
                    total_loss += robot["loss"]
                    count += 1
                if robot.get("accuracy") is not None:
                    total_acc += robot["accuracy"]
                    best_acc = max(best_acc, robot["accuracy"])

            return {
                "coordinator_state": self.coordinator_state,
                "current_round": self.current_round,
                "total_aggregations": self.total_aggregations,
                "active_robots": len(self.robots),
                "mean_divergence": self.mean_divergence,
                "avg_loss": total_loss / count if count > 0 else None,
                "avg_accuracy": total_acc / count if count > 0 else None,
                "best_accuracy": best_acc if best_acc > 0 else None,
                "training_time": time.time() - self.start_time,
                "robots": robots_data,
                "events": list(self.event_log),
                "loss_history": self.loss_history[-50:],
                "acc_history": self.acc_history[-50:],
            }

    # ────────────────────────────────────────────────────────────────
    # Controller: Commands
    # ────────────────────────────────────────────────────────────────

    def _send_command(self, command: str):
        msg = String()
        msg.data = json.dumps(
            {"command": command, "round": self.current_round, "timestamp": time.time()}
        )
        self.command_publisher.publish(msg)
        self._add_event(f"Sent command: {command}")

    def _call_trigger_aggregation(self) -> dict:
        """Call TriggerAggregation service."""
        if not self.trigger_agg_client or not self.trigger_agg_client.wait_for_service(
            timeout_sec=1.0
        ):
            return {"success": False, "message": "Service not available"}

        req = TriggerAggregation.Request()
        req.force = True
        req.min_participants = 0
        # Fire-and-forget; attach done_callback in production to surface
        # the result back to the client.
        self.trigger_agg_client.call_async(req)
        # Non-blocking: return pending
        return {"success": True, "message": "Aggregation triggered (async)"}

    def _create_hp_client(self, robot_id: str):
        """Lazily create a service client for updating a robot's hyperparameters."""
        if robot_id not in self.update_hp_clients and CUSTOM_INTERFACES:
            self.update_hp_clients[robot_id] = self.create_client(
                UpdateHyperparameters,
                f"/fl/{robot_id}/update_hyperparameters",
                callback_group=self.cb_group_clients,
            )

    def _call_update_hyperparameters(self, lr: float, bs: int, epochs: int) -> dict:
        """Update hyperparameters on all known robots via services."""
        results = []
        for rid in list(self.robots.keys()):
            self._create_hp_client(rid)
            client = self.update_hp_clients.get(rid)
            if client and client.wait_for_service(timeout_sec=0.5):
                req = UpdateHyperparameters.Request()
                req.robot_id = rid
                req.learning_rate = lr
                req.batch_size = bs
                req.local_epochs = epochs
                req.samples_per_round = 0
                client.call_async(req)
                results.append(rid)

        if results:
            return {"success": True, "message": f"Updated {len(results)} robots: {results}"}
        return {"success": False, "message": "No robots available or service not ready"}

    # ────────────────────────────────────────────────────────────────
    # View: Flask + Socket.IO Server
    # ────────────────────────────────────────────────────────────────

    def _run_flask(self):
        """Run Flask web server with optional Socket.IO."""
        # Resolve template and static directories
        web_dir = Path(__file__).parent / "web"
        template_dir = web_dir / "templates"
        static_dir = web_dir / "static"

        app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
        CORS(app)

        # Same security headers baseline as standalone_web.py. Kept in sync
        # manually — both dashboards serve the same static asset family so
        # policies should not drift between them.
        @app.after_request
        def _security_headers(response):  # pragma: no cover - requires flask-cors
            response.headers.setdefault(
                "Content-Security-Policy",
                os.environ.get(
                    "FL_ROBOTS_CSP",
                    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                    "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                    "connect-src 'self' ws: wss:; font-src 'self' data:; "
                    "frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
                ),
            )
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("Referrer-Policy", "no-referrer")
            response.headers.setdefault(
                "Permissions-Policy",
                "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
                "magnetometer=(), microphone=(), payment=(), usb=()",
            )
            return response

        # Prometheus exposition — shares the global REGISTRY with
        # ``standalone_web.py`` and ``observability/metrics.py`` so scrapers
        # see the same metric names in both deployment modes.
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

            from .observability.metrics import REGISTRY

            @app.route("/metrics")
            def prom_metrics():  # pragma: no cover - scraped by tests elsewhere
                from flask import Response as _Resp

                return _Resp(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)
        except ImportError:  # pragma: no cover
            pass

        node = self

        # Initialize Socket.IO if available
        if SOCKETIO_AVAILABLE:
            socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
            node.socketio = socketio
        else:
            socketio = None

        # ── Routes (View + Controller) ──────────────────────────────

        @app.route("/")
        def index():
            return render_template("dashboard.html")

        @app.route("/api/status")
        def get_status():
            return jsonify(node._get_status())

        @app.route("/api/command", methods=["POST"])
        def send_command():
            data = request.get_json(silent=True) or {}
            cmd = data.get("command")
            if not cmd or not isinstance(cmd, str):
                return jsonify({"success": False, "error": "No command specified"}), 400

            _VALID_COMMANDS = {
                "start_training",
                "stop_training",
                "publish_weights",
            }
            if cmd not in _VALID_COMMANDS:
                return jsonify({"success": False, "error": f"Unknown command: {cmd}"}), 400

            node._send_command(cmd)
            return jsonify({"success": True, "command": cmd})

        @app.route("/api/trigger-aggregation", methods=["POST"])
        def trigger_aggregation():
            if CUSTOM_INTERFACES:
                result = node._call_trigger_aggregation()
                return jsonify(result)
            else:
                # Fallback: publish command
                node._send_command("publish_weights")
                return jsonify({"success": True, "message": "Published weights request"})

        @app.route("/api/update-hyperparameters", methods=["POST"])
        def update_hyperparameters():
            data = request.get_json()
            lr = float(data.get("learning_rate", 0))
            bs = int(data.get("batch_size", 0))
            ep = int(data.get("local_epochs", 0))

            if CUSTOM_INTERFACES:
                result = node._call_update_hyperparameters(lr, bs, ep)
                return jsonify(result)
            else:
                return jsonify({"success": False, "message": "Custom interfaces not available"})

        @app.route("/api/digital-twin")
        def get_digital_twin():
            twin_path = Path(node.output_dir) / DIGITAL_TWIN_IMAGE
            if twin_path.exists():
                return send_file(twin_path, mimetype="image/png")
            return "", 404

        @app.route("/api/download-results")
        def download_results():
            import io
            import zipfile

            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
                for filepath in iter_bundle_paths(node.output_dir):
                    zf.write(filepath, filepath.name)
            memory_file.seek(0)
            return send_file(
                memory_file,
                mimetype="application/zip",
                as_attachment=True,
                download_name="fl_results.zip",
            )

        @app.route("/api/robots")
        def get_robots():
            with node.state_lock:
                return jsonify(list(node.robots.keys()))

        # ── Start server ────────────────────────────────────────────
        if socketio:
            socketio.run(
                app, host=node.host, port=node.port, allow_unsafe_werkzeug=True, use_reloader=False
            )
        else:
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


if __name__ == "__main__":
    main()
