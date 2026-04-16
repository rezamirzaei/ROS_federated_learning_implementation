"""
Flask application factory for the standalone web dashboard.

Serves the *same* HTML/JS/CSS templates used by the ROS2
:class:`WebDashboardNode`, so the UI is identical whether running with
or without a ROS2 installation.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from flask_cors import CORS

if TYPE_CHECKING:
    from ros_web.simulation import SimulationEngine

# Locate shared web assets (same templates the ROS2 node uses)
_WEB_DIR = Path(__file__).resolve().parent.parent / "src" / "fl_robots" / "fl_robots" / "web"
_TEMPLATE_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def create_app(simulation: "SimulationEngine") -> Flask:
    """Return a configured Flask app wired to *simulation*."""

    app = Flask(
        __name__,
        template_folder=str(_TEMPLATE_DIR),
        static_folder=str(_STATIC_DIR),
    )
    CORS(app)

    # Optionally attach Socket.IO
    socketio = None
    try:
        from flask_socketio import SocketIO
        socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    except ImportError:
        pass

    # ── Routes ──────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("dashboard.html")

    @app.route("/api/status")
    def api_status():
        return jsonify(simulation.status)

    @app.route("/api/command", methods=["POST"])
    def api_command():
        data = request.get_json(force=True)
        cmd = data.get("command", "")
        result = simulation.issue_command(cmd)
        return jsonify(result)

    @app.route("/api/trigger-aggregation", methods=["POST"])
    def api_trigger_aggregation():
        result = simulation.issue_command("publish_weights")
        return jsonify(result)

    @app.route("/api/update-hyperparameters", methods=["POST"])
    def api_update_hyperparameters():
        data = request.get_json(force=True)
        lr = float(data.get("learning_rate", 0))
        bs = int(data.get("batch_size", 0))
        ep = int(data.get("local_epochs", 0))
        # Apply to all virtual robots
        import torch.optim as optim
        for r in simulation.robots.values():
            if lr > 0:
                for pg in r.optimizer.param_groups:
                    pg["lr"] = lr
        return jsonify({"success": True, "message": f"Updated LR={lr}, BS={bs}, Epochs={ep}"})

    @app.route("/api/robots")
    def api_robots():
        return jsonify(list(simulation.robots.keys()))

    @app.route("/api/digital-twin")
    def api_digital_twin():
        twin_path = _RESULTS_DIR / "digital_twin.png"
        if twin_path.exists():
            return send_file(str(twin_path), mimetype="image/png")
        return "", 404

    @app.route("/api/download-results")
    def api_download_results():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in ("aggregation_history.csv", "robot_metrics.json",
                         "training_summary.json", "digital_twin.png",
                         "aggregation_history.json"):
                fp = _RESULTS_DIR / name
                if fp.exists():
                    zf.write(str(fp), name)
        buf.seek(0)
        return send_file(buf, mimetype="application/zip",
                         as_attachment=True, download_name="fl_results.zip")

    # ── WebSocket push (if available) ───────────────────────────────

    if socketio is not None:
        import threading

        def _push_loop():
            while True:
                try:
                    socketio.emit("status_update", simulation.status, namespace="/")
                except Exception:
                    pass
                socketio.sleep(1.5)

        @socketio.on("connect")
        def _on_connect():
            pass  # client connected

        # Start background push thread once
        socketio.start_background_task(_push_loop)

    return app

