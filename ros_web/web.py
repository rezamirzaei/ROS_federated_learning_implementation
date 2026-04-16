from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from .simulation import SimulationEngine

logger = logging.getLogger(__name__)

_VALID_COMMANDS = frozenset({
    "start_training", "stop_training", "toggle_autopilot",
    "step", "disturbance", "reset",
})


def create_app(simulation: SimulationEngine | None = None) -> Flask:
    base_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )
    app.simulation = simulation or SimulationEngine()  # type: ignore[attr-defined]

    @app.get("/")
    def index() -> str:
        return render_template("dashboard.html")

    @app.get("/api/health")
    def health():
        """Lightweight liveness probe."""
        return jsonify({"ok": True, "uptime_ticks": app.simulation.tick_count})  # type: ignore[attr-defined]

    @app.get("/api/status")
    def status():
        return jsonify(app.simulation.snapshot())  # type: ignore[attr-defined]

    @app.post("/api/command")
    def command():
        payload = request.get_json(silent=True) or {}
        command_name = payload.get("command")
        if not command_name:
            return jsonify({"ok": False, "error": "command is required"}), 400

        command_name = str(command_name)
        if command_name not in _VALID_COMMANDS:
            return jsonify({"ok": False, "error": f"Unknown command: {command_name}"}), 400

        try:
            app.simulation.issue_command(command_name)  # type: ignore[attr-defined]
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        return jsonify({"ok": True, "command": command_name})

    @app.get("/api/results")
    def results():
        body = json.dumps(app.simulation.export_results(), indent=2)  # type: ignore[attr-defined]
        return Response(
            body,
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=ros_showcase_results.json"},
        )

    return app
