#!/usr/bin/env python3
"""
Standalone web server — runs the full simulation dashboard **without** ROS2.

Uses the :class:`SimulationEngine` for state and the Flask app to serve the
MPC + federated-learning dashboard in a single process.

Launch with:
    python -m fl_robots.standalone_web          # or
    python main.py web
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request, Response

from .simulation import SimulationEngine


def create_app(simulation: SimulationEngine | None = None) -> Flask:
    """Create the standalone Flask application."""
    web_dir = Path(__file__).resolve().parent / "web"
    app = Flask(
        __name__,
        template_folder=str(web_dir / "templates"),
        static_folder=str(web_dir / "static"),
    )
    app.simulation = simulation or SimulationEngine()  # type: ignore[attr-defined]

    @app.get("/")
    def index() -> str:
        return render_template("standalone.html")

    @app.get("/api/status")
    def status():
        return jsonify(app.simulation.snapshot())  # type: ignore[attr-defined]

    @app.post("/api/command")
    def command():
        payload = request.get_json(silent=True) or {}
        command_name = payload.get("command")
        if not command_name:
            return jsonify({"ok": False, "error": "command is required"}), 400

        app.simulation.issue_command(str(command_name))  # type: ignore[attr-defined]
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


def main() -> None:
    """Entry point for the standalone web dashboard."""
    print("\n🚀  Starting standalone FL + MPC dashboard at http://127.0.0.1:5000\n")
    app = create_app()
    try:
        app.run(host="127.0.0.1", port=5000, debug=False)
    finally:
        app.simulation.shutdown()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()

