#!/usr/bin/env python3
"""
Standalone web server — runs the full simulation dashboard **without** ROS2.

Uses the :class:`SimulationEngine` for state and the Flask app to serve the
MPC + federated-learning dashboard in a single process.

Routes
------
``GET  /``             – Single-page dashboard.
``GET  /api/health``   – Lightweight liveness probe.
``GET  /api/status``   – Full simulation snapshot (JSON).
``POST /api/command``  – Issue a simulation command.
``GET  /api/results``  – Download full results as JSON attachment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from flask import Flask, Response, jsonify, render_template, request
from pydantic import BaseModel, ConfigDict

from .simulation import SimulationEngine

logger = logging.getLogger(__name__)

__all__ = ["CommandRequest", "create_app", "main"]

_VALID_COMMANDS = frozenset({
    "start_training", "stop_training", "toggle_autopilot",
    "step", "disturbance", "reset",
})


class CommandRequest(BaseModel):
    """Pydantic model for ``POST /api/command`` payloads."""

    model_config = ConfigDict(frozen=True, slots=True)

    command: Literal[
        "start_training", "stop_training", "toggle_autopilot",
        "step", "disturbance", "reset",
    ]


def create_app(simulation: SimulationEngine | None = None) -> Flask:
    """Create the standalone Flask application.

    Parameters
    ----------
    simulation : Optional pre-configured engine.  A default engine is
                 created when *None*.
    """
    web_dir = Path(__file__).resolve().parent / "web"
    app = Flask(
        __name__,
        template_folder=str(web_dir / "templates"),
        static_folder=str(web_dir / "static"),
    )
    app.simulation = simulation or SimulationEngine()  # type: ignore[attr-defined]

    # ── Routes ───────────────────────────────────────────────────────

    @app.get("/")
    def index() -> str:
        return render_template("standalone.html")

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
        try:
            cmd = CommandRequest.model_validate(payload)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        try:
            app.simulation.issue_command(cmd.command)  # type: ignore[attr-defined]
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        return jsonify({"ok": True, "command": cmd.command})

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    print("\n🚀  Starting standalone FL + MPC dashboard at http://127.0.0.1:5000\n")
    app = create_app()
    try:
        app.run(host="127.0.0.1", port=5000, debug=False)
    finally:
        app.simulation.shutdown()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()

