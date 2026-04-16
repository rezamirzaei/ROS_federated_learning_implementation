#!/usr/bin/env python3
"""
Standalone web server — runs the full simulation dashboard **without** ROS2.

Uses :class:`SimulationEngine` for state and Flask to serve the MPC +
federated-learning dashboard in a single process.

Routes
------
``GET  /``               – Single-page dashboard.
``GET  /api/health``     – Liveness probe (process is up).
``GET  /api/ready``      – Readiness probe (simulation is stepping).
``GET  /api/status``     – Full simulation snapshot (JSON).
``POST /api/command``    – Issue a simulation command (Bearer auth optional, rate-limited).
``GET  /api/results``    – Download full results as JSON attachment.
``GET  /api/openapi.json`` – Machine-readable API schema (OpenAPI 3.1).
``GET  /metrics``        – Prometheus metrics exposition.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .controller import CommandRequest
from .observability.logging import configure_logging
from .observability.metrics import REGISTRY, update_from_snapshot
from .simulation import SimulationEngine

logger = logging.getLogger(__name__)

__all__ = ["MPC_EXPLAINER", "OPENAPI_SCHEMA", "CommandRequest", "create_app", "main"]

#: Set ``FL_ROBOTS_API_TOKEN=<token>`` in the environment to require a
#: matching ``Authorization: Bearer <token>`` header on mutating endpoints.
_AUTH_ENV_VAR = "FL_ROBOTS_API_TOKEN"

#: Rate-limit window / max requests for mutating endpoints. Tunable via env.
_RATE_WINDOW_S = float(os.environ.get("FL_ROBOTS_RATE_WINDOW_S", "10"))
_RATE_MAX_HITS = int(os.environ.get("FL_ROBOTS_RATE_MAX_HITS", "30"))

#: Readiness threshold — how stale the simulation tick can be before
#: ``/api/ready`` returns 503. Keeps Kubernetes from serving traffic to a
#: process whose background thread has hung.
_READY_STALE_S = float(os.environ.get("FL_ROBOTS_READY_STALE_S", "5"))


def _parse_limit(raw: str | None) -> int | None:
    """Parse a ``?limit=N`` query arg; clamp to a safe maximum."""
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return max(1, min(n, 2000))


#: Human-readable description of the MPC QP. Served at ``/api/mpc/explainer``
#: so the dashboard can render LaTeX-ish formulas without hard-coding them in
#: JS. Kept plain-text / Markdown so it renders in any client.
MPC_EXPLAINER: dict = {
    "title": "Distributed MPC for Formation Control",
    "summary": (
        "Each robot solves its own small quadratic program every tick and "
        "shares predicted trajectories with neighbours. The team tracks a "
        "moving leader while avoiding collisions."
    ),
    "decision_variables": (
        "u = [vx₀, vy₀, …, vx_{N-1}, vy_{N-1}]  ∈ ℝ^{2N}  "
        "— stacked velocity commands over the prediction horizon N."
    ),
    "dynamics": "x_{k+1} = x_k + dt · u_k  (double-integrator kinematics in 2D).",
    "objective": (
        "min_u  ½‖u‖²_R + ‖X(u) − X_ref‖²_Q + collision_penalty  "
        "where X(u) = x₀ + dt · (cumulative sum of u)."
    ),
    "constraints": [
        "‖u_k‖_∞ ≤ u_max   (per-axis speed box)",
        "Soft inter-robot separation penalised inside the objective "
        "via predicted neighbour positions (convex linearisation).",
    ],
    "weights": {
        "Q (tracking)": "Pulls the terminal position toward the formation slot.",
        "R (control)": "Damps large / jerky velocity commands.",
        "penalty (collision)": "Quadratic bump when a neighbour is within safe_distance.",
    },
    "solver": (
        "OSQP (sparse ADMM) with warm-starting from the previous tick's "
        "primal/dual solution. Falls back to a velocity-grid heuristic if "
        "scipy/osqp aren't installed."
    ),
    "distributed_aspect": (
        "Robots solve sequentially in a round-robin; each robot's predicted "
        "trajectory is broadcast to later robots so they see up-to-date "
        "neighbour positions when computing their collision penalty."
    ),
}

#: OpenAPI 3.1 schema for the mutating + introspection surface.
OPENAPI_SCHEMA: dict = {
    "openapi": "3.1.0",
    "info": {
        "title": "ROS Federated Learning + MPC Dashboard",
        "version": "1.0.0",
        "description": (
            "Control and inspection API for the standalone Flask dashboard. "
            "Mutating endpoints accept optional Bearer-token auth and are rate-limited."
        ),
        "license": {"name": "MIT"},
    },
    "components": {
        "securitySchemes": {
            "BearerAuth": {"type": "http", "scheme": "bearer"},
        },
        "schemas": {
            "CommandRequest": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [
                            "start_training",
                            "stop_training",
                            "reset",
                            "step",
                            "toggle_autopilot",
                            "trigger_disturbance",
                        ],
                    },
                },
            },
            "Error": {
                "type": "object",
                "required": ["ok", "error"],
                "properties": {
                    "ok": {"type": "boolean", "const": False},
                    "error": {"type": "string"},
                },
            },
        },
    },
    "paths": {
        "/api/health": {
            "get": {
                "summary": "Liveness probe",
                "responses": {"200": {"description": "Process is alive"}},
            }
        },
        "/api/ready": {
            "get": {
                "summary": "Readiness probe (simulation stepping recently)",
                "responses": {
                    "200": {"description": "Ready to serve traffic"},
                    "503": {"description": "Simulation is stale or shut down"},
                },
            }
        },
        "/api/status": {
            "get": {
                "summary": "Current simulation snapshot",
                "responses": {"200": {"description": "JSON snapshot"}},
            }
        },
        "/api/command": {
            "post": {
                "summary": "Issue a simulation command",
                "security": [{"BearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/CommandRequest"}
                        }
                    },
                },
                "responses": {
                    "200": {"description": "Accepted"},
                    "400": {
                        "description": "Validation error",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        },
                    },
                    "401": {"description": "Missing/invalid bearer token"},
                    "429": {"description": "Rate limit exceeded"},
                },
            }
        },
        "/api/results": {
            "get": {
                "summary": "Download full results JSON",
                "responses": {"200": {"description": "JSON attachment"}},
            }
        },
        "/metrics": {
            "get": {
                "summary": "Prometheus scrape endpoint",
                "responses": {"200": {"description": "text/plain metrics"}},
            }
        },
        "/api/history/global": {
            "get": {
                "summary": "Global FL training history (loss / accuracy per round)",
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "schema": {"type": "integer", "minimum": 1, "maximum": 2000},
                    }
                ],
                "responses": {"200": {"description": "JSON series"}},
            }
        },
        "/api/history/robots/{robot_id}": {
            "get": {
                "summary": "Per-robot training history (local loss / accuracy per tick)",
                "parameters": [
                    {"name": "robot_id", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                ],
                "responses": {"200": {"description": "JSON series"}},
            }
        },
        "/api/history/mpc": {
            "get": {
                "summary": "MPC per-robot diagnostics history",
                "parameters": [{"name": "limit", "in": "query", "schema": {"type": "integer"}}],
                "responses": {"200": {"description": "JSON series with system geometry"}},
            }
        },
        "/api/history/localization": {
            "get": {
                "summary": "Distributed TOA localization history",
                "parameters": [{"name": "limit", "in": "query", "schema": {"type": "integer"}}],
                "responses": {"200": {"description": "JSON series with target trajectory + RMSE"}},
            }
        },
        "/api/mpc/explainer": {
            "get": {
                "summary": "Human-readable description of the MPC QP",
                "responses": {"200": {"description": "JSON body (plain-text / Markdown)"}},
            }
        },
    },
}


def _get_simulation(app: Flask) -> SimulationEngine:
    sim = app.config["SIMULATION"]
    if not isinstance(sim, SimulationEngine):  # pragma: no cover
        raise RuntimeError("SIMULATION is not a SimulationEngine instance")
    return sim


def _check_auth() -> tuple[bool, str | None]:
    """Return ``(ok, reason)``. If no token configured, auth is disabled."""
    expected = os.environ.get(_AUTH_ENV_VAR, "").strip()
    if not expected:
        return True, None
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or token.strip() != expected:
        return False, "Invalid or missing bearer token"
    return True, None


class _SlidingWindowRateLimiter:
    """Per-IP sliding-window rate limiter.

    Simple in-memory implementation — adequate for the single-container
    default deployment. For multi-replica deployments front with an Nginx
    ``limit_req`` zone or a Redis-backed limiter.
    """

    def __init__(self, window_s: float, max_hits: int) -> None:
        self.window_s = window_s
        self.max_hits = max_hits
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            # Drop expired hits.
            while bucket and now - bucket[0] > self.window_s:
                bucket.popleft()
            if len(bucket) >= self.max_hits:
                return False
            bucket.append(now)
            return True


def create_app(simulation: SimulationEngine | None = None) -> Flask:
    """Create the standalone Flask application."""
    web_dir = Path(__file__).resolve().parent / "web"
    app = Flask(
        __name__,
        template_folder=str(web_dir / "templates"),
        static_folder=str(web_dir / "static"),
    )
    app.config["SIMULATION"] = simulation or SimulationEngine()
    limiter = _SlidingWindowRateLimiter(_RATE_WINDOW_S, _RATE_MAX_HITS)
    app.config["RATE_LIMITER"] = limiter

    @app.get("/")
    def index() -> str:
        return render_template("standalone.html")

    @app.get("/api/health")
    def health():
        """Liveness — the HTTP handler is up."""
        sim = _get_simulation(app)
        return jsonify({"ok": True, "uptime_ticks": sim.tick_count})

    @app.get("/api/ready")
    def ready():
        """Readiness — simulation thread has stepped recently."""
        sim = _get_simulation(app)
        last_tick_age = sim.seconds_since_last_tick()
        ready = sim.is_running() and last_tick_age <= _READY_STALE_S
        body = {
            "ok": ready,
            "running": sim.is_running(),
            "last_tick_age_s": round(last_tick_age, 3),
            "stale_threshold_s": _READY_STALE_S,
        }
        return jsonify(body), (200 if ready else 503)

    @app.get("/api/status")
    def status():
        sim = _get_simulation(app)
        snap = sim.snapshot()
        update_from_snapshot(snap)
        return jsonify(snap)

    @app.post("/api/command")
    def command():
        ok, reason = _check_auth()
        if not ok:
            return jsonify({"ok": False, "error": reason}), 401

        client_key = request.remote_addr or "anonymous"
        if not limiter.allow(client_key):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Rate limit exceeded",
                        "window_s": _RATE_WINDOW_S,
                        "max_hits": _RATE_MAX_HITS,
                    }
                ),
                429,
            )

        payload = request.get_json(silent=True) or {}
        try:
            cmd = CommandRequest.model_validate(payload)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        sim = _get_simulation(app)
        try:
            sim.issue_command(cmd.command)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        return jsonify({"ok": True, "command": cmd.command})

    @app.get("/api/results")
    def results():
        sim = _get_simulation(app)
        body = json.dumps(sim.export_results(), indent=2)
        return Response(
            body,
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=ros_showcase_results.json"},
        )

    @app.get("/api/openapi.json")
    def openapi_schema():
        return jsonify(OPENAPI_SCHEMA)

    @app.get("/api/history/global")
    def history_global():
        """Time-series of global aggregation metrics."""
        sim = _get_simulation(app)
        limit = _parse_limit(request.args.get("limit"))
        with sim._lock:
            series = list(sim.global_history)
        if limit is not None:
            series = series[-limit:]
        return jsonify({"series": [p.as_dict() for p in series]})

    @app.get("/api/history/robots/<robot_id>")
    def history_robot(robot_id: str):
        """Time-series of per-robot training metrics."""
        sim = _get_simulation(app)
        limit = _parse_limit(request.args.get("limit"))
        with sim._lock:
            buf = sim.robot_history.get(robot_id)
            series = list(buf) if buf else []
        if limit is not None:
            series = series[-limit:]
        return jsonify({"robot_id": robot_id, "series": [p.as_dict() for p in series]})

    @app.get("/api/history/mpc")
    def history_mpc():
        """Time-series of MPC per-robot diagnostics (tracking error, iters, solve time)."""
        sim = _get_simulation(app)
        limit = _parse_limit(request.args.get("limit"))
        with sim._lock:
            series = list(sim.mpc_robot_history)
            system = sim.last_mpc_system.as_dict() if sim.last_mpc_system else None
        if limit is not None:
            series = series[-limit:]
        return jsonify({"system": system, "series": [d.as_dict() for d in series]})

    @app.get("/api/history/localization")
    def history_localization():
        """Time-series of TOA localization (target trajectory, RMSE, consensus)."""
        sim = _get_simulation(app)
        limit = _parse_limit(request.args.get("limit"))
        with sim._lock:
            series = list(sim.toa_history)
        if limit is not None:
            series = series[-limit:]
        return jsonify(
            {
                "enabled": sim._toa_estimator is not None,
                "series": [s.as_dict() for s in series],
            }
        )

    @app.get("/api/mpc/explainer")
    def mpc_explainer():
        """Human-readable description of the MPC QP (decision vars, constraints)."""
        return jsonify(MPC_EXPLAINER)

    @app.get("/metrics")
    def metrics() -> Response:
        update_from_snapshot(_get_simulation(app).snapshot())
        return Response(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)

    return app


def _install_signal_handlers(sim: SimulationEngine) -> None:
    """Install SIGTERM / SIGINT handlers so Docker/Kubernetes can shut us down cleanly."""
    import signal

    def _handler(signum, _frame):  # pragma: no cover - signal-driven
        logger.info("Received signal %s — shutting down simulation", signum)
        try:
            sim.shutdown()
        finally:
            # Re-raise as SystemExit so Flask's dev server unwinds.
            raise SystemExit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # pragma: no cover - not on main thread
            pass


def main() -> None:
    """Entry point for the standalone web dashboard."""
    configure_logging()
    print("\n🚀  Starting standalone FL + MPC dashboard at http://127.0.0.1:5000\n")
    app = create_app()
    sim = _get_simulation(app)
    _install_signal_handlers(sim)
    try:
        app.run(host="127.0.0.1", port=5000, debug=False)
    finally:
        sim.shutdown()


if __name__ == "__main__":
    main()
