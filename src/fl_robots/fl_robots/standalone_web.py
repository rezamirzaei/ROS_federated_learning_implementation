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

import hmac
import json
import logging
import os
import secrets
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from flask import Flask, Response, g, jsonify, render_template, request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .controller import COMMAND_NAMES, CommandRequest
from .observability.logging import configure_logging
from .observability.metrics import (
    REGISTRY,
    fl_http_request_duration_seconds,
    fl_http_requests_total,
    update_from_snapshot,
)
from .observability.tracing import maybe_setup_tracing, span
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

_CSRF_COOKIE_NAME = "fl_robots_csrf_token"
_CSRF_HEADER_NAME = "X-CSRF-Token"


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
                        "enum": list(COMMAND_NAMES),
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
                            "application/json": {"schema": {"$ref": "#/components/schemas/Error"}}
                        },
                    },
                    "401": {"description": "Missing/invalid bearer token"},
                    "403": {"description": "Missing/invalid CSRF token"},
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
                    {
                        "name": "robot_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
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
                "description": (
                    "Returns historical TOA estimates when localization is enabled. "
                    "If the simulation was started without localization support, "
                    "the endpoint still returns 200 with enabled=false and an empty series."
                ),
                "parameters": [{"name": "limit", "in": "query", "schema": {"type": "integer"}}],
                "responses": {
                    "200": {
                        "description": "JSON series with target trajectory + RMSE",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["enabled", "series"],
                                    "properties": {
                                        "enabled": {"type": "boolean"},
                                        "series": {
                                            "type": "array",
                                            "description": (
                                                "TOA history points. Empty when localization "
                                                "is disabled or no samples have been produced yet."
                                            ),
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "tick": {"type": "integer"},
                                                    "timestamp": {"type": "number"},
                                                    "target": {
                                                        "type": "object",
                                                        "properties": {
                                                            "x": {"type": "number"},
                                                            "y": {"type": "number"},
                                                        },
                                                    },
                                                    "estimates": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "robot_id": {"type": "string"},
                                                                "x": {"type": "number"},
                                                                "y": {"type": "number"},
                                                                "error": {"type": "number"},
                                                            },
                                                        },
                                                    },
                                                    "mean_rmse": {"type": "number"},
                                                    "consensus_gap": {"type": "number"},
                                                },
                                            },
                                        },
                                    },
                                }
                            }
                        },
                    }
                },
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
    if scheme.lower() != "bearer":
        return False, "Invalid or missing bearer token"
    if not hmac.compare_digest(token.strip().encode("utf-8"), expected.encode("utf-8")):
        return False, "Invalid or missing bearer token"
    return True, None


def _csrf_protection_enabled() -> bool:
    """Use CSRF protection when mutating endpoints are open in dev mode."""
    return not bool(os.environ.get(_AUTH_ENV_VAR, "").strip())


def _check_csrf() -> tuple[bool, str | None]:
    """Double-submit cookie check for mutating endpoints when auth is open."""
    if not _csrf_protection_enabled():
        return True, None
    cookie_token = request.cookies.get(_CSRF_COOKIE_NAME, "").strip()
    header_token = request.headers.get(_CSRF_HEADER_NAME, "").strip()
    if not cookie_token or not header_token:
        return False, "Missing CSRF token"
    if not hmac.compare_digest(cookie_token.encode("utf-8"), header_token.encode("utf-8")):
        return False, "Invalid CSRF token"
    return True, None


def _bind_request_context(request_id: str) -> None:
    """Best-effort request-id propagation into structlog contextvars."""
    try:
        import structlog  # type: ignore

        structlog.contextvars.bind_contextvars(request_id=request_id)
    except ImportError:  # pragma: no cover - optional dependency
        pass


def _clear_request_context() -> None:
    """Clear per-request structlog bindings if structlog is installed."""
    try:
        import structlog  # type: ignore

        structlog.contextvars.clear_contextvars()
    except ImportError:  # pragma: no cover - optional dependency
        pass


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
    seed = int(os.environ.get("FL_ROBOTS_SEED", "42"))
    app = Flask(
        __name__,
        template_folder=str(web_dir / "templates"),
        static_folder=str(web_dir / "static"),
    )
    app.config["SIMULATION"] = simulation or SimulationEngine(seed=seed)
    limiter = _SlidingWindowRateLimiter(_RATE_WINDOW_S, _RATE_MAX_HITS)
    app.config["RATE_LIMITER"] = limiter

    # Best-effort OpenTelemetry setup — no-op unless FL_ROBOTS_OTEL=1 and
    # the `otel` extra is installed. Call is idempotent.
    maybe_setup_tracing(service_name="fl-robots-standalone")

    # ── Security headers ─────────────────────────────────────────────
    # Applied to every response (including static assets and errors).
    # CSP is deliberately tight: the dashboard ships its own JS/CSS
    # bundles and talks only to same-origin JSON endpoints. Override
    # via FL_ROBOTS_CSP if you embed in a parent frame.
    _default_csp = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "font-src 'self' data:; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "object-src 'none'"
    )
    _csp = os.environ.get("FL_ROBOTS_CSP", _default_csp)

    @app.before_request
    def _request_setup() -> None:
        g.request_started_at = time.perf_counter()
        g.request_id = request.headers.get("X-Request-ID", "").strip() or str(uuid.uuid4())
        g.csrf_token = request.cookies.get(_CSRF_COOKIE_NAME, "").strip() or secrets.token_urlsafe(
            32
        )
        _bind_request_context(g.request_id)

    @app.after_request
    def _security_headers(response: Response) -> Response:
        response.headers.setdefault("Content-Security-Policy", _csp)
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy",
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "magnetometer=(), microphone=(), payment=(), usb=()",
        )
        response.headers.setdefault("X-Request-ID", getattr(g, "request_id", str(uuid.uuid4())))
        if _csrf_protection_enabled() and request.cookies.get(_CSRF_COOKIE_NAME) != getattr(
            g, "csrf_token", None
        ):
            response.set_cookie(
                _CSRF_COOKIE_NAME,
                getattr(g, "csrf_token", secrets.token_urlsafe(32)),
                httponly=False,
                secure=request.is_secure,
                samesite="Strict",
                path="/",
            )
        path = request.url_rule.rule if request.url_rule is not None else request.path
        method = request.method
        status = str(response.status_code)
        fl_http_requests_total.labels(path=path, method=method, status=status).inc()
        started_at = getattr(g, "request_started_at", None)
        if isinstance(started_at, (int, float)):
            fl_http_request_duration_seconds.labels(
                path=path, method=method, status=status
            ).observe(max(0.0, time.perf_counter() - started_at))
        _clear_request_context()
        return response

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

        ok, reason = _check_csrf()
        if not ok:
            return jsonify({"ok": False, "error": reason}), 403

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
            with span("fl_robots.command", command=cmd.command):
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
