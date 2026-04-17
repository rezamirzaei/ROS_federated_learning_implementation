"""Regression tests for security hardening on the standalone Flask app.

These lock in that every response carries the baseline security headers,
so an accidental rewrite of ``create_app`` can't silently strip them.
"""

from __future__ import annotations

from typing import Any

from fl_robots.simulation import SimulationEngine
from fl_robots.standalone_web import create_app

_EXPECTED_HEADERS = {
    "Content-Security-Policy",
    "X-Frame-Options",
    "X-Content-Type-Options",
    "Referrer-Policy",
    "Permissions-Policy",
}


def _client() -> tuple[Any, Any]:
    sim = SimulationEngine(num_robots=2, auto_start=False)
    app = create_app(sim)
    return sim, app.test_client()


def test_security_headers_present_on_index() -> None:
    sim, client = _client()
    try:
        resp = client.get("/")
        assert resp.status_code == 200
        for name in _EXPECTED_HEADERS:
            assert name in resp.headers, f"missing {name} on /"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert "frame-ancestors 'none'" in resp.headers["Content-Security-Policy"]
        assert "'unsafe-inline'" not in resp.headers["Content-Security-Policy"]
    finally:
        sim.shutdown()


def test_security_headers_present_on_json_api() -> None:
    sim, client = _client()
    try:
        resp = client.get("/api/status")
        assert resp.status_code == 200
        for name in _EXPECTED_HEADERS:
            assert name in resp.headers, f"missing {name} on /api/status"
    finally:
        sim.shutdown()


def test_security_headers_present_on_error_response(csrf_headers: Any) -> None:
    """Even 4xx responses must carry the headers — they're user-facing."""
    sim, client = _client()
    try:
        resp = client.post("/api/command", json={}, headers=csrf_headers(client))
        assert resp.status_code == 400
        for name in _EXPECTED_HEADERS:
            assert name in resp.headers, f"missing {name} on 400 response"
    finally:
        sim.shutdown()


def test_csp_env_override(monkeypatch: Any) -> None:
    """FL_ROBOTS_CSP should override the default policy."""
    monkeypatch.setenv("FL_ROBOTS_CSP", "default-src 'none'")
    sim = SimulationEngine(num_robots=2, auto_start=False)
    try:
        app = create_app(sim)
        resp = app.test_client().get("/api/health")
        assert resp.headers["Content-Security-Policy"] == "default-src 'none'"
    finally:
        sim.shutdown()
