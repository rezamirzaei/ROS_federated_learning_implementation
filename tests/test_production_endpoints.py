"""Tests for new production endpoints: /api/ready, OpenAPI, rate limiting."""

from __future__ import annotations

import time

import pytest
from fl_robots.simulation import SimulationEngine
from fl_robots.standalone_web import OPENAPI_SCHEMA, create_app


@pytest.fixture
def running_app():
    sim = SimulationEngine(num_robots=2, auto_start=True)
    # Give the background thread one real tick so readiness flips green.
    time.sleep(0.25)
    app = create_app(sim)
    yield app, sim
    sim.shutdown()


@pytest.fixture
def paused_app():
    sim = SimulationEngine(num_robots=2, auto_start=False)
    app = create_app(sim)
    yield app, sim
    sim.shutdown()


# ── Health / readiness ───────────────────────────────────────────────


def test_health_is_live_even_when_sim_paused(paused_app):
    app, _ = paused_app
    client = app.test_client()
    rv = client.get("/api/health")
    assert rv.status_code == 200
    assert rv.get_json()["ok"] is True


def test_ready_is_503_when_sim_not_running(paused_app):
    app, _ = paused_app
    client = app.test_client()
    rv = client.get("/api/ready")
    assert rv.status_code == 503
    body = rv.get_json()
    assert body["ok"] is False
    assert body["running"] is False


def test_ready_is_200_when_sim_stepping(running_app):
    app, _ = running_app
    client = app.test_client()
    rv = client.get("/api/ready")
    assert rv.status_code == 200
    body = rv.get_json()
    assert body["ok"] is True
    assert body["running"] is True
    assert body["last_tick_age_s"] < body["stale_threshold_s"]


# ── OpenAPI ─────────────────────────────────────────────────────────


def test_openapi_schema_is_served_and_valid_structure(paused_app):
    app, _ = paused_app
    client = app.test_client()
    rv = client.get("/api/openapi.json")
    assert rv.status_code == 200
    schema = rv.get_json()
    assert schema["openapi"].startswith("3.")
    # Critical endpoints are documented.
    for path in ("/api/health", "/api/ready", "/api/command", "/metrics"):
        assert path in schema["paths"], f"missing path in OpenAPI: {path}"
    # Command endpoint enumerates allowed values — guards against silent drift.
    allowed = schema["components"]["schemas"]["CommandRequest"]["properties"]["command"]["enum"]
    assert "start_training" in allowed
    assert "reset" in allowed


def test_openapi_constant_matches_endpoint(paused_app):
    """The in-memory constant and the served schema must not diverge."""
    app, _ = paused_app
    rv = app.test_client().get("/api/openapi.json")
    assert rv.get_json() == OPENAPI_SCHEMA


# ── Rate limiting ────────────────────────────────────────────────────


def test_command_endpoint_is_rate_limited(paused_app, monkeypatch):
    """Rapid bursts beyond the per-IP cap must return 429."""
    # Shrink the window so the test is fast and deterministic.
    from fl_robots import standalone_web

    monkeypatch.setattr(standalone_web, "_RATE_WINDOW_S", 60.0)
    monkeypatch.setattr(standalone_web, "_RATE_MAX_HITS", 3)

    # Rebuild app so it picks up the new limits.
    sim = SimulationEngine(num_robots=2, auto_start=False)
    app = standalone_web.create_app(sim)
    try:
        client = app.test_client()

        for _ in range(3):
            rv = client.post("/api/command", json={"command": "step"})
            assert rv.status_code == 200

        rv = client.post("/api/command", json={"command": "step"})
        assert rv.status_code == 429
        body = rv.get_json()
        assert body["ok"] is False
        assert "Rate limit" in body["error"]
    finally:
        sim.shutdown()
