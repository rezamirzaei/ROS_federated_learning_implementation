"""End-to-end command flow test.

Drives the Flask app through a full lifecycle: issue start_training, step
the simulation, force an aggregation, scrape /metrics, then stop. This
exercises the seams between ``create_app``, ``SimulationEngine``, the
Prometheus registry, and the rate limiter in one test — catches the
class of regressions where individual unit tests pass but the boundary
wiring is broken.
"""

from __future__ import annotations

import re
from typing import Any

from fl_robots.simulation import SimulationEngine
from fl_robots.standalone_web import create_app


def test_command_lifecycle_and_metrics_scrape(csrf_headers: Any) -> None:
    sim = SimulationEngine(num_robots=3, auto_start=False)
    app = create_app(sim)
    client = app.test_client()

    try:
        assert client.get("/api/health").status_code == 200
        headers = csrf_headers(client)

        start = client.post("/api/command", json={"command": "start_training"}, headers=headers)
        assert start.status_code == 200, start.get_json()

        for _ in range(5):
            step = client.post("/api/command", json={"command": "step"}, headers=headers)
            assert step.status_code == 200

        status = client.get("/api/status")
        assert status.status_code == 200
        body = status.get_json()
        assert body["system"]["robot_count"] == 3

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        text = metrics.data.decode("utf-8")
        # Must expose the standalone FL/MPC metrics used by our alert rules.
        assert "fl_robot_count" in text
        assert "fl_training_active" in text
        assert "fl_tracking_error_bucket" in text
        # Prometheus exposition format starts with a HELP line.
        assert re.search(r"^# HELP ", text, re.MULTILINE)

        stop = client.post("/api/command", json={"command": "stop_training"}, headers=headers)
        assert stop.status_code == 200
    finally:
        sim.shutdown()


def test_rate_limiter_triggers_after_burst(monkeypatch: Any, csrf_headers: Any) -> None:
    monkeypatch.setenv("FL_ROBOTS_RATE_WINDOW_S", "60")
    monkeypatch.setenv("FL_ROBOTS_RATE_MAX_HITS", "3")
    # Reload to pick up the new env-derived defaults.
    import importlib

    import fl_robots.standalone_web as mod

    importlib.reload(mod)

    sim = SimulationEngine(num_robots=2, auto_start=False)
    app = mod.create_app(sim)
    client = app.test_client()

    try:
        headers = csrf_headers(client)
        for _ in range(3):
            r = client.post("/api/command", json={"command": "step"}, headers=headers)
            assert r.status_code == 200
        over = client.post("/api/command", json={"command": "step"}, headers=headers)
        assert over.status_code == 429
        assert over.get_json()["ok"] is False
    finally:
        sim.shutdown()


def test_auth_required_when_token_set(monkeypatch: Any) -> None:
    monkeypatch.setenv("FL_ROBOTS_API_TOKEN", "s3cret")
    sim = SimulationEngine(num_robots=2, auto_start=False)
    app = create_app(sim)
    client = app.test_client()

    try:
        r1 = client.post("/api/command", json={"command": "step"})
        assert r1.status_code == 401
        r2 = client.post(
            "/api/command",
            json={"command": "step"},
            headers={"Authorization": "Bearer s3cret"},
        )
        assert r2.status_code == 200
    finally:
        sim.shutdown()
