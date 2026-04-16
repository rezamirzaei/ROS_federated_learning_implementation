"""Tests for the new /api/history/* and /api/mpc/explainer endpoints."""

from __future__ import annotations

import pytest

from fl_robots.simulation import SimulationEngine
from fl_robots.standalone_web import MPC_EXPLAINER, create_app


@pytest.fixture
def client():
    sim = SimulationEngine(num_robots=3, tick_interval=0.5, auto_start=False)
    # Drive the sim forward so history buffers fill. Enable training so
    # we also get global-history samples at the aggregation tick (tick % 5).
    sim.issue_command("start_training")
    for _ in range(12):
        sim.step_once()
    app = create_app(sim)
    app.testing = True
    with app.test_client() as c:
        yield c, sim


def test_history_global_endpoint(client):
    c, _ = client
    r = c.get("/api/history/global")
    assert r.status_code == 200
    body = r.get_json()
    assert "series" in body
    assert isinstance(body["series"], list)
    assert len(body["series"]) >= 1
    pt = body["series"][0]
    for k in (
        "tick",
        "round_id",
        "timestamp",
        "mean_loss",
        "val_loss",
        "mean_accuracy",
        "val_accuracy",
    ):
        assert k in pt, f"missing key {k}"


def test_history_robot_endpoint(client):
    c, sim = client
    robot_id = next(iter(sim.robots))
    r = c.get(f"/api/history/robots/{robot_id}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["robot_id"] == robot_id
    assert len(body["series"]) >= 5  # one per tick for 12 ticks


def test_history_robot_limit_arg(client):
    c, sim = client
    robot_id = next(iter(sim.robots))
    r = c.get(f"/api/history/robots/{robot_id}?limit=3")
    body = r.get_json()
    assert len(body["series"]) == 3


def test_history_mpc_endpoint(client):
    c, _ = client
    r = c.get("/api/history/mpc")
    assert r.status_code == 200
    body = r.get_json()
    assert body["system"] is not None
    assert body["system"]["n_robots"] == 3
    assert len(body["series"]) >= 3


def test_history_localization_endpoint(client):
    c, _ = client
    r = c.get("/api/history/localization")
    assert r.status_code == 200
    body = r.get_json()
    # Enabled iff numpy is installed — test env has ml extra.
    if body["enabled"]:
        assert len(body["series"]) >= 1
        first = body["series"][0]
        assert "target" in first
        assert "estimates" in first
        assert "mean_rmse" in first


def test_mpc_explainer_endpoint(client):
    c, _ = client
    r = c.get("/api/mpc/explainer")
    assert r.status_code == 200
    body = r.get_json()
    for key in ("title", "summary", "decision_variables", "objective", "constraints"):
        assert key in body
    assert body["title"] == MPC_EXPLAINER["title"]


def test_status_snapshot_includes_history_mpc_localization(client):
    c, _ = client
    r = c.get("/api/status")
    body = r.get_json()
    assert "history" in body
    assert "global" in body["history"]
    assert "robots" in body["history"]
    assert "mpc" in body
    assert body["mpc"]["system"] is not None
    assert "localization" in body


def test_openapi_lists_new_paths(client):
    c, _ = client
    r = c.get("/api/openapi.json")
    schema = r.get_json()
    paths = schema["paths"]
    assert "/api/history/global" in paths
    assert "/api/history/robots/{robot_id}" in paths
    assert "/api/history/mpc" in paths
    assert "/api/history/localization" in paths
    assert "/api/mpc/explainer" in paths

