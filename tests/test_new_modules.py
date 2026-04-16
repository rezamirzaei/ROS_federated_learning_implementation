"""Tests for the newly-added modules: controller, persistence, metrics."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fl_robots.controller import (
    COMMAND_NAMES,
    CommandRequest,
    is_valid_command,
    validate_command,
)
from fl_robots.observability.metrics import (
    REGISTRY,
    fl_avg_accuracy,
    fl_robot_count,
    update_from_snapshot,
)
from fl_robots.persistence import MetricsStore

# ── Controller ───────────────────────────────────────────────────────


def test_all_valid_commands_pass_validator():
    for name in COMMAND_NAMES:
        assert is_valid_command(name)
        assert validate_command(name) == name


def test_invalid_command_rejected():
    with pytest.raises(ValueError, match="Unknown command"):
        validate_command("self_destruct")


def test_command_request_rejects_extra_fields():
    with pytest.raises(Exception):
        CommandRequest.model_validate({"command": "step", "extra": 1})


def test_command_request_rejects_unknown_command():
    with pytest.raises(Exception):
        CommandRequest.model_validate({"command": "nope"})


def test_command_request_accepts_known_command():
    cmd = CommandRequest.model_validate({"command": "step"})
    assert cmd.command == "step"


# ── Persistence ──────────────────────────────────────────────────────


def test_metrics_store_roundtrips_rounds(tmp_path: Path):
    db = MetricsStore(tmp_path / "run.sqlite")
    db.record_round(
        {
            "round": 1,
            "participants": 3,
            "mean_loss": 0.9,
            "mean_accuracy": 75.5,
            "mean_divergence": 0.4,
            "formation_error": 0.1,
        }
    )
    db.record_round(
        {
            "round": 2,
            "participants": 3,
            "mean_loss": 0.5,
            "mean_accuracy": 85.0,
            "mean_divergence": 0.2,
            "formation_error": 0.05,
        }
    )

    rows = db.fetch_rounds(limit=10)
    assert len(rows) == 2
    assert rows[0]["round_id"] == 2
    assert rows[0]["mean_accuracy"] == 85.0
    db.close()


def test_metrics_store_records_robot_metrics(tmp_path: Path):
    with MetricsStore(tmp_path / "run.sqlite") as db:
        db.record_robot_metric("robot_1", 1, loss=0.8, accuracy=70.0, tracking_error=0.12)
        db.record_robot_metric("robot_1", 2, loss=0.5, accuracy=85.0, tracking_error=0.06)
        rows = db.fetch_robot_metrics("robot_1", limit=5)
        assert len(rows) == 2
        assert rows[0]["round_id"] == 2


def test_metrics_store_records_events(tmp_path: Path):
    path = tmp_path / "run.sqlite"
    with MetricsStore(path) as db:
        db.record_event("/fl/robot_status", "robot_1", {"status": "registered"})

    # Re-open raw to verify row is persistent.
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute("SELECT topic, source, payload_json FROM events;")
        row = cur.fetchone()
        assert row[0] == "/fl/robot_status"
        assert row[1] == "robot_1"
        assert json.loads(row[2]) == {"status": "registered"}
    finally:
        conn.close()


# ── Metrics ──────────────────────────────────────────────────────────


def test_update_from_snapshot_populates_gauges():
    snapshot = {
        "system": {
            "controller_state": "RUNNING",
            "robot_count": 4,
            "tick_count": 42,
            "current_round": 3,
        },
        "metrics": {
            "avg_loss": 0.321,
            "avg_accuracy": 77.5,
            "best_accuracy": 90.0,
            "mean_tracking_error": 0.12,
        },
    }
    update_from_snapshot(snapshot)
    assert fl_robot_count._value.get() == 4.0  # type: ignore[attr-defined]
    assert abs(fl_avg_accuracy._value.get() - 77.5) < 1e-6  # type: ignore[attr-defined]

    # Scrape produces well-formed Prometheus text format.
    from prometheus_client import generate_latest

    text = generate_latest(REGISTRY).decode("utf-8")
    assert "fl_robot_count" in text
    assert "fl_avg_accuracy" in text
