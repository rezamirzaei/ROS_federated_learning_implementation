"""Tests for the newly-added modules: controller, persistence, metrics."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fl_robots.controller import (
    COMMAND_NAMES,
    CommandRequest,
    is_valid_command,
    validate_command,
)
from fl_robots.observability.metrics import (
    REGISTRY,
    fl_aggregation_divergence,
    fl_avg_accuracy,
    fl_controller_state,
    fl_mpc_solve_time_ms,
    fl_robot_count,
    fl_round_latency_seconds,
    fl_tracking_error,
    fl_training_active,
    update_from_snapshot,
)
from fl_robots.persistence import MetricsStore
from fl_robots.results_artifacts import LEGACY_SUMMARY_JSON, SUMMARY_JSON, resolve_summary_path

# ── Controller ───────────────────────────────────────────────────────


def _sample_value(metric: Any, sample_name: str) -> float:
    for family in metric.collect():
        for sample in family.samples:
            if sample.name == sample_name and not sample.labels:
                return float(sample.value)
    raise AssertionError(f"missing sample {sample_name}")


def test_all_valid_commands_pass_validator() -> None:
    for name in COMMAND_NAMES:
        assert is_valid_command(name)
        assert validate_command(name) == name


def test_invalid_command_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown command"):
        validate_command("self_destruct")


def test_command_request_rejects_extra_fields() -> None:
    with pytest.raises(Exception):
        CommandRequest.model_validate({"command": "step", "extra": 1})


def test_command_request_rejects_unknown_command() -> None:
    with pytest.raises(Exception):
        CommandRequest.model_validate({"command": "nope"})


def test_command_request_accepts_known_command() -> None:
    cmd = CommandRequest.model_validate({"command": "step"})
    assert cmd.command == "step"


# ── Persistence ──────────────────────────────────────────────────────


def test_metrics_store_roundtrips_rounds(tmp_path: Path) -> None:
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


def test_metrics_store_records_robot_metrics(tmp_path: Path) -> None:
    with MetricsStore(tmp_path / "run.sqlite") as db:
        db.record_robot_metric("robot_1", 1, loss=0.8, accuracy=70.0, tracking_error=0.12)
        db.record_robot_metric("robot_1", 2, loss=0.5, accuracy=85.0, tracking_error=0.06)
        rows = db.fetch_robot_metrics("robot_1", limit=5)
        assert len(rows) == 2
        assert rows[0]["round_id"] == 2


def test_metrics_store_records_events(tmp_path: Path) -> None:
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


def test_results_artifacts_prefers_summary_json_and_falls_back_to_legacy(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    legacy_path = results_dir / LEGACY_SUMMARY_JSON
    legacy_path.write_text('{"source":"legacy"}', encoding="utf-8")
    assert resolve_summary_path(results_dir) == legacy_path

    preferred_path = results_dir / SUMMARY_JSON
    preferred_path.write_text('{"source":"preferred"}', encoding="utf-8")
    assert resolve_summary_path(results_dir) == preferred_path


# ── Metrics ──────────────────────────────────────────────────────────


def test_update_from_snapshot_populates_gauges() -> None:
    snapshot = {
        "system": {
            "controller_state": "RUNNING",
            "training_active": True,
            "robot_count": 4,
            "tick_count": 9000,
            "current_round": 1000,
        },
        "metrics": {
            "avg_loss": 0.321,
            "avg_accuracy": 77.5,
            "best_accuracy": 90.0,
            "mean_tracking_error": 0.12,
            "last_aggregation": {"mean_divergence": 0.42},
        },
        "history": {
            "global": [
                {"round_id": 999, "timestamp": 1000.0},
                {"round_id": 1000, "timestamp": 1001.5},
            ]
        },
        "mpc": {
            "system": {"tick": 9000},
            "per_robot": [
                {"robot_id": "robot_1", "tracking_error": 0.2, "qp_solve_time_ms": 4.0},
                {"robot_id": "robot_2", "tracking_error": 0.4, "qp_solve_time_ms": 6.5},
            ],
        },
    }
    before_tracking = _sample_value(fl_tracking_error, "fl_tracking_error_count")
    before_solve = _sample_value(fl_mpc_solve_time_ms, "fl_mpc_solve_time_ms_count")
    before_latency = _sample_value(fl_round_latency_seconds, "fl_round_latency_seconds_count")
    update_from_snapshot(snapshot)
    update_from_snapshot(snapshot)
    assert fl_robot_count._value.get() == 4.0  # type: ignore[attr-defined]
    assert abs(fl_avg_accuracy._value.get() - 77.5) < 1e-6  # type: ignore[attr-defined]
    assert fl_training_active._value.get() == 1.0  # type: ignore[attr-defined]
    assert fl_aggregation_divergence._value.get() == 0.42  # type: ignore[attr-defined]
    assert fl_controller_state.labels(state="RUNNING")._value.get() == 1.0  # type: ignore[attr-defined]
    assert fl_controller_state.labels(state="ERROR")._value.get() == 0.0  # type: ignore[attr-defined]
    assert _sample_value(fl_tracking_error, "fl_tracking_error_count") - before_tracking == 2.0
    assert _sample_value(fl_mpc_solve_time_ms, "fl_mpc_solve_time_ms_count") - before_solve == 2.0
    assert (
        _sample_value(fl_round_latency_seconds, "fl_round_latency_seconds_count") - before_latency
        == 1.0
    )

    # Scrape produces well-formed Prometheus text format.
    from prometheus_client import generate_latest

    text = generate_latest(REGISTRY).decode("utf-8")
    assert "fl_robot_count" in text
    assert "fl_avg_accuracy" in text
    assert "fl_training_active" in text
    assert "fl_aggregation_divergence" in text
    assert "fl_tracking_error_bucket" in text
    assert "fl_mpc_solve_time_ms_bucket" in text
