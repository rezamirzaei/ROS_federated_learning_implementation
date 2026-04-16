"""Prometheus metrics for the standalone simulation and FL pipeline.

We keep a *dedicated* :class:`CollectorRegistry` rather than using the default
global one, so tests and multi-process runs don't collide. The `/metrics`
endpoint in ``standalone_web`` calls :func:`update_from_snapshot` to refresh
gauges right before serializing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

__all__ = [
    "REGISTRY",
    "fl_aggregation_rounds_total",
    "fl_avg_accuracy",
    "fl_avg_loss",
    "fl_best_accuracy",
    "fl_controller_state",
    "fl_mean_tracking_error",
    "fl_robot_count",
    "fl_round_latency_seconds",
    "fl_tick_count",
    "update_from_snapshot",
]


REGISTRY = CollectorRegistry(auto_describe=True)

# ── System state ──────────────────────────────────────────────────────

fl_robot_count = Gauge(
    "fl_robot_count",
    "Number of active robots currently tracked by the simulation engine.",
    registry=REGISTRY,
)

fl_tick_count = Gauge(
    "fl_tick_count",
    "Total simulation ticks since last reset.",
    registry=REGISTRY,
)

fl_controller_state = Gauge(
    "fl_controller_state",
    "Current coordinator state encoded as numeric enum.",
    labelnames=("state",),
    registry=REGISTRY,
)

# ── FL metrics ────────────────────────────────────────────────────────

fl_avg_loss = Gauge(
    "fl_avg_loss",
    "Mean per-robot training loss after the latest round.",
    registry=REGISTRY,
)

fl_avg_accuracy = Gauge(
    "fl_avg_accuracy",
    "Mean per-robot training accuracy (%) after the latest round.",
    registry=REGISTRY,
)

fl_best_accuracy = Gauge(
    "fl_best_accuracy",
    "Best per-robot training accuracy (%) across all active robots.",
    registry=REGISTRY,
)

fl_aggregation_rounds_total = Counter(
    "fl_aggregation_rounds_total",
    "Total number of federated-averaging rounds completed.",
    registry=REGISTRY,
)

fl_round_latency_seconds = Histogram(
    "fl_round_latency_seconds",
    "Wall-clock latency of a single federated round.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

# ── Control metrics ───────────────────────────────────────────────────

fl_mean_tracking_error = Gauge(
    "fl_mean_tracking_error",
    "Mean MPC terminal-state tracking error across robots (meters).",
    registry=REGISTRY,
)


# ── Snapshot bridge ───────────────────────────────────────────────────

_STATE_CODES = {
    "IDLE": 0,
    "RUNNING": 1,
    "PAUSED": 2,
    "MANUAL": 3,
    "RECOVERING": 4,
}

#: Tracks the last observed round so we only advance the counter on change.
_last_round_seen: dict[str, int] = {"round": 0}


def update_from_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Refresh gauges from a :meth:`SimulationEngine.snapshot` payload.

    Idempotent and side-effect-free apart from metric writes.
    """
    system = snapshot.get("system") or {}
    metrics = snapshot.get("metrics") or {}

    fl_robot_count.set(float(system.get("robot_count", 0)))
    fl_tick_count.set(float(system.get("tick_count", 0)))

    state = str(system.get("controller_state", "IDLE"))
    for known in _STATE_CODES:
        fl_controller_state.labels(state=known).set(1.0 if known == state else 0.0)

    fl_avg_loss.set(float(metrics.get("avg_loss", 0.0)))
    fl_avg_accuracy.set(float(metrics.get("avg_accuracy", 0.0)))
    fl_best_accuracy.set(float(metrics.get("best_accuracy", 0.0)))
    fl_mean_tracking_error.set(float(metrics.get("mean_tracking_error", 0.0)))

    current_round = int(system.get("current_round", 0))
    if current_round > _last_round_seen["round"]:
        fl_aggregation_rounds_total.inc(current_round - _last_round_seen["round"])
        _last_round_seen["round"] = current_round
    elif current_round < _last_round_seen["round"]:
        # Reset event — counter can't go down, so we just rebase the baseline.
        _last_round_seen["round"] = current_round
