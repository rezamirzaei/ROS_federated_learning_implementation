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
    "fl_aggregation_divergence",
    "fl_aggregation_rounds_total",
    "fl_avg_accuracy",
    "fl_avg_loss",
    "fl_best_accuracy",
    "fl_controller_state",
    "fl_fedavg_aggregation_duration_seconds",
    "fl_http_request_duration_seconds",
    "fl_http_requests_total",
    "fl_mean_tracking_error",
    "fl_mpc_solve_time_ms",
    "fl_robot_count",
    "fl_round_latency_seconds",
    "fl_tick_count",
    "fl_tracking_error",
    "fl_training_active",
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

fl_training_active = Gauge(
    "fl_training_active",
    "Whether the simulation currently considers federated training active (1/0).",
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

fl_aggregation_divergence = Gauge(
    "fl_aggregation_divergence",
    "Mean post-aggregation client divergence for the latest completed round.",
    registry=REGISTRY,
)

fl_round_latency_seconds = Histogram(
    "fl_round_latency_seconds",
    "Wall-clock latency of a single federated round.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

fl_fedavg_aggregation_duration_seconds = Histogram(
    "fl_fedavg_aggregation_duration_seconds",
    "Wall-clock latency of the FedAvg aggregation step itself.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY,
)

fl_http_requests_total = Counter(
    "fl_http_requests_total",
    "Total number of standalone HTTP requests served.",
    labelnames=("path", "method", "status"),
    registry=REGISTRY,
)

fl_http_request_duration_seconds = Histogram(
    "fl_http_request_duration_seconds",
    "HTTP request latency for standalone endpoints.",
    labelnames=("path", "method", "status"),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)

# ── Control metrics ───────────────────────────────────────────────────

fl_mean_tracking_error = Gauge(
    "fl_mean_tracking_error",
    "Mean MPC terminal-state tracking error across robots (meters).",
    registry=REGISTRY,
)

fl_tracking_error = Histogram(
    "fl_tracking_error",
    "Per-robot MPC tracking error samples (meters).",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 5.0),
    registry=REGISTRY,
)

fl_mpc_solve_time_ms = Histogram(
    "fl_mpc_solve_time_ms",
    "Per-robot MPC solve time samples (milliseconds).",
    buckets=(0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0),
    registry=REGISTRY,
)


# ── Snapshot bridge ───────────────────────────────────────────────────

_STATE_CODES = {
    "IDLE": 0,
    "RUNNING": 1,
    "PAUSED": 2,
    "MANUAL": 3,
    "RECOVERING": 4,
    "ERROR": 5,
}

_last_round_seen: dict[str, int | float | None] = {"round": 0, "timestamp": None}
_last_mpc_tick_seen: dict[str, int] = {"tick": -1}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def update_from_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Refresh gauges from a :meth:`SimulationEngine.snapshot` payload.

    Idempotent and side-effect-free apart from metric writes.
    """
    system = _mapping(snapshot.get("system"))
    metrics = _mapping(snapshot.get("metrics"))
    history = _mapping(snapshot.get("history"))
    mpc = _mapping(snapshot.get("mpc"))

    fl_robot_count.set(float(system.get("robot_count", 0)))
    fl_tick_count.set(float(system.get("tick_count", 0)))
    fl_training_active.set(1.0 if bool(system.get("training_active", False)) else 0.0)

    state = str(system.get("controller_state", "IDLE"))
    for known in _STATE_CODES:
        fl_controller_state.labels(state=known).set(1.0 if known == state else 0.0)

    fl_avg_loss.set(float(metrics.get("avg_loss", 0.0)))
    fl_avg_accuracy.set(float(metrics.get("avg_accuracy", 0.0)))
    fl_best_accuracy.set(float(metrics.get("best_accuracy", 0.0)))
    fl_mean_tracking_error.set(float(metrics.get("mean_tracking_error", 0.0)))
    last_aggregation = _mapping(metrics.get("last_aggregation"))
    fl_aggregation_divergence.set(float(last_aggregation.get("mean_divergence", 0.0)))

    current_round = int(system.get("current_round", 0))
    previous_round = int(_last_round_seen["round"] or 0)
    if current_round < previous_round:
        # Reset event — counter can't go down, so we just rebase the baseline.
        _last_round_seen["round"] = current_round
        _last_round_seen["timestamp"] = None
        previous_round = current_round

    last_timestamp_seen = _last_round_seen["timestamp"]
    previous_timestamp = (
        float(last_timestamp_seen) if isinstance(last_timestamp_seen, (int, float)) else None
    )
    global_series = history.get("global")
    if isinstance(global_series, list):
        for point_raw in global_series:
            point = _mapping(point_raw)
            point_round = int(point.get("round_id", point.get("round", 0)))
            if point_round <= previous_round:
                continue
            point_timestamp = float(point.get("timestamp", 0.0))
            fl_aggregation_rounds_total.inc(point_round - previous_round)
            if previous_timestamp is not None and point_timestamp > previous_timestamp:
                fl_round_latency_seconds.observe(point_timestamp - previous_timestamp)
            previous_round = point_round
            if point_timestamp > 0.0:
                previous_timestamp = point_timestamp

    if current_round > previous_round:
        fl_aggregation_rounds_total.inc(current_round - previous_round)
        previous_round = current_round

    _last_round_seen["round"] = previous_round
    _last_round_seen["timestamp"] = previous_timestamp

    mpc_system = _mapping(mpc.get("system"))
    mpc_tick = int(mpc_system.get("tick", system.get("tick_count", -1)))
    previous_mpc_tick = _last_mpc_tick_seen["tick"]
    if mpc_tick < previous_mpc_tick:
        _last_mpc_tick_seen["tick"] = mpc_tick
        previous_mpc_tick = mpc_tick
    if mpc_tick > previous_mpc_tick:
        per_robot = mpc.get("per_robot")
        if isinstance(per_robot, list):
            for diag_raw in per_robot:
                diag = _mapping(diag_raw)
                fl_tracking_error.observe(float(diag.get("tracking_error", 0.0)))
                fl_mpc_solve_time_ms.observe(float(diag.get("qp_solve_time_ms", 0.0)))
        _last_mpc_tick_seen["tick"] = mpc_tick
