"""Observability: Prometheus metrics + structured JSON logging."""

from __future__ import annotations

from .logging import configure_logging
from .metrics import (
    REGISTRY,
    fl_aggregation_rounds_total,
    fl_avg_accuracy,
    fl_avg_loss,
    fl_controller_state,
    fl_mean_tracking_error,
    fl_robot_count,
    fl_tick_count,
    update_from_snapshot,
)

__all__ = [
    "REGISTRY",
    "configure_logging",
    "fl_aggregation_rounds_total",
    "fl_avg_accuracy",
    "fl_avg_loss",
    "fl_controller_state",
    "fl_mean_tracking_error",
    "fl_robot_count",
    "fl_tick_count",
    "update_from_snapshot",
]
