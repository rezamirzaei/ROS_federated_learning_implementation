"""Shared result-artifact names for monitoring, dashboards, and tooling."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

AGGREGATION_HISTORY_JSON = "aggregation_history.json"
AGGREGATION_HISTORY_CSV = "aggregation_history.csv"
ROBOT_METRICS_JSON = "robot_metrics.json"
SUMMARY_JSON = "summary.json"
LEGACY_SUMMARY_JSON = "training_summary.json"
DIGITAL_TWIN_IMAGE = "digital_twin.png"

SUMMARY_CANDIDATES = (SUMMARY_JSON, LEGACY_SUMMARY_JSON)
RESULT_BUNDLE_ORDER = (
    AGGREGATION_HISTORY_CSV,
    ROBOT_METRICS_JSON,
    DIGITAL_TWIN_IMAGE,
    AGGREGATION_HISTORY_JSON,
)


def resolve_summary_path(output_dir: str | Path) -> Path | None:
    """Return the preferred summary file for a results directory, if present."""
    base = Path(output_dir)
    for filename in SUMMARY_CANDIDATES:
        path = base / filename
        if path.exists():
            return path
    return None


def iter_bundle_paths(output_dir: str | Path) -> Iterator[Path]:
    """Yield existing result artifacts in a stable order for downloads."""
    base = Path(output_dir)
    summary_path = resolve_summary_path(base)
    if summary_path is not None:
        yield summary_path
    for filename in RESULT_BUNDLE_ORDER:
        path = base / filename
        if path.exists():
            yield path
