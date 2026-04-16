"""
Standalone model helpers — thin re-exports of the core FL models.

Keeps the ``ros_web`` package self-contained while reusing the canonical
model implementations that live under ``src/fl_robots/fl_robots/models/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the fl_robots models importable even outside a ROS2 workspace.
_fl_robots_root = Path(__file__).resolve().parent.parent / "src" / "fl_robots" / "fl_robots"
if str(_fl_robots_root) not in sys.path:
    sys.path.insert(0, str(_fl_robots_root))

from models.simple_nn import (  # noqa: E402
    SimpleNavigationNet,
    ObstacleAvoidanceNet,
    federated_averaging,
    compute_gradient_divergence,
)

__all__ = [
    "SimpleNavigationNet",
    "ObstacleAvoidanceNet",
    "federated_averaging",
    "compute_gradient_divergence",
]

