"""Models package for federated learning."""

from .simple_nn import (
    ObstacleAvoidanceNet,
    SimpleNavigationNet,
    compute_gradient_divergence,
    compute_weight_l2_drift,
    federated_averaging,
)

__all__ = [
    "ObstacleAvoidanceNet",
    "SimpleNavigationNet",
    "compute_gradient_divergence",
    "compute_weight_l2_drift",
    "federated_averaging",
]
