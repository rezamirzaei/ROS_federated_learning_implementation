"""Models package for federated learning."""

from .simple_nn import (
    SimpleNavigationNet,
    ObstacleAvoidanceNet,
    federated_averaging,
    compute_gradient_divergence
)

__all__ = [
    'SimpleNavigationNet',
    'ObstacleAvoidanceNet',
    'federated_averaging',
    'compute_gradient_divergence'
]
