"""Distributed localization algorithms for multi-agent sensor networks.

Exposes the :class:`~fl_robots.localization.tda.DistributedTOAEstimator`
time-of-arrival estimator plus its configuration and result schemas, and
the :class:`~fl_robots.localization.predictor.ConstantVelocityTargetPredictor`
motion-model predictor used to seed each TOA update with a physics-based
prior.
"""

from fl_robots.localization.predictor import (
    ConstantVelocityTargetPredictor,
    PredictorConfig,
)
from fl_robots.localization.tda import (
    DistributedTOAEstimator,
    TOAConfig,
    TOAUpdateResult,
)

__all__ = [
    "ConstantVelocityTargetPredictor",
    "DistributedTOAEstimator",
    "PredictorConfig",
    "TOAConfig",
    "TOAUpdateResult",
]
