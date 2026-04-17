"""Tests for the distributed TOA localization estimator."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from fl_robots.localization import DistributedTOAEstimator, TOAConfig


def _fixed_sensors() -> dict[str, tuple[float, float]]:
    # 4 sensors at the corners of a 4x4 square — well-conditioned geometry.
    return {
        "robot_1": (-2.0, -2.0),
        "robot_2": (2.0, -2.0),
        "robot_3": (2.0, 2.0),
        "robot_4": (-2.0, 2.0),
    }


def _fully_connected(ids: Any) -> dict[Any, list[Any]]:
    return {i: [j for j in ids if j != i] for i in ids}


def test_static_target_converges_without_noise() -> None:
    sensors = _fixed_sensors()
    est = DistributedTOAEstimator(
        robot_ids=list(sensors),
        config=TOAConfig(step_size=0.12, rho=0.6, max_inner_iters=6),
        seed=7,
    )
    target = (0.3, -0.4)
    neighbors = _fully_connected(list(sensors))
    measurements = {
        rid: math.hypot(target[0] - p[0], target[1] - p[1]) for rid, p in sensors.items()
    }

    last_rmse = float("inf")
    res = None
    for _ in range(120):
        res = est.update(sensors, measurements, neighbors, ground_truth=target)
        last_rmse = res.mean_rmse

    assert last_rmse < 0.1, f"expected RMSE < 0.1, got {last_rmse}"
    # All estimates should agree on roughly the same point.
    assert res is not None
    assert res.consensus_gap < 0.15


def test_consensus_gap_decreases_on_static_target() -> None:
    sensors = _fixed_sensors()
    est = DistributedTOAEstimator(
        robot_ids=list(sensors),
        config=TOAConfig(step_size=0.1, rho=0.5, max_inner_iters=4),
        seed=11,
    )
    target = (0.0, 0.0)
    nbrs = _fully_connected(list(sensors))
    meas = {rid: math.hypot(target[0] - p[0], target[1] - p[1]) for rid, p in sensors.items()}

    gaps = []
    for _ in range(80):
        r = est.update(sensors, meas, nbrs, ground_truth=target)
        gaps.append(r.consensus_gap)

    # Not strictly monotonic due to dual dynamics, but the trailing window
    # should be dramatically tighter than the first window.
    early = sum(gaps[:10]) / 10
    late = sum(gaps[-10:]) / 10
    assert late < 0.3 * early, f"gap failed to shrink: early={early:.3f}, late={late:.3f}"


def test_moving_target_bounded_rmse() -> None:
    sensors = _fixed_sensors()
    est = DistributedTOAEstimator(
        robot_ids=list(sensors),
        config=TOAConfig(step_size=0.18, rho=0.6, max_inner_iters=5),
        seed=3,
    )
    rng = random.Random(0)
    nbrs = _fully_connected(list(sensors))

    # Warm-up period: let the estimator converge on a static pose first.
    warmup_target = (0.0, 0.0)
    warm_meas = {
        rid: math.hypot(warmup_target[0] - p[0], warmup_target[1] - p[1])
        for rid, p in sensors.items()
    }
    for _ in range(60):
        est.update(sensors, warm_meas, nbrs, ground_truth=warmup_target)

    # Moving target with additive measurement noise.
    sigma = 0.05
    rmses = []
    for k in range(80):
        t = (1.2 * math.cos(0.15 * k), 0.8 * math.sin(0.3 * k))
        meas = {
            rid: math.hypot(t[0] - p[0], t[1] - p[1]) + rng.gauss(0.0, sigma)
            for rid, p in sensors.items()
        }
        r = est.update(sensors, meas, nbrs, ground_truth=t)
        rmses.append(r.mean_rmse)

    mean_rmse = sum(rmses[-20:]) / 20
    assert mean_rmse < 0.6, f"tracking RMSE too large: {mean_rmse}"


def test_dual_variables_stay_bounded() -> None:
    sensors = _fixed_sensors()
    est = DistributedTOAEstimator(
        robot_ids=list(sensors),
        config=TOAConfig(),
        seed=42,
    )
    target = (0.5, 0.5)
    nbrs = _fully_connected(list(sensors))
    meas = {rid: math.hypot(target[0] - p[0], target[1] - p[1]) for rid, p in sensors.items()}

    for _ in range(150):
        est.update(sensors, meas, nbrs, ground_truth=target)

    # Every dual variable norm should stay bounded (no divergence).
    max_lam = max(float(np.linalg.norm(v)) for v in est._duals.values())
    assert math.isfinite(max_lam) and math.isfinite(max_lam) and max_lam < 50.0


def test_predicted_target_prior_accelerates_convergence() -> None:
    """With a good motion-model prior, the estimator locks on faster."""
    sensors = _fixed_sensors()
    target = (0.5, -0.25)
    nbrs = _fully_connected(list(sensors))
    meas = {rid: math.hypot(target[0] - p[0], target[1] - p[1]) for rid, p in sensors.items()}

    est_plain = DistributedTOAEstimator(
        robot_ids=list(sensors),
        config=TOAConfig(step_size=0.12, rho=0.4, max_inner_iters=2, prior_weight=0.0),
        seed=0,
    )
    est_prior = DistributedTOAEstimator(
        robot_ids=list(sensors),
        config=TOAConfig(step_size=0.12, rho=0.4, max_inner_iters=2, prior_weight=0.6),
        seed=0,
    )

    r_plain = None
    r_prior = None
    for _ in range(8):  # only a few iterations so the prior matters
        r_plain = est_plain.update(sensors, meas, nbrs, ground_truth=target)
        r_prior = est_prior.update(
            sensors, meas, nbrs, ground_truth=target, predicted_target=target
        )
    assert r_plain is not None and r_prior is not None
    assert r_prior.mean_rmse < r_plain.mean_rmse, (
        f"prior failed to help: plain={r_plain.mean_rmse:.3f}, prior={r_prior.mean_rmse:.3f}"
    )


def test_predictor_tracks_constant_velocity_target() -> None:
    """The α-β predictor should follow a constant-velocity target closely."""
    from fl_robots.localization import ConstantVelocityTargetPredictor, PredictorConfig

    dt = 0.25
    pred = ConstantVelocityTargetPredictor(x=0.0, y=0.0, config=PredictorConfig(dt=dt))
    # True target moves at (0.3, -0.1) m/s.
    vx_true, vy_true = 0.3, -0.1
    x, y = 0.0, 0.0
    errs = []
    for _ in range(40):
        x += vx_true * dt
        y += vy_true * dt
        px, py = pred.predict()
        pred.update((x, y))
        errs.append(math.hypot(px - x, py - y))
    # After warm-up the tracking error should stabilise well below 0.05 m.
    assert sum(errs[-10:]) / 10 < 0.05
