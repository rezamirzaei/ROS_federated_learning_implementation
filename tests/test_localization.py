"""Tests for the distributed TOA localization estimator."""

from __future__ import annotations

import math
import random

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


def _fully_connected(ids):
    return {i: [j for j in ids if j != i] for i in ids}


def test_static_target_converges_without_noise():
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
    for _ in range(120):
        res = est.update(sensors, measurements, neighbors, ground_truth=target)
        last_rmse = res.mean_rmse

    assert last_rmse < 0.1, f"expected RMSE < 0.1, got {last_rmse}"
    # All estimates should agree on roughly the same point.
    assert res.consensus_gap < 0.15


def test_consensus_gap_decreases_on_static_target():
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


def test_moving_target_bounded_rmse():
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


def test_dual_variables_stay_bounded():
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
    assert math.isfinite(max_lam) and max_lam < 50.0

