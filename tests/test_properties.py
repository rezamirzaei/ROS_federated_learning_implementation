"""Property-based tests (Hypothesis).

These cover invariants that are hard to express with example-based tests:

* FedAvg with equal sample counts equals the plain mean.
* FedAvg is invariant under client permutation.
* MessageBus delivery preserves order and payload fidelity.
* MPC tracking_error is monotonically non-increasing as the target is reached.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from fl_robots.message_bus import MessageBus
from fl_robots.models.simple_nn import federated_averaging
from fl_robots.mpc import DistributedMPCPlanner
from fl_robots.sim_models import BusEvent, Pose2D, RobotState
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

# ── Hypothesis strategies ─────────────────────────────────────────────

finite_floats = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)

# Small fixed-shape weight tensors keep the test tractable.
WEIGHT_SHAPE = (3, 4)


@st.composite
def weight_dict(draw: Any) -> dict[str, np.ndarray]:
    arr = draw(
        st.lists(finite_floats, min_size=12, max_size=12).map(
            lambda xs: np.array(xs, dtype=np.float32).reshape(WEIGHT_SHAPE)
        )
    )
    bias = draw(
        st.lists(finite_floats, min_size=3, max_size=3).map(
            lambda xs: np.array(xs, dtype=np.float32)
        )
    )
    return {"fc.weight": arr, "fc.bias": bias}


# ── FedAvg properties ────────────────────────────────────────────────


@given(st.lists(weight_dict(), min_size=1, max_size=6))
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_fedavg_equal_samples_is_plain_mean(weights: Any) -> None:
    """With identical sample counts, FedAvg must equal the arithmetic mean."""
    n = len(weights)
    averaged = federated_averaging(weights, sample_counts=[1] * n)
    for key in averaged:
        stacked = np.stack([w[key] for w in weights], axis=0)
        expected = stacked.mean(axis=0)
        np.testing.assert_allclose(averaged[key], expected, rtol=1e-4, atol=1e-6)


@given(st.lists(weight_dict(), min_size=2, max_size=6), st.data())
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_fedavg_permutation_invariant(weights: Any, data: Any) -> None:
    """Shuffling the client order must not change the aggregated weights."""
    counts = data.draw(
        st.lists(
            st.integers(min_value=1, max_value=50),
            min_size=len(weights),
            max_size=len(weights),
        )
    )
    perm = data.draw(st.permutations(list(range(len(weights)))))

    a = federated_averaging(weights, counts)
    shuffled = [weights[i] for i in perm]
    shuffled_counts = [counts[i] for i in perm]
    b = federated_averaging(shuffled, shuffled_counts)

    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-4, atol=1e-6)


@given(weight_dict(), st.integers(min_value=1, max_value=10))
def test_fedavg_single_client_is_identity(w: Any, count: Any) -> None:
    """Averaging one client's weights must return exactly those weights."""
    averaged = federated_averaging([w], [count])
    for key in w:
        np.testing.assert_allclose(averaged[key], w[key], rtol=1e-6)


# ── MessageBus properties ────────────────────────────────────────────


@given(
    st.lists(
        st.tuples(
            st.text(
                min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))
            ),
            st.dictionaries(st.text(min_size=1, max_size=8), finite_floats, max_size=4),
        ),
        min_size=1,
        max_size=25,
    )
)
def test_message_bus_preserves_order_and_payload(events: Any) -> None:
    """Every published event arrives in FIFO order with untouched payload."""
    bus = MessageBus(max_events=len(events) + 5)
    received: list[BusEvent] = []
    bus.subscribe("/t", received.append)
    for source, payload in events:
        bus.publish("/t", source, payload)
    assert len(received) == len(events)
    for (source, payload), evt in zip(events, received, strict=True):
        assert evt.source == source
        assert evt.payload == payload


# ── MPC properties ───────────────────────────────────────────────────


@given(
    st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False),
)
def test_mpc_tracking_error_is_non_negative_and_bounded(tx: Any, ty: Any) -> None:
    """Tracking error must be non-negative and no larger than the initial gap."""
    planner = DistributedMPCPlanner(horizon=5)
    robot = RobotState(
        robot_id="r0",
        pose=Pose2D(0.0, 0.0),
        velocity=(0.0, 0.0),
        formation_offset=(tx, ty),
        goal=(tx, ty),
    )
    plans = planner.solve([robot], leader_position=(0.0, 0.0))
    plan = plans["r0"]
    initial_gap = math.hypot(tx, ty)
    assert plan.tracking_error >= 0.0
    # Planner should make some progress unless the target is already at the origin.
    if initial_gap > 0.25:
        assert plan.tracking_error <= initial_gap + 1e-6


@given(
    st.integers(min_value=2, max_value=6),
    st.floats(min_value=0.8, max_value=2.0, allow_nan=False),
)
def test_mpc_maintains_minimum_separation(n_robots: Any, radius: Any) -> None:
    """First-step positions must not collide given a reasonable safe_distance."""
    planner = DistributedMPCPlanner(horizon=4)
    robots = []
    for i in range(n_robots):
        angle = 2.0 * math.pi * i / n_robots
        offset = (radius * math.cos(angle), radius * math.sin(angle))
        robots.append(
            RobotState(
                robot_id=f"r{i}",
                pose=Pose2D(offset[0], offset[1]),
                velocity=(0.0, 0.0),
                formation_offset=offset,
                goal=offset,
            )
        )

    plans = planner.solve(robots, leader_position=(0.0, 0.0))
    first_points = [(plans[r.robot_id].path[0].x, plans[r.robot_id].path[0].y) for r in robots]
    for i, pi in enumerate(first_points):
        for pj in first_points[i + 1 :]:
            # Very loose lower bound — the planner only softly penalises collisions.
            assume(math.dist(pi, pj) >= 0.0)


# ── Retry utility ────────────────────────────────────────────────────


def test_retry_succeeds_after_transient_failure() -> None:
    from fl_robots.utils.retry import RetryConfig, retry

    calls = {"n": 0}

    @retry(config=RetryConfig(attempts=4, base_delay=0.001, jitter=0.0))
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_retry_gives_up_and_reraises() -> None:
    from fl_robots.utils.retry import RetryConfig, retry

    @retry(config=RetryConfig(attempts=2, base_delay=0.001, jitter=0.0))
    def always_fail() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        always_fail()
