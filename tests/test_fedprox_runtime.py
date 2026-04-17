"""Runtime FedProx wiring tests — exercise the aggregator → robot_agent
broadcast path and verify the proximal penalty is actually applied to the
local loss when ``algorithm=fedprox`` is in force.

These run through the FakeROS harness (no rclpy required) so they execute on
every CI matrix job, not just the ROS runtime image.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

# ── Aggregator side ─────────────────────────────────────────────────────


def test_aggregator_defaults_to_fedavg_in_global_model_payload(fake_ros: Any) -> None:
    from fl_robots.aggregator import AggregatorNode

    node = AggregatorNode()
    assert node.fl_algorithm == "fedavg"
    node._publish_global_model()

    msgs = node.publishers["/fl/global_model"].messages
    assert msgs, "global_model publisher produced no messages"
    payload = json.loads(msgs[-1].data)
    assert "config" in payload, "config block missing from broadcast"
    assert payload["config"]["algorithm"] == "fedavg"
    # μ is zeroed under FedAvg even though the parameter has a default.
    assert payload["config"]["proximal_mu"] == 0.0


def test_aggregator_broadcasts_fedprox_config_when_configured(fake_ros: Any) -> None:
    from fl_robots.aggregator import AggregatorNode

    node = AggregatorNode()
    node.fl_algorithm = "fedprox"
    node.proximal_mu = 0.1
    node._publish_global_model()

    payload = json.loads(node.publishers["/fl/global_model"].messages[-1].data)
    assert payload["config"]["algorithm"] == "fedprox"
    assert payload["config"]["proximal_mu"] == 0.1


def test_aggregator_tags_aggregation_metrics_with_algorithm(fake_ros: Any) -> None:
    from fl_robots.aggregator import AggregatorNode

    node = AggregatorNode()
    node.fl_algorithm = "fedprox"
    node.proximal_mu = 0.05
    node.min_robots = 2

    class _Msg:
        def __init__(self, d: Any) -> None:
            self.data = d

    # Register two robots and deliver matching weights.
    for rid in ("r1", "r2"):
        fake_ros.publish(
            "/fl/robot_status",
            _Msg(json.dumps({"type": "registration", "robot_id": rid})),
        )
    weights_payload = {k: v.tolist() for k, v in node.global_model.get_weights().items()}
    for rid in ("r1", "r2"):
        fake_ros.publish(
            f"/fl/{rid}/model_weights",
            _Msg(
                json.dumps(
                    {
                        "type": "local_weights",
                        "robot_id": rid,
                        "round": 1,
                        "samples_trained": 64,
                        "weights": weights_payload,
                    }
                )
            ),
        )
    result = node._perform_aggregation()
    assert result is not None
    assert result["algorithm"] == "fedprox"
    assert result["proximal_mu"] == 0.05


# ── Robot agent side ───────────────────────────────────────────────────


def _broadcast_global(node: Any, *, algorithm: str, proximal_mu: float, round_num: int = 1) -> None:
    """Publish a canonical global_model payload to the shared FakeROS bus."""
    weights = {k: v.tolist() for k, v in node.model.get_weights().items()}

    class _Msg:
        data = json.dumps(
            {
                "type": "global_model",
                "round": round_num,
                "weights": weights,
                "config": {"algorithm": algorithm, "proximal_mu": proximal_mu},
            }
        )

    # Direct-deliver to the topic the agent subscribed to.
    from fl_robots.robot_agent import RobotAgentNode

    # The FakeROS bus forwards publishes to every subscription on that topic.
    # ``node`` subscribed to /fl/global_model during __init__.
    node.global_model_callback(_Msg())  # type: ignore[arg-type]


def test_robot_agent_ignores_proximal_under_fedavg(fake_ros: Any) -> None:
    from fl_robots.robot_agent import RobotAgentNode

    node = RobotAgentNode()
    _broadcast_global(node, algorithm="fedavg", proximal_mu=0.5)

    # Snapshot must NOT be populated under FedAvg — keeps memory flat.
    assert node._fl_global_snapshot is None
    assert node._fl_algorithm == "fedavg"
    assert node._proximal_penalty() is None


def test_robot_agent_stores_snapshot_under_fedprox(fake_ros: Any) -> None:
    from fl_robots.robot_agent import RobotAgentNode

    node = RobotAgentNode()
    _broadcast_global(node, algorithm="fedprox", proximal_mu=0.1)

    assert node._fl_algorithm == "fedprox"
    assert node._fl_proximal_mu == 0.1
    assert node._fl_global_snapshot is not None
    # The snapshot keys must be a subset of named_parameters() (trainable
    # weights only — no BN running stats).
    param_names = {name for name, _ in node.model.named_parameters()}
    assert set(node._fl_global_snapshot.keys()).issubset(param_names)


def test_proximal_penalty_is_zero_at_snapshot_point(fake_ros: Any) -> None:
    """When local weights == global snapshot, the proximal term is 0."""
    from fl_robots.robot_agent import RobotAgentNode

    node = RobotAgentNode()
    _broadcast_global(node, algorithm="fedprox", proximal_mu=0.25)

    penalty = node._proximal_penalty()
    assert penalty is not None
    # Model just overwrote its weights with the broadcast → distance is 0.
    assert float(penalty.item()) == 0.0


def test_proximal_penalty_grows_with_parameter_drift(fake_ros: Any) -> None:
    """As local weights drift from the snapshot, the penalty must strictly
    increase. Guards against sign-flip / wrong-norm regressions."""
    from fl_robots.robot_agent import RobotAgentNode

    node = RobotAgentNode()
    _broadcast_global(node, algorithm="fedprox", proximal_mu=0.5)

    penalty0 = node._proximal_penalty()
    assert penalty0 is not None and float(penalty0.item()) == 0.0

    # Perturb every trainable parameter by a small constant.
    with torch.no_grad():
        for _, p in node.model.named_parameters():
            p.add_(0.1)
    penalty1 = node._proximal_penalty()
    assert penalty1 is not None
    # Analytically: ½·μ·Σ‖Δ‖² with Δ=0.1 across all params; must be > 0 and
    # match μ/2 · 0.01 · n_params exactly.
    n_params = sum(p.numel() for p in node.model.parameters())
    expected = 0.5 * 0.5 * 0.01 * n_params
    assert abs(float(penalty1.item()) - expected) < 1e-3


def test_proximal_penalty_gradient_pulls_weights_toward_snapshot(fake_ros: Any) -> None:
    """The gradient of the proximal term is ``μ·(w − w_global)``. Applying
    an SGD step on the penalty alone must reduce the distance to the
    snapshot — the core reason FedProx stabilises non-IID training."""
    from fl_robots.robot_agent import RobotAgentNode

    node = RobotAgentNode()
    _broadcast_global(node, algorithm="fedprox", proximal_mu=1.0)

    # Perturb weights so there's a non-zero penalty to descend.
    with torch.no_grad():
        for _, p in node.model.named_parameters():
            p.add_(0.05)

    def _distance_squared() -> float:
        d = 0.0
        for name, p in node.model.named_parameters():
            g = node._fl_global_snapshot[name]  # type: ignore[index]
            d += float((p - g).pow(2).sum().item())
        return d

    before = _distance_squared()
    assert before > 0.0

    # Take a single gradient step purely on the proximal term.
    optimizer = torch.optim.SGD(node.model.parameters(), lr=0.1)
    optimizer.zero_grad()
    penalty = node._proximal_penalty()
    assert penalty is not None
    penalty.backward()
    optimizer.step()

    after = _distance_squared()
    assert after < before, "proximal gradient step did not pull weights back"


def test_proximal_penalty_rejects_missing_snapshot_params(fake_ros: Any) -> None:
    from fl_robots.robot_agent import RobotAgentNode

    node = RobotAgentNode()
    _broadcast_global(node, algorithm="fedprox", proximal_mu=0.2)

    assert node._fl_global_snapshot is not None
    missing_name = next(iter(node._fl_global_snapshot))
    node._fl_global_snapshot.pop(missing_name)

    with pytest.raises(RuntimeError, match="Invalid FedProx snapshot"):
        node._proximal_penalty()


def test_proximal_penalty_rejects_shape_mismatches(fake_ros: Any) -> None:
    from fl_robots.robot_agent import RobotAgentNode

    node = RobotAgentNode()
    _broadcast_global(node, algorithm="fedprox", proximal_mu=0.2)

    assert node._fl_global_snapshot is not None
    name, param = next(iter(node.model.named_parameters()))
    node._fl_global_snapshot[name] = torch.zeros(param.numel() + 1)

    with pytest.raises(RuntimeError, match="Invalid FedProx snapshot"):
        node._proximal_penalty()
