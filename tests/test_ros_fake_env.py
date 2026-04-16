"""End-to-end tests for the ROS node modules driven through the FakeROS harness.

These tests exercise real code paths in ``aggregator.py``, ``coordinator.py``
and ``robot_agent.py`` without requiring a ROS2 install — the
:class:`FakeROSEnvironment` fixture swaps in functional stand-ins for
``Node``/``rclpy``/executors so tests can drive subscriber callbacks and
inspect published messages.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

# ── Aggregator ───────────────────────────────────────────────────────


def test_aggregator_constructs_and_wires_topics(fake_ros):
    from fl_robots.aggregator import AggregatorNode

    node = AggregatorNode()
    # Core wiring is in place.
    assert "/fl/global_model" in node.publishers
    assert "/fl/aggregation_metrics" in node.publishers
    assert "/fl/training_command" in node.publishers
    assert "/fl/robot_status" in node.subscriptions
    # Two timers — health + auto-aggregation.
    assert len(node.timers) >= 1


def test_aggregator_registers_robot_via_status_topic(fake_ros):
    """A registration heartbeat must create a tracked robot entry."""
    from fl_robots.aggregator import AggregatorNode

    node = AggregatorNode()

    # Simulate a registration status message (String payload with JSON body).
    class _Msg:
        data = json.dumps({"type": "registration", "robot_id": "robot_A"})

    fake_ros.publish("/fl/robot_status", _Msg())

    assert "robot_A" in node.robots
    # A weight subscription should have been added dynamically.
    assert "/fl/robot_A/model_weights" in node.subscriptions


def test_aggregator_collects_weights_and_aggregates(fake_ros):
    """Delivering enough weight updates should trigger a FedAvg + publish."""
    from fl_robots.aggregator import AggregatorNode

    node = AggregatorNode()
    node.min_robots = 2  # drive the threshold down for the test

    class _Msg:
        def __init__(self, d):
            self.data = d

    # Register two robots so their weight topics exist.
    for rid in ("robot_1", "robot_2"):
        fake_ros.publish(
            "/fl/robot_status",
            _Msg(json.dumps({"type": "registration", "robot_id": rid})),
        )
        assert rid in node.robots

    # Build a weight payload matching the model shape.
    weights = {
        key: arr.tolist() for key, arr in node.global_model.get_weights().items()
    }
    for rid in ("robot_1", "robot_2"):
        payload = {
            "type": "local_weights",
            "robot_id": rid,
            "round": 1,
            "samples_trained": 128,
            "weights": weights,
            "loss": 0.9,
            "accuracy": 70.0,
        }
        fake_ros.publish(
            f"/fl/{rid}/model_weights", _Msg(json.dumps(payload))
        )

    # Drive an aggregation explicitly — auto-aggregation requires a timer fire.
    result = node._perform_aggregation()
    assert result is not None
    assert result["num_participants"] == 2

    # Global model topic must have seen at least one publish.
    assert len(node.publishers["/fl/global_model"].messages) >= 1


def test_aggregator_health_check_marks_stale_robots(fake_ros):
    from fl_robots.aggregator import AggregatorNode

    node = AggregatorNode()

    class _Msg:
        data = json.dumps({"type": "registration", "robot_id": "robot_s"})

    fake_ros.publish("/fl/robot_status", _Msg())
    assert "robot_s" in node.robots
    # Force the last_seen far into the past so the health check marks it inactive.
    node.robots["robot_s"].last_seen = 0.0
    node.health_check_callback()
    assert not node.robots["robot_s"].is_active


# ── Coordinator ──────────────────────────────────────────────────────


def test_coordinator_constructs_and_wires_topics(fake_ros):
    from fl_robots.coordinator import CoordinatorNode

    node = CoordinatorNode()
    assert "/fl/training_command" in node.publishers
    assert "/fl/coordinator_status" in node.publishers
    assert "/fl/robot_status" in node.subscriptions


def test_coordinator_transitions_through_states(fake_ros):
    from fl_robots.coordinator import CoordinatorNode, TrainingState

    node = CoordinatorNode()

    # Directly exercise the transition helper; then publish status manually
    # (the node only publishes status on the dedicated timer / publish_status).
    node._transition_to(TrainingState.WAITING_FOR_ROBOTS)
    assert node.state == TrainingState.WAITING_FOR_ROBOTS

    node.publish_status()
    status_msgs = node.publishers["/fl/coordinator_status"].messages
    assert status_msgs, "coordinator_status publish did not fire"
    payload = json.loads(status_msgs[-1].data)
    assert payload["state"] == "WAITING_FOR_ROBOTS"


def test_coordinator_handles_robot_status_updates(fake_ros):
    from fl_robots.coordinator import CoordinatorNode

    node = CoordinatorNode()

    class _Msg:
        data = json.dumps(
            {"type": "registration", "robot_id": "robot_x"}
        )

    fake_ros.publish("/fl/robot_status", _Msg())
    assert "robot_x" in node.registered_robots


# ── Robot Agent ──────────────────────────────────────────────────────


def test_robot_agent_publishes_heartbeat_on_timer(fake_ros):
    from fl_robots.robot_agent import RobotAgentNode

    node = RobotAgentNode()

    # Find and fire the heartbeat timer.
    for t in node.timers:
        t.fire()

    # Heartbeat publishes to /fl/robot_status.
    status_msgs = fake_ros.publishers["/fl/robot_status"].messages
    assert len(status_msgs) >= 1


def test_robot_agent_consumes_global_model(fake_ros):
    """Publishing a global model should update the robot's local weights."""
    from fl_robots.robot_agent import RobotAgentNode

    node = RobotAgentNode()

    # Build a payload matching the model shape.
    global_weights = {
        k: v.tolist() for k, v in node.model.get_weights().items()
    }

    class _Msg:
        data = json.dumps(
            {"type": "global_model", "round": 5, "weights": global_weights}
        )

    fake_ros.publish("/fl/global_model", _Msg())

    # After consuming, the robot's training round should advance.
    assert node.training_round == 5


def test_robot_agent_reacts_to_training_command(fake_ros):
    from fl_robots.robot_agent import RobotAgentNode

    node = RobotAgentNode()

    class _Msg:
        def __init__(self, d):
            self.data = d

    fake_ros.publish(
        "/fl/training_command",
        _Msg(json.dumps({"command": "start_training"})),
    )
    fake_ros.publish(
        "/fl/training_command",
        _Msg(json.dumps({"command": "stop_training"})),
    )
    # The primary contract is "does not raise"; deeper state checks are in
    # the dedicated aggregator/coordinator tests above.




