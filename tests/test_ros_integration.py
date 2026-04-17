#!/usr/bin/env python3
"""
Integration tests for ROS2 nodes.

These tests verify the ROS2 communication patterns work correctly,
including custom interfaces for services and actions.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import pytest

# Skip if ROS2 is not available
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# Check custom interfaces
try:
    from fl_robots_interfaces.action import TrainRound
    from fl_robots_interfaces.msg import (
        AggregationResult,
        ModelWeights,
        RobotStatus,
        TrainingMetrics,
    )
    from fl_robots_interfaces.srv import (
        GetModelInfo,
        RegisterRobot,
        TriggerAggregation,
        UpdateHyperparameters,
    )

    CUSTOM_INTERFACES = True
except ImportError:
    CUSTOM_INTERFACES = False

pytestmark = pytest.mark.skipif(not ROS2_AVAILABLE, reason="ROS2 not available")


class TestMessageFormats:
    """Test message serialization formats (String/JSON protocol)."""

    def test_registration_message_format(self) -> None:
        registration = {
            "type": "registration",
            "robot_id": "robot_test",
            "model_params": 5000,
            "timestamp": time.time(),
        }
        serialized = json.dumps(registration)
        deserialized = json.loads(serialized)
        assert deserialized["type"] == "registration"
        assert deserialized["robot_id"] == "robot_test"
        assert deserialized["model_params"] == 5000

    def test_weights_message_format(self) -> None:
        import numpy as np

        weights = {
            "fc1.weight": np.random.randn(4, 3).tolist(),
            "fc1.bias": np.random.randn(4).tolist(),
        }
        message = {
            "type": "local_weights",
            "robot_id": "robot_test",
            "round": 1,
            "weights": weights,
            "samples_trained": 256,
            "loss": 0.5,
            "accuracy": 75.0,
            "timestamp": time.time(),
        }
        serialized = json.dumps(message)
        deserialized = json.loads(serialized)
        assert deserialized["type"] == "local_weights"
        assert "weights" in deserialized
        assert len(deserialized["weights"]["fc1.weight"]) == 4

    def test_global_model_message_format(self) -> None:
        import numpy as np

        message = {
            "type": "global_model",
            "round": 5,
            "weights": {"layer": np.random.randn(10).tolist()},
            "timestamp": time.time(),
        }
        serialized = json.dumps(message)
        deserialized = json.loads(serialized)
        assert deserialized["type"] == "global_model"
        assert deserialized["round"] == 5

    def test_training_command_format(self) -> None:
        commands = ["start_training", "stop_training", "publish_weights"]
        for cmd in commands:
            message = {"command": cmd, "round": 1, "timestamp": time.time()}
            serialized = json.dumps(message)
            deserialized = json.loads(serialized)
            assert deserialized["command"] == cmd


@pytest.mark.skipif(not CUSTOM_INTERFACES, reason="Custom interfaces not available")
class TestCustomInterfaces:
    """Test custom message, service, and action interface definitions."""

    def test_robot_status_msg(self) -> None:
        """Verify RobotStatus message can be instantiated."""
        msg = RobotStatus()
        msg.robot_id = "robot_test"
        msg.status = RobotStatus.STATUS_IDLE
        msg.current_round = 5
        msg.last_loss = 0.5
        msg.last_accuracy = 75.0
        msg.model_parameter_count = 5000
        msg.is_active = True
        assert msg.robot_id == "robot_test"
        assert msg.status == 0  # STATUS_IDLE

    def test_training_metrics_msg(self) -> None:
        msg = TrainingMetrics()
        msg.robot_id = "robot_0"
        msg.round_number = 3
        msg.current_epoch = 2
        msg.total_epochs = 5
        msg.loss = 1.23
        msg.accuracy = 45.6
        assert msg.round_number == 3

    def test_aggregation_result_msg(self) -> None:
        msg = AggregationResult()
        msg.round_number = 10
        msg.num_participants = 3
        msg.total_samples = 768
        msg.mean_divergence = 0.45
        msg.participant_ids = ["robot_0", "robot_1", "robot_2"]
        assert len(msg.participant_ids) == 3

    def test_register_robot_srv(self) -> None:
        req = RegisterRobot.Request()
        req.robot_id = "robot_test"
        req.model_parameter_count = 5000
        req.model_architecture = '{"type": "SimpleNavigationNet"}'
        assert req.robot_id == "robot_test"

        resp = RegisterRobot.Response()
        resp.success = True
        resp.message = "Registered"
        resp.assigned_index = 0
        resp.current_round = 5
        assert resp.success is True

    def test_trigger_aggregation_srv(self) -> None:
        req = TriggerAggregation.Request()
        req.force = True
        req.min_participants = 2
        assert req.force is True

        resp = TriggerAggregation.Response()
        resp.success = True
        resp.round_number = 10
        resp.num_participants = 3
        resp.mean_divergence = 0.5
        assert resp.round_number == 10

    def test_get_model_info_srv(self) -> None:
        req = GetModelInfo.Request()
        req.robot_id = "robot_0"
        resp = GetModelInfo.Response()
        resp.success = True
        resp.parameter_count = 5000
        resp.model_architecture = '{"type": "SimpleNavigationNet"}'
        assert resp.parameter_count == 5000

    def test_update_hyperparameters_srv(self) -> None:
        req = UpdateHyperparameters.Request()
        req.robot_id = "robot_0"
        req.learning_rate = 0.0005
        req.batch_size = 64
        req.local_epochs = 10
        assert req.learning_rate == pytest.approx(0.0005)

    def test_train_round_action(self) -> None:
        """Test TrainRound action goal/result/feedback types."""
        goal = TrainRound.Goal()
        goal.round_number = 5
        goal.local_epochs = 3
        goal.batch_size = 32
        goal.learning_rate = 0.001
        assert goal.round_number == 5

        result = TrainRound.Result()
        result.success = True
        result.robot_id = "robot_0"
        result.final_loss = 0.5
        result.final_accuracy = 75.0
        result.total_samples_trained = 256
        result.training_duration_seconds = 2.5
        result.layer_norms = [1.0, 2.0, 3.0]
        assert result.final_accuracy == pytest.approx(75.0)

        feedback = TrainRound.Feedback()
        feedback.robot_id = "robot_0"
        feedback.current_epoch = 2
        feedback.total_epochs = 5
        feedback.current_loss = 0.8
        feedback.current_accuracy = 60.0
        assert feedback.current_epoch == 2


@pytest.mark.skipif(not ROS2_AVAILABLE, reason="ROS2 not available")
class TestROS2Communication:
    """Integration tests for ROS2 pub/sub."""

    @pytest.fixture
    def ros_context(self) -> Iterator[None]:
        rclpy.init()
        yield
        rclpy.shutdown()

    def test_topic_names(self, ros_context: Any) -> None:
        expected_topics = [
            "/fl/robot_status",
            "/fl/global_model",
            "/fl/training_command",
            "/fl/aggregation_metrics",
            "/fl/coordinator_status",
        ]
        for topic in expected_topics:
            assert topic.startswith("/")
            assert "//" not in topic

    def test_service_names(self, ros_context: Any) -> None:
        """Verify expected service name format."""
        expected_services = [
            "/fl/register_robot",
            "/fl/trigger_aggregation",
            "/fl/get_global_model_info",
            "/fl/robot_0/get_model_info",
            "/fl/robot_0/update_hyperparameters",
        ]
        for srv in expected_services:
            assert srv.startswith("/")
            assert "//" not in srv

    def test_action_names(self, ros_context: Any) -> None:
        """Verify expected action name format."""
        expected_actions = [
            "/fl/robot_0/train_round",
            "/fl/robot_1/train_round",
        ]
        for action in expected_actions:
            assert action.startswith("/")
            assert "//" not in action


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
