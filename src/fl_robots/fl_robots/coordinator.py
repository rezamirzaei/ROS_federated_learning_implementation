#!/usr/bin/env python3
"""
Coordinator Node — Training Orchestration.

This node coordinates the federated learning process across all robot agents.
It manages training rounds, monitors progress, and ensures synchronization.

ROS2 Concepts Demonstrated:
- Action clients (initiating training with progress tracking)
- Service clients (checking robot availability)
- Parameter server integration
- State machine for training orchestration
- Lifecycle management patterns
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .ros_compat import (
    DurabilityPolicy,
    HistoryPolicy,
    MultiThreadedExecutor,
    MutuallyExclusiveCallbackGroup,
    Node,
    QoSProfile,
    ReentrantCallbackGroup,
    ReliabilityPolicy,
    String,
    rclpy,
)


class TrainingState(Enum):
    """States for the training state machine."""

    IDLE = auto()
    WAITING_FOR_ROBOTS = auto()
    TRAINING_ROUND = auto()
    AGGREGATING = auto()
    EVALUATING = auto()
    COMPLETED = auto()
    ERROR = auto()


class TrainingConfig(BaseModel):
    """Validated configuration for a training session."""

    model_config = ConfigDict(frozen=True)

    total_rounds: int = Field(default=20, ge=1, description="Total FL rounds")
    min_robots: int = Field(default=2, ge=1, description="Minimum robots to begin")
    round_timeout: float = Field(default=60.0, gt=0.0, description="Round timeout (s)")
    warmup_time: float = Field(default=10.0, ge=0.0, description="Initial warmup delay (s)")
    evaluation_interval: int = Field(default=5, ge=1, description="Evaluate every N rounds")


@dataclass
class RoundStats:
    """Statistics for a training round."""

    round_number: int
    start_time: float
    end_time: float | None = None
    participants: int = 0
    avg_loss: float | None = None
    avg_accuracy: float | None = None


class CoordinatorNode(Node):
    """
    Training Coordinator Node.

    Orchestrates the federated learning process:
    1. Waits for sufficient robots to register
    2. Initiates training rounds
    3. Monitors progress and handles timeouts
    4. Triggers aggregation
    5. Evaluates global model periodically
    6. Manages training completion
    """

    def __init__(self):
        super().__init__("coordinator")

        # Callback groups
        self.cb_group_subs = ReentrantCallbackGroup()
        self.cb_group_timers = MutuallyExclusiveCallbackGroup()

        # Declare parameters
        self._declare_parameters()

        self.config = TrainingConfig(
            total_rounds=self.get_parameter("total_rounds").value,
            min_robots=self.get_parameter("min_robots").value,
            round_timeout=self.get_parameter("round_timeout").value,
        )

        self.get_logger().info("Initializing Training Coordinator")

        # State management
        self.state = TrainingState.IDLE
        self.current_round = 0
        self.round_start_time = 0.0
        self.state_lock = threading.Lock()

        # Robot tracking
        self.registered_robots: dict[str, dict[str, Any]] = {}
        self.round_participants: set = set()

        # Training history (bounded)
        self.round_stats: list[RoundStats] = []
        self.global_metrics: list[dict[str, Any]] = []
        self._max_history = 500

        # QoS profiles
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Publishers
        self.training_command_publisher = self.create_publisher(
            String, "/fl/training_command", qos_reliable
        )

        self.coordinator_status_publisher = self.create_publisher(
            String, "/fl/coordinator_status", qos_reliable
        )

        # Subscribers
        self.robot_status_subscriber = self.create_subscription(
            String,
            "/fl/robot_status",
            self.robot_status_callback,
            qos_reliable,
            callback_group=self.cb_group_subs,
        )

        self.aggregation_metrics_subscriber = self.create_subscription(
            String,
            "/fl/aggregation_metrics",
            self.aggregation_metrics_callback,
            qos_reliable,
            callback_group=self.cb_group_subs,
        )

        # Main coordination timer
        self.coordination_timer = self.create_timer(
            2.0, self.coordination_loop, callback_group=self.cb_group_timers
        )

        # Status publishing timer
        self.status_timer = self.create_timer(
            5.0, self.publish_status, callback_group=self.cb_group_timers
        )

        # Start time for warmup
        self.start_time = time.time()

        self.get_logger().info(f"Coordinator initialized. Config: {self.config}")
        self._transition_to(TrainingState.WAITING_FOR_ROBOTS)

    def _declare_parameters(self):
        """Declare coordinator parameters."""
        self.declare_parameter("total_rounds", 20)
        self.declare_parameter("min_robots", 2)
        self.declare_parameter("round_timeout", 60.0)
        self.declare_parameter("evaluation_interval", 5)

    def _transition_to(self, new_state: TrainingState):
        """Transition to a new state with logging."""
        old_state = self.state
        self.state = new_state
        self.get_logger().info(f"State transition: {old_state.name} -> {new_state.name}")

    def robot_status_callback(self, msg: String):
        """Handle robot status updates."""
        try:
            data = json.loads(msg.data)
            robot_id = data.get("robot_id")
            msg_type = data.get("type")

            if not robot_id:
                return

            with self.state_lock:
                if msg_type == "registration":
                    self.registered_robots[robot_id] = {
                        "registered_at": time.time(),
                        "last_seen": time.time(),
                        "is_training": False,
                    }
                    self.get_logger().info(
                        f"Robot {robot_id} registered. Total: {len(self.registered_robots)}"
                    )

                elif msg_type == "status":
                    if robot_id in self.registered_robots:
                        self.registered_robots[robot_id]["last_seen"] = time.time()
                        self.registered_robots[robot_id]["is_training"] = data.get(
                            "is_training", False
                        )

                        # Track round completion
                        if data.get("training_round") == self.current_round:
                            if not data.get("is_training") and data.get("last_loss") is not None:
                                self.round_participants.add(robot_id)

        except Exception as e:
            self.get_logger().error(f"Error processing robot status: {e}")

    def aggregation_metrics_callback(self, msg: String):
        """Handle aggregation completion notifications."""
        try:
            data = json.loads(msg.data)

            self.global_metrics.append(data)
            if len(self.global_metrics) > self._max_history:
                self.global_metrics = self.global_metrics[-self._max_history :]

            round_num = data.get("round", 0)

            self.get_logger().info(
                f"Aggregation complete for round {round_num}: "
                f"{data.get('num_participants')} participants, "
                f"divergence: {data.get('mean_divergence', 0):.4f}"
            )

            # Update round stats
            if self.round_stats and self.round_stats[-1].round_number == round_num:
                self.round_stats[-1].end_time = time.time()
                self.round_stats[-1].participants = data.get("num_participants", 0)

            # Transition based on current state
            if self.state == TrainingState.TRAINING_ROUND:
                self._transition_to(TrainingState.AGGREGATING)

        except Exception as e:
            self.get_logger().error(f"Error processing aggregation metrics: {e}")

    def coordination_loop(self):
        """Main coordination loop - runs periodically."""
        try:
            with self.state_lock:
                if self.state == TrainingState.WAITING_FOR_ROBOTS:
                    self._handle_waiting_for_robots()

                elif self.state == TrainingState.TRAINING_ROUND:
                    self._handle_training_round()

                elif self.state == TrainingState.AGGREGATING:
                    self._handle_aggregating()

                elif self.state == TrainingState.EVALUATING:
                    self._handle_evaluating()

                elif self.state == TrainingState.COMPLETED:
                    pass  # Training complete

                elif self.state == TrainingState.ERROR:
                    self._handle_error_recovery()

        except Exception as e:
            self.get_logger().error(f"Coordination loop error: {e}")
            self._transition_to(TrainingState.ERROR)

    def _handle_waiting_for_robots(self):
        """Wait for sufficient robots to register."""
        # Allow warmup time
        if time.time() - self.start_time < self.config.warmup_time:
            return

        active_robots = self._count_active_robots()

        if active_robots >= self.config.min_robots:
            self.get_logger().info(f"{active_robots} robots ready, starting training")
            self._start_training_round()
        else:
            self.get_logger().debug(f"Waiting for robots: {active_robots}/{self.config.min_robots}")

    def _handle_training_round(self):
        """Monitor ongoing training round."""
        elapsed = time.time() - self.round_start_time

        # Check for timeout
        if elapsed > self.config.round_timeout:
            self.get_logger().warning(f"Round {self.current_round} timed out after {elapsed:.1f}s")
            self._request_aggregation()
            return

        # Check if all robots have completed
        active_robots = self._count_active_robots()
        completed = len(self.round_participants)

        if completed >= active_robots and completed >= self.config.min_robots:
            self.get_logger().info(f"Round {self.current_round}: All {completed} robots completed")
            # Aggregation will be triggered by aggregator when it has enough weights
            # Wait for aggregation metrics callback

    def _handle_aggregating(self):
        """Handle post-aggregation processing."""
        # Check if we should evaluate
        if self.current_round % self.get_parameter("evaluation_interval").value == 0:
            self._transition_to(TrainingState.EVALUATING)
        else:
            # Start next round
            if self.current_round < self.config.total_rounds:
                self._start_training_round()
            else:
                self._log_milestone()

    def _handle_evaluating(self):
        """Evaluate global model performance."""
        self.get_logger().info(f"Evaluating global model after round {self.current_round}")

        # Log current metrics
        if self.global_metrics:
            latest = self.global_metrics[-1]
            self.get_logger().info(
                f"Current metrics - Round: {self.current_round}, "
                f"Participants: {latest.get('num_participants')}, "
                f"Divergence: {latest.get('mean_divergence', 0):.4f}"
            )

        # Continue training or finish
        if self.current_round < self.config.total_rounds:
            self._start_training_round()
        else:
            self._log_milestone()

    def _handle_error_recovery(self):
        """Attempt to recover from error state."""
        self.get_logger().info("Attempting error recovery...")

        # Reset round participants
        self.round_participants.clear()

        # Check if we have enough robots
        if self._count_active_robots() >= self.config.min_robots:
            self._transition_to(TrainingState.WAITING_FOR_ROBOTS)
        else:
            self.get_logger().warning("Cannot recover: insufficient robots")

    def _start_training_round(self):
        """Start a new training round."""
        self.current_round += 1
        self.round_start_time = time.time()
        self.round_participants.clear()

        # Record round stats
        self.round_stats.append(
            RoundStats(round_number=self.current_round, start_time=self.round_start_time)
        )

        self.get_logger().info(
            f"Starting training round {self.current_round}/{self.config.total_rounds}"
        )

        # Send training command
        self._send_command("start_training")
        self._transition_to(TrainingState.TRAINING_ROUND)

    def _request_aggregation(self):
        """Request aggregation from the aggregator."""
        self._send_command("publish_weights")

    def _log_milestone(self):
        """Log training milestone and continue to the next batch of rounds."""
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"TRAINING MILESTONE: {self.current_round} rounds completed")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Total rounds so far: {self.current_round}")
        self.get_logger().info(f"Total robots: {len(self.registered_robots)}")

        # Log current metrics
        if self.global_metrics:
            final = self.global_metrics[-1]
            self.get_logger().info(f"Current participants: {final.get('num_participants')}")
            self.get_logger().info(f"Current divergence: {final.get('mean_divergence', 0):.4f}")

        self.get_logger().info("Continuing training... (use ./run.sh stop to end)")
        self.get_logger().info("=" * 50)

        # Continue training - don't transition to COMPLETED
        # Just start the next round
        self._start_training_round()

    def _send_command(self, command: str):
        """Send a training command."""
        data = {"command": command, "round": self.current_round, "timestamp": time.time()}

        msg = String()
        msg.data = json.dumps(data)
        self.training_command_publisher.publish(msg)

    def _count_active_robots(self) -> int:
        """Count robots that are currently active."""
        current_time = time.time()
        timeout = 30.0  # Consider robot inactive after 30s

        active = 0
        for info in self.registered_robots.values():
            if current_time - info["last_seen"] < timeout:
                active += 1

        return active

    def publish_status(self):
        """Publish coordinator status."""
        status = {
            "state": self.state.name,
            "current_round": self.current_round,
            "total_rounds": self.config.total_rounds,
            "registered_robots": len(self.registered_robots),
            "active_robots": self._count_active_robots(),
            "round_participants": len(self.round_participants),
            "timestamp": time.time(),
        }

        msg = String()
        msg.data = json.dumps(status)
        self.coordinator_status_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    coordinator = CoordinatorNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(coordinator)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
