"""
Simulation data models for the standalone (non-ROS) simulation engine.

These lightweight dataclasses mirror the ROS2 message types used by the
full system, allowing the simulation to run without any ROS2 dependency.

Classes
-------
Pose2D            – 2-D position + heading.
TrajectoryPoint   – Single waypoint on a planned path.
RobotState        – Full observable state of one robot.
BusEvent          – Immutable record of a single pub/sub event.
AggregationRecord – Summary of one federated-averaging round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Pose2D", "TrajectoryPoint", "RobotState",
    "BusEvent", "AggregationRecord",
]


@dataclass
class Pose2D:
    """2-D pose with heading (radians, counter-clockwise from +x)."""

    x: float
    y: float
    heading: float = 0.0

    def distance_to(self, other: Pose2D) -> float:
        """Euclidean distance to *other* pose."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "heading": self.heading}

    def __repr__(self) -> str:
        return f"Pose2D(x={self.x:.3f}, y={self.y:.3f}, heading={self.heading:.3f})"


@dataclass
class TrajectoryPoint:
    """Single waypoint on a planned trajectory."""

    x: float
    y: float

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass
class RobotState:
    """Full observable state of a single robot agent."""

    robot_id: str
    pose: Pose2D
    velocity: tuple[float, float]
    formation_offset: tuple[float, float]
    goal: tuple[float, float]
    predicted_path: list[TrajectoryPoint] = field(default_factory=list)
    training_loss: float = 1.2
    accuracy: float = 42.0
    training_round: int = 0
    is_training: bool = False
    messages_sent: int = 0
    last_plan_cost: float = 0.0
    last_tracking_error: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "pose": self.pose.as_dict(),
            "velocity": {"x": self.velocity[0], "y": self.velocity[1]},
            "formation_offset": {"x": self.formation_offset[0], "y": self.formation_offset[1]},
            "goal": {"x": self.goal[0], "y": self.goal[1]},
            "predicted_path": [point.as_dict() for point in self.predicted_path],
            "training_loss": self.training_loss,
            "accuracy": self.accuracy,
            "training_round": self.training_round,
            "is_training": self.is_training,
            "messages_sent": self.messages_sent,
            "last_plan_cost": self.last_plan_cost,
            "last_tracking_error": self.last_tracking_error,
        }


@dataclass(frozen=True)
class BusEvent:
    """Immutable record of a single message-bus event."""

    timestamp: float
    topic: str
    source: str
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "topic": self.topic,
            "source": self.source,
            "payload": self.payload,
        }


@dataclass
class AggregationRecord:
    """Summary metrics for one federated-averaging round."""

    round_id: int
    participants: int
    mean_loss: float
    mean_accuracy: float
    mean_divergence: float
    formation_error: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "round": self.round_id,
            "participants": self.participants,
            "mean_loss": self.mean_loss,
            "mean_accuracy": self.mean_accuracy,
            "mean_divergence": self.mean_divergence,
            "formation_error": self.formation_error,
        }

