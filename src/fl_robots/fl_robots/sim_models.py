"""
Simulation data models for the standalone (non-ROS) simulation engine.

These Pydantic models mirror the ROS2 message types used by the full system,
allowing the simulation to run without any ROS2 dependency.  Pydantic provides
runtime validation, immutability control, and schema generation.

Classes
-------
Pose2D            – 2-D position + heading (mutable).
TrajectoryPoint   – Single waypoint on a planned path (frozen).
RobotState        – Full observable state of one robot (mutable).
BusEvent          – Immutable record of a single pub/sub event.
AggregationRecord – Summary of one federated-averaging round (frozen).
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "AggregationRecord",
    "BusEvent",
    "Pose2D",
    "RobotState",
    "TrajectoryPoint",
]


# ── Mutable value objects ─────────────────────────────────────────────


class Pose2D(BaseModel):
    """2-D pose with heading (radians, counter-clockwise from +x)."""

    model_config = ConfigDict(frozen=False, slots=False, validate_assignment=True)

    x: float
    y: float
    heading: float = 0.0

    def __init__(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0, **kwargs: Any) -> None:
        super().__init__(x=x, y=y, heading=heading, **kwargs)

    def distance_to(self, other: Pose2D) -> float:
        """Euclidean distance to *other* pose."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_dict(self) -> dict[str, float]:
        """Serialize to a plain dictionary."""
        return self.model_dump()

    def __repr__(self) -> str:
        return f"Pose2D(x={self.x:.3f}, y={self.y:.3f}, heading={self.heading:.3f})"


# ── Frozen value objects ──────────────────────────────────────────────


class TrajectoryPoint(BaseModel):
    """Single waypoint on a planned trajectory."""

    model_config = ConfigDict(frozen=True, slots=False)

    x: float
    y: float

    def __init__(self, x: float = 0.0, y: float = 0.0, **kwargs: Any) -> None:
        super().__init__(x=x, y=y, **kwargs)

    def as_dict(self) -> dict[str, float]:
        """Serialize to a plain dictionary."""
        return self.model_dump()


class BusEvent(BaseModel):
    """Immutable record of a single message-bus event."""

    model_config = ConfigDict(frozen=True)

    timestamp: float = Field(..., ge=0.0, description="Unix epoch seconds")
    topic: str = Field(..., min_length=1, description="ROS-style topic name")
    source: str = Field(..., min_length=1, description="Publishing node name")
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return self.model_dump()


class AggregationRecord(BaseModel):
    """Summary metrics for one federated-averaging round."""

    model_config = ConfigDict(frozen=True, slots=True)

    round_id: int = Field(..., ge=0)
    participants: int = Field(..., ge=0)
    mean_loss: float = Field(..., ge=0.0)
    mean_accuracy: float
    mean_divergence: float = Field(..., ge=0.0)
    formation_error: float = Field(..., ge=0.0)

    def as_dict(self) -> dict[str, float | int]:
        """Serialize to a plain dictionary."""
        return {
            "round": self.round_id,
            "participants": self.participants,
            "mean_loss": self.mean_loss,
            "mean_accuracy": self.mean_accuracy,
            "mean_divergence": self.mean_divergence,
            "formation_error": self.formation_error,
        }


# ── Mutable runtime state ────────────────────────────────────────────


class RobotState(BaseModel):
    """Full observable state of a single robot agent."""

    model_config = ConfigDict(frozen=False, slots=True, validate_assignment=True)

    robot_id: str = Field(..., min_length=1)
    pose: Pose2D
    velocity: tuple[float, float]
    formation_offset: tuple[float, float]
    goal: tuple[float, float]
    predicted_path: list[TrajectoryPoint] = Field(default_factory=list)
    training_loss: float = Field(default=1.2, ge=0.0)
    accuracy: float = 42.0
    training_round: int = Field(default=0, ge=0)
    is_training: bool = False
    messages_sent: int = Field(default=0, ge=0)
    last_plan_cost: float = 0.0
    last_tracking_error: float = 0.0

    @field_validator("robot_id")
    @classmethod
    def _robot_id_no_whitespace(cls, v: str) -> str:
        if " " in v:
            raise ValueError("robot_id must not contain spaces")
        return v

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary matching the REST snapshot schema."""
        return {
            "robot_id": self.robot_id,
            "pose": self.pose.as_dict(),
            "velocity": {"x": self.velocity[0], "y": self.velocity[1]},
            "formation_offset": {
                "x": self.formation_offset[0],
                "y": self.formation_offset[1],
            },
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
