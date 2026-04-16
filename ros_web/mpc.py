"""
Distributed Model Predictive Control (MPC) planner for robot formation.

Provides a lightweight formation-control planner that can run alongside the
federated-learning loop, demonstrating how MPC complements learned policies.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class RobotPose:
    """2-D pose of a robot."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass
class MPCConfig:
    """Tunable MPC parameters."""
    horizon: int = 10
    dt: float = 0.1
    max_linear_vel: float = 1.0
    max_angular_vel: float = 1.5
    formation_radius: float = 2.0
    goal_weight: float = 1.0
    formation_weight: float = 0.5
    obstacle_weight: float = 2.0


class DistributedMPCPlanner:
    """
    Simplified distributed MPC for multi-robot formation control.

    Each robot plans locally using its own pose and the *communicated*
    poses of its neighbours.  The planner returns ``(linear_vel, angular_vel)``
    commands that can be published on ``/{robot_id}/cmd_vel``.
    """

    def __init__(self, config: MPCConfig | None = None) -> None:
        self.config = config or MPCConfig()
        self.poses: Dict[str, RobotPose] = {}

    # ── Public API ──────────────────────────────────────────────────

    def update_pose(self, robot_id: str, x: float, y: float, theta: float) -> None:
        self.poses[robot_id] = RobotPose(x, y, theta)

    def plan(self, robot_id: str, goal: Tuple[float, float]) -> Tuple[float, float]:
        """Return (linear_vel, angular_vel) towards *goal* while keeping formation."""
        pose = self.poses.get(robot_id)
        if pose is None:
            return 0.0, 0.0

        # Goal attraction
        dx = goal[0] - pose.x
        dy = goal[1] - pose.y
        dist = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        angle_err = self._wrap(angle_to_goal - pose.theta)

        lin = min(self.config.max_linear_vel, dist) * math.cos(angle_err)
        ang = self.config.max_angular_vel * (2.0 / math.pi) * angle_err

        # Formation repulsion / attraction
        for rid, other in self.poses.items():
            if rid == robot_id:
                continue
            ox = other.x - pose.x
            oy = other.y - pose.y
            d = math.hypot(ox, oy) + 1e-6
            desired = self.config.formation_radius
            force = (d - desired) / d * self.config.formation_weight
            lin += force * math.cos(math.atan2(oy, ox) - pose.theta) * 0.3
            ang += force * math.sin(math.atan2(oy, ox) - pose.theta) * 0.3

        lin = np.clip(lin, -self.config.max_linear_vel, self.config.max_linear_vel)
        ang = np.clip(ang, -self.config.max_angular_vel, self.config.max_angular_vel)
        return float(lin), float(ang)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _wrap(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

