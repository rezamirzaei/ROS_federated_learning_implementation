"""
Distributed Model Predictive Control (MPC) planner for formation control.

Each robot optimises a short-horizon trajectory against goal tracking,
control smoothness, speed cost, and collision avoidance.  Configuration
is validated at construction time via :class:`MPCConfig`.
"""

from __future__ import annotations

import math
import time

from pydantic import BaseModel, ConfigDict, Field

from .sim_models import (
    MPCRobotDiagnostic,
    MPCSystemDiagnostic,
    Pose2D,
    RobotState,
    TrajectoryPoint,
)

__all__ = ["DistributedMPCPlanner", "MPCConfig", "MPCPlan"]


def _safe_atan2(y: float, x: float) -> float:
    """``math.atan2`` that avoids exact-zero *x* values."""
    return math.atan2(y, x if x != 0.0 else 1e-9)


# ── Configuration ─────────────────────────────────────────────────────


class MPCConfig(BaseModel):
    """Validated configuration for the MPC planner."""

    model_config = ConfigDict(frozen=True, slots=True)

    horizon: int = Field(default=8, ge=1, le=50, description="Prediction horizon steps")
    dt: float = Field(default=0.35, gt=0.0, le=2.0, description="Time step (seconds)")
    max_speed: float = Field(default=0.32, gt=0.0, le=5.0, description="Maximum velocity (m/s)")
    safe_distance: float = Field(default=0.55, gt=0.0, description="Collision avoidance radius (m)")


# ── Plan result ───────────────────────────────────────────────────────


class MPCPlan(BaseModel):
    """Result of planning for a single robot over the full horizon."""

    model_config = ConfigDict(frozen=True, slots=True)

    path: list[TrajectoryPoint]
    first_velocity: tuple[float, float]
    cost: float
    tracking_error: float = Field(..., ge=0.0)


# ── Planner ───────────────────────────────────────────────────────────


class DistributedMPCPlanner:
    """Lightweight distributed MPC-style planner."""

    def __init__(
        self,
        horizon: int = 8,
        dt: float = 0.35,
        max_speed: float = 0.32,
    ) -> None:
        cfg = MPCConfig(horizon=horizon, dt=dt, max_speed=max_speed)
        self.horizon = cfg.horizon
        self.dt = cfg.dt
        self.max_speed = cfg.max_speed
        self.safe_distance = cfg.safe_distance
        #: Per-robot diagnostics captured during the last ``solve`` call.
        self._last_solve_time_ms: dict[str, float] = {}
        self._last_control_effort: dict[str, float] = {}
        self._last_tracking_error: dict[str, float] = {}

    def solve(
        self,
        robots: list[RobotState],
        leader_position: tuple[float, float],
    ) -> dict[str, MPCPlan]:
        """Solve for every robot sequentially."""
        plans: dict[str, MPCPlan] = {}
        predicted_neighbors: dict[str, list[TrajectoryPoint]] = {}

        for robot in robots:
            target = (
                leader_position[0] + robot.formation_offset[0],
                leader_position[1] + robot.formation_offset[1],
            )
            t0 = time.perf_counter()
            plan = self._plan_robot(robot, target, predicted_neighbors, robots)
            self._last_solve_time_ms[robot.robot_id] = (time.perf_counter() - t0) * 1e3
            self._last_control_effort[robot.robot_id] = math.hypot(*plan.first_velocity)
            self._last_tracking_error[robot.robot_id] = plan.tracking_error
            plans[robot.robot_id] = plan
            predicted_neighbors[robot.robot_id] = plan.path

        return plans

    # ── Diagnostics ─────────────────────────────────────────────────

    def diagnostics(
        self, tick: int, robots: list[RobotState]
    ) -> tuple[MPCSystemDiagnostic, list[MPCRobotDiagnostic]]:
        """Return last-solve diagnostics tagged with ``tick``.

        The grid-search planner has no explicit QP, so we report ``0`` for
        iterations and tag the status as ``grid-search`` — this keeps the
        dashboard UI uniform across planners.
        """
        per_robot: list[MPCRobotDiagnostic] = []
        for robot in robots:
            per_robot.append(
                MPCRobotDiagnostic(
                    tick=tick,
                    robot_id=robot.robot_id,
                    tracking_error=self._last_tracking_error.get(robot.robot_id, 0.0),
                    control_effort=self._last_control_effort.get(robot.robot_id, 0.0),
                    qp_iterations=0,
                    qp_solve_time_ms=self._last_solve_time_ms.get(robot.robot_id, 0.0),
                    qp_status="grid-search",
                )
            )
        times = list(self._last_solve_time_ms.values()) or [0.0]
        # Grid search has no QP, so variables/constraints are reported as
        # the equivalent "decision surface": horizon × (|candidates|=8).
        system = MPCSystemDiagnostic(
            tick=tick,
            planner_kind="grid-search",
            n_robots=max(len(robots), 1),
            horizon=self.horizon,
            nu=2,
            n_variables=self.horizon * 8,
            n_constraints=0,
            mean_solve_time_ms=sum(times) / len(times),
        )
        return system, per_robot

    def _plan_robot(
        self,
        robot: RobotState,
        target: tuple[float, float],
        predicted_neighbors: dict[str, list[TrajectoryPoint]],
        all_robots: list[RobotState],
    ) -> MPCPlan:
        current = Pose2D(robot.pose.x, robot.pose.y, robot.pose.heading)
        current_velocity = robot.velocity
        path: list[TrajectoryPoint] = []
        total_cost = 0.0
        first_velocity = current_velocity

        for step in range(self.horizon):
            candidates = self._candidate_velocities(current, current_velocity, target)
            best_velocity = current_velocity
            best_point = TrajectoryPoint(current.x, current.y)
            best_cost = float("inf")

            for vx, vy in candidates:
                next_x = current.x + (vx * self.dt)
                next_y = current.y + (vy * self.dt)
                candidate_cost = self._cost(
                    next_x,
                    next_y,
                    vx,
                    vy,
                    current_velocity,
                    target,
                    step,
                    robot.robot_id,
                    predicted_neighbors,
                    all_robots,
                )
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_velocity = (vx, vy)
                    best_point = TrajectoryPoint(next_x, next_y)

            if step == 0:
                first_velocity = best_velocity

            path.append(best_point)
            total_cost += best_cost
            current = Pose2D(
                best_point.x, best_point.y, _safe_atan2(best_velocity[1], best_velocity[0])
            )
            current_velocity = best_velocity

        tracking_error = (
            math.dist((path[-1].x, path[-1].y), target)
            if path
            else math.dist((current.x, current.y), target)
        )
        return MPCPlan(
            path=path,
            first_velocity=first_velocity,
            cost=total_cost / max(self.horizon, 1),
            tracking_error=tracking_error,
        )

    def _candidate_velocities(
        self,
        current: Pose2D,
        current_velocity: tuple[float, float],
        target: tuple[float, float],
    ) -> list[tuple[float, float]]:
        to_target_x = target[0] - current.x
        to_target_y = target[1] - current.y
        target_norm = math.hypot(to_target_x, to_target_y) or 1.0
        desired = (
            self.max_speed * to_target_x / target_norm,
            self.max_speed * to_target_y / target_norm,
        )

        angles = (-0.75, -0.35, 0.0, 0.35, 0.75)
        candidates: list[tuple[float, float]] = [(0.0, 0.0), current_velocity, desired]
        base_angle = _safe_atan2(desired[1], desired[0])

        for offset in angles:
            angle = base_angle + offset
            candidates.append((self.max_speed * math.cos(angle), self.max_speed * math.sin(angle)))

        return candidates

    def _cost(
        self,
        next_x: float,
        next_y: float,
        vx: float,
        vy: float,
        previous_velocity: tuple[float, float],
        target: tuple[float, float],
        step: int,
        robot_id: str,
        predicted_neighbors: dict[str, list[TrajectoryPoint]],
        all_robots: list[RobotState],
    ) -> float:
        tracking_cost = 4.0 * math.dist((next_x, next_y), target)
        smoothness_cost = 0.8 * math.dist((vx, vy), previous_velocity)
        speed_cost = 0.2 * math.hypot(vx, vy)
        separation_cost = 0.0

        for other in all_robots:
            if other.robot_id == robot_id:
                continue

            predicted = predicted_neighbors.get(other.robot_id)
            if predicted and step < len(predicted):
                other_x, other_y = predicted[step].x, predicted[step].y
            else:
                other_x, other_y = other.pose.x, other.pose.y

            distance = math.dist((next_x, next_y), (other_x, other_y))
            if distance < self.safe_distance:
                separation_cost += 10.0 * (self.safe_distance - distance + 1e-3)

        return tracking_cost + smoothness_cost + speed_cost + separation_cost
