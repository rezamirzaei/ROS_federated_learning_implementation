"""
Standalone simulation engine — runs the full multi-agent federated-learning
demonstration **without** ROS2.

It uses the in-process :class:`MessageBus` instead of ROS topics and drives
robot motion via the :class:`DistributedMPCPlanner`.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .controller import COMMAND_NAMES, is_valid_command, validate_command
from .message_bus import MessageBus
from .mpc import DistributedMPCPlanner, _safe_atan2
from .sim_models import AggregationRecord, Pose2D, RobotState

__all__ = ["SimulationConfig", "SimulationEngine"]


class SimulationConfig(BaseModel):
    """Validated configuration for the standalone simulation engine."""

    model_config = ConfigDict(frozen=True, slots=True)

    num_robots: int = Field(default=4, ge=1, le=20, description="Number of simulated robots")
    tick_interval: float = Field(default=0.45, gt=0.0, le=5.0, description="Seconds between ticks")
    formation_radius: float = Field(default=1.4, gt=0.0, description="Initial formation radius")
    max_events: int = Field(default=300, ge=10, description="MessageBus event buffer size")
    max_aggregation_history: int = Field(default=60, ge=10)
    max_command_history: int = Field(default=30, ge=5)


class SimulationEngine:
    """ROS-inspired multi-agent system with federated rounds and distributed MPC."""

    #: Delegated to :mod:`fl_robots.controller` so the standalone web layer,
    #: the ROS dashboard, and the engine stay in lockstep.
    _VALID_COMMANDS = frozenset(COMMAND_NAMES)

    def __init__(
        self, num_robots: int = 4, tick_interval: float = 0.45, auto_start: bool = True
    ) -> None:
        cfg = SimulationConfig(num_robots=num_robots, tick_interval=tick_interval)
        self.num_robots = cfg.num_robots
        self.tick_interval = cfg.tick_interval
        self._formation_radius = cfg.formation_radius
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._rng = random.Random(42)
        self.bus = MessageBus(max_events=cfg.max_events)
        self.planner = DistributedMPCPlanner()
        self.robots: dict[str, RobotState] = {}
        self.aggregation_history: deque[AggregationRecord] = deque(
            maxlen=cfg.max_aggregation_history
        )
        self.command_history: deque[str] = deque(maxlen=cfg.max_command_history)
        self.training_active = False
        self.autopilot = True
        self.controller_state = "IDLE"
        self.current_round = 0
        self.tick_count = 0
        self.leader_phase = 0.0
        self.leader_position = (0.0, 0.0)
        self.last_aggregation: AggregationRecord | None = None
        #: Wall-clock timestamp of the last completed tick. Used by the
        #: ``/api/ready`` probe to detect a hung background thread.
        self._last_tick_time: float = time.monotonic()

        self.bus.subscribe("/system/command", self._handle_command_event)
        self._reset_locked()

        if auto_start:
            self.start()

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def is_running(self) -> bool:
        """Return True while the background simulation thread is alive."""
        return bool(self._thread and self._thread.is_alive())

    def seconds_since_last_tick(self) -> float:
        """How long ago (wall-clock) the last simulation step completed.

        Used by the readiness probe — if this exceeds the deployment's
        threshold (default 5 s) the simulation thread is assumed to be
        hung and ``/api/ready`` returns 503.
        """
        return max(0.0, time.monotonic() - self._last_tick_time)

    def issue_command(self, command: str) -> None:
        """Dispatch a user command through the message bus."""
        validate_command(command)  # raises ValueError for unknowns
        self.bus.publish("/system/command", "web-ui", {"command": command})

    def step_once(self) -> None:
        with self._lock:
            self._tick_locked()

    # ── Snapshots / Export ───────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            robots = [robot.as_dict() for robot in self.robots.values()]
            avg_loss = sum(r.training_loss for r in self.robots.values()) / max(len(self.robots), 1)
            avg_accuracy = sum(r.accuracy for r in self.robots.values()) / max(len(self.robots), 1)
            best_accuracy = max((r.accuracy for r in self.robots.values()), default=0.0)
            mean_tracking_error = sum(r.last_tracking_error for r in self.robots.values()) / max(
                len(self.robots), 1
            )

            last_aggregation = self.last_aggregation.as_dict() if self.last_aggregation else None
            return {
                "system": {
                    "controller_state": self.controller_state,
                    "training_active": self.training_active,
                    "autopilot": self.autopilot,
                    "current_round": self.current_round,
                    "tick_count": self.tick_count,
                    "leader_position": {"x": self.leader_position[0], "y": self.leader_position[1]},
                    "robot_count": len(self.robots),
                },
                "metrics": {
                    "avg_loss": avg_loss,
                    "avg_accuracy": avg_accuracy,
                    "best_accuracy": best_accuracy,
                    "mean_tracking_error": mean_tracking_error,
                    "last_aggregation": last_aggregation,
                    "aggregation_history": [rec.as_dict() for rec in self.aggregation_history],
                },
                "robots": robots,
                "messages": [evt.as_dict() for evt in self.bus.recent_events(limit=80)],
                "commands": list(self.command_history),
            }

    def export_results(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        snapshot["exported_at"] = time.time()
        return snapshot

    # ── Internal loop ────────────────────────────────────────────────

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                if self.autopilot:
                    self._tick_locked()
            self._stop_event.wait(self.tick_interval)

    def _handle_command_event(self, event) -> None:
        command = str(event.payload["command"])
        if not is_valid_command(command):
            # Defensive: bus publishers may be untrusted. Silently drop
            # anything we don't understand — never raise inside a bus handler.
            return
        with self._lock:
            self.command_history.appendleft(command)

            if command == "start_training":
                self.training_active = True
                self.controller_state = "RUNNING"
            elif command == "stop_training":
                self.training_active = False
                self.controller_state = "PAUSED"
            elif command == "toggle_autopilot":
                self.autopilot = not self.autopilot
                self.controller_state = "RUNNING" if self.autopilot else "MANUAL"
            elif command == "step":
                self._tick_locked()
            elif command == "disturbance":
                self._inject_disturbance_locked()
            elif command == "reset":
                self._reset_locked()

            self.bus.publish(
                "/fl/training_command",
                "coordinator",
                {
                    "command": command,
                    "round": self.current_round,
                    "controller_state": self.controller_state,
                },
            )

    def _reset_locked(self) -> None:
        self.robots.clear()
        self.aggregation_history.clear()
        self.command_history.clear()
        self.training_active = False
        self.autopilot = True
        self.controller_state = "IDLE"
        self.current_round = 0
        self.tick_count = 0
        self.leader_phase = 0.0
        self.leader_position = (0.0, 0.0)
        self.last_aggregation = None

        for index in range(self.num_robots):
            angle = (2.0 * math.pi * index) / self.num_robots
            robot_id = f"robot_{index + 1}"
            offset = (
                self._formation_radius * math.cos(angle),
                self._formation_radius * math.sin(angle),
            )
            pose = Pose2D(offset[0], offset[1], angle)
            self.robots[robot_id] = RobotState(
                robot_id=robot_id,
                pose=pose,
                velocity=(0.0, 0.0),
                formation_offset=offset,
                goal=offset,
                training_loss=1.15 + (index * 0.08),
                accuracy=45.0 + (index * 3.5),
            )
            self.bus.publish("/fl/robot_status", robot_id, {"status": "registered"})

    def _tick_locked(self) -> None:
        self.tick_count += 1
        self._last_tick_time = time.monotonic()
        # Advance the leader in *world* units (rad/s) rather than per-tick so the
        # trajectory is invariant to tick_interval changes.
        self.leader_phase += 0.6 * self.tick_interval
        self.leader_position = (
            1.4 * math.cos(self.leader_phase * 0.55),
            0.9 * math.sin(self.leader_phase * 0.35),
        )
        plans = self.planner.solve(list(self.robots.values()), self.leader_position)

        min_separation = float("inf")
        formation_error = 0.0
        robot_items = list(self.robots.items())
        for robot_id, robot in robot_items:
            plan = plans[robot_id]
            next_point = plan.path[0]
            robot.velocity = plan.first_velocity
            robot.pose = Pose2D(
                next_point.x,
                next_point.y,
                _safe_atan2(plan.first_velocity[1], plan.first_velocity[0]),
            )
            robot.goal = (
                self.leader_position[0] + robot.formation_offset[0],
                self.leader_position[1] + robot.formation_offset[1],
            )
            robot.predicted_path = plan.path
            robot.last_plan_cost = plan.cost
            robot.last_tracking_error = plan.tracking_error
            robot.is_training = self.training_active
            formation_error += plan.tracking_error
            robot.messages_sent += 2

            self.bus.publish(
                f"/fl/{robot_id}/mpc_plan",
                robot_id,
                {
                    "goal": {"x": robot.goal[0], "y": robot.goal[1]},
                    "tracking_error": plan.tracking_error,
                    "plan_cost": plan.cost,
                },
            )
            self.bus.publish(
                f"/fl/{robot_id}/telemetry",
                robot_id,
                {
                    "pose": robot.pose.as_dict(),
                    "velocity": {"x": robot.velocity[0], "y": robot.velocity[1]},
                    "round": self.current_round,
                },
            )

        for index, (_, robot) in enumerate(robot_items):
            for _, other in robot_items[index + 1 :]:
                min_separation = min(
                    min_separation,
                    math.dist((robot.pose.x, robot.pose.y), (other.pose.x, other.pose.y)),
                )

        if self.training_active:
            self._update_training_locked(formation_error / max(len(self.robots), 1), min_separation)

        self.bus.publish(
            "/fl/coordinator_status",
            "coordinator",
            {
                "state": self.controller_state,
                "current_round": self.current_round,
                "training_active": self.training_active,
                "autopilot": self.autopilot,
            },
        )

    def _update_training_locked(self, formation_error: float, min_separation: float) -> None:
        improvement_factor = max(0.015, 0.05 - (formation_error * 0.01))
        safety_bonus = 0.02 if min_separation > 0.75 else -0.03

        for index, robot in enumerate(self.robots.values(), start=1):
            robot.training_loss = max(
                0.08, robot.training_loss - improvement_factor + (index * 0.002)
            )
            robot.accuracy = min(
                98.5, robot.accuracy + (1.4 - formation_error * 0.1) + safety_bonus
            )

            self.bus.publish(
                f"/fl/{robot.robot_id}/model_weights",
                robot.robot_id,
                {
                    "round": self.current_round + 1,
                    "samples_trained": 256,
                    "loss": robot.training_loss,
                    "accuracy": robot.accuracy,
                },
            )

        if self.tick_count % 5 == 0:
            self.current_round += 1
            divergence = (formation_error * 0.35) + max(0.0, 0.8 - min_separation)
            record = AggregationRecord(
                round_id=self.current_round,
                participants=len(self.robots),
                mean_loss=sum(r.training_loss for r in self.robots.values()) / len(self.robots),
                mean_accuracy=sum(r.accuracy for r in self.robots.values()) / len(self.robots),
                mean_divergence=divergence,
                formation_error=formation_error,
            )
            self.last_aggregation = record
            self.aggregation_history.append(record)

            for robot in self.robots.values():
                robot.training_round = self.current_round

            self.bus.publish("/fl/aggregation_metrics", "aggregator", record.as_dict())

    def _inject_disturbance_locked(self) -> None:
        for robot in self.robots.values():
            robot.pose.x += self._rng.uniform(-0.15, 0.15)
            robot.pose.y += self._rng.uniform(-0.15, 0.15)
        self.controller_state = "RECOVERING"
