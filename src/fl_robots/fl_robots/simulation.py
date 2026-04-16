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
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from .controller import COMMAND_NAMES, is_valid_command, validate_command
from .message_bus import MessageBus
from .mpc import DistributedMPCPlanner, _safe_atan2
from .sim_models import (
    AggregationRecord,
    GlobalMetricPoint,
    MPCRobotDiagnostic,
    MPCSystemDiagnostic,
    Pose2D,
    RobotMetricPoint,
    RobotState,
    TOAEstimatePoint,
    TOASnapshot,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from .localization import (
        ConstantVelocityTargetPredictor as _ConstantVelocityTargetPredictor,
    )
    from .localization import DistributedTOAEstimator as _DistributedTOAEstimator

try:  # Optional — TOA localization needs numpy (shipped via the ml extra).
    from .localization import (
        ConstantVelocityTargetPredictor,
        DistributedTOAEstimator,
        PredictorConfig,
        TOAConfig,
    )

    _TOA_AVAILABLE = True
except ImportError:  # pragma: no cover - numpy missing in minimal install
    DistributedTOAEstimator = None  # type: ignore[assignment,misc]
    TOAConfig = None  # type: ignore[assignment,misc]
    ConstantVelocityTargetPredictor = None  # type: ignore[assignment,misc]
    PredictorConfig = None  # type: ignore[assignment,misc]
    _TOA_AVAILABLE = False

__all__ = ["SimulationConfig", "SimulationEngine"]


class SimulationConfig(BaseModel):
    """Validated configuration for the standalone simulation engine."""

    model_config = ConfigDict(frozen=True)

    num_robots: int = Field(default=4, ge=1, le=20, description="Number of simulated robots")
    tick_interval: float = Field(default=0.45, gt=0.0, le=5.0, description="Seconds between ticks")
    formation_radius: float = Field(default=1.4, gt=0.0, description="Initial formation radius")
    max_events: int = Field(default=300, ge=10, description="MessageBus event buffer size")
    max_aggregation_history: int = Field(default=60, ge=10)
    max_command_history: int = Field(default=30, ge=5)
    max_history: int = Field(
        default=512, ge=32, description="Ring-buffer depth for history time series"
    )
    max_snapshot_history: int = Field(
        default=40, ge=0, description="Slice of history emitted inline in /api/status"
    )
    enable_localization: bool = Field(default=True, description="Run distributed TOA estimator")
    toa_noise_std: float = Field(default=0.05, ge=0.0, le=2.0)
    toa_target: tuple[float, float] = Field(
        default=(1.0, 0.5),
        description=(
            "Ground-truth target the TOA estimator chases. Fixed by default so "
            "RMSE actually collapses; enable ``toa_target_moving`` for a "
            "figure-8 stress test."
        ),
    )
    toa_target_moving: bool = Field(
        default=False,
        description=(
            "If True, overlay a slow figure-8 motion on ``toa_target`` to "
            "exercise the constant-velocity predictor."
        ),
    )
    toa_use_predictor: bool = Field(
        default=True,
        description=(
            "Feed a constant-velocity predicted target into the TOA estimator "
            "as a prior each tick (see TOAConfig.prior_weight)."
        ),
    )
    formation_rotation_rate: float = Field(
        default=0.25,
        description=(
            "Angular rate (rad/s) at which every robot's formation slot rotates "
            "around the leader. 0 freezes the formation into a rigid "
            "translation — which is exactly why, with default params of 0 in "
            "earlier versions, robots appeared to move identically."
        ),
    )


class SimulationEngine:
    """ROS-inspired multi-agent system with federated rounds and distributed MPC."""

    #: Delegated to :mod:`fl_robots.controller` so the standalone web layer,
    #: the ROS dashboard, and the engine stay in lockstep.
    _VALID_COMMANDS = frozenset(COMMAND_NAMES)

    def __init__(
        self, num_robots: int = 4, tick_interval: float = 0.45, auto_start: bool = True
    ) -> None:
        cfg = SimulationConfig(num_robots=num_robots, tick_interval=tick_interval)
        self.cfg = cfg
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
        # ── Time-series ring buffers (for plotting) ─────────────────
        self.global_history: deque[GlobalMetricPoint] = deque(maxlen=cfg.max_history)
        self.robot_history: dict[str, deque[RobotMetricPoint]] = {}
        self.mpc_system_history: deque[MPCSystemDiagnostic] = deque(maxlen=cfg.max_history)
        self.mpc_robot_history: deque[MPCRobotDiagnostic] = deque(maxlen=cfg.max_history * 4)
        self.toa_history: deque[TOASnapshot] = deque(maxlen=cfg.max_history)
        # Latest diagnostics (also pushed to history each tick).
        self.last_mpc_system: MPCSystemDiagnostic | None = None
        self.last_mpc_per_robot: list[MPCRobotDiagnostic] = []
        self.last_toa: TOASnapshot | None = None
        # ── Flags ───────────────────────────────────────────────────
        self.training_active = False
        self.autopilot = True
        self.controller_state = "IDLE"
        self.current_round = 0
        self.tick_count = 0
        self.leader_phase = 0.0
        self.leader_position = (0.0, 0.0)
        # Target the TOA estimator tracks (figure-8 moving beacon).
        self.target_position: tuple[float, float] = (0.0, 0.0)
        self._target_phase = 0.0
        self.last_aggregation: AggregationRecord | None = None
        #: Wall-clock timestamp of the last completed tick. Used by the
        #: ``/api/ready`` probe to detect a hung background thread.
        self._last_tick_time: float = time.monotonic()

        self._toa_estimator: _DistributedTOAEstimator | None = None  # initialised in _reset_locked
        self._target_predictor: _ConstantVelocityTargetPredictor | None = None
        #: Per-robot static formation slot (base_angle, radius). Combined
        #: with ``formation_rotation_rate`` and ``leader_phase`` at each
        #: tick to produce a *rotating* formation — this is what actually
        #: gives each robot a distinct trajectory instead of every robot
        #: rigidly translating with the leader.
        self._formation_slots: dict[str, tuple[float, float]] = {}

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
            history_slice = self.cfg.max_snapshot_history
            global_tail = list(self.global_history)[-history_slice:]
            robot_tail = {
                rid: [p.as_dict() for p in list(buf)[-history_slice:]]
                for rid, buf in self.robot_history.items()
            }
            mpc_tail_robot = [d.as_dict() for d in list(self.mpc_robot_history)[-history_slice:]]
            toa_tail = [t.as_dict() for t in list(self.toa_history)[-history_slice:]]
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
                "history": {
                    "global": [p.as_dict() for p in global_tail],
                    "robots": robot_tail,
                },
                "mpc": {
                    "system": self.last_mpc_system.as_dict() if self.last_mpc_system else None,
                    "per_robot": [d.as_dict() for d in self.last_mpc_per_robot],
                    "history": mpc_tail_robot,
                },
                "localization": {
                    "enabled": self._toa_estimator is not None,
                    "current": self.last_toa.as_dict() if self.last_toa else None,
                    "history": toa_tail,
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
        self.global_history.clear()
        self.robot_history.clear()
        self.mpc_system_history.clear()
        self.mpc_robot_history.clear()
        self.toa_history.clear()
        self.last_mpc_system = None
        self.last_mpc_per_robot = []
        self.last_toa = None
        self.training_active = False
        self.autopilot = True
        self.controller_state = "IDLE"
        self.current_round = 0
        self.tick_count = 0
        self.leader_phase = 0.0
        self.leader_position = (0.0, 0.0)
        self._target_phase = 0.0
        self.target_position = tuple(self.cfg.toa_target)  # type: ignore[assignment]
        self.last_aggregation = None

        self._formation_slots.clear()
        for index in range(self.num_robots):
            angle = (2.0 * math.pi * index) / self.num_robots
            # Stagger radii so robots don't all trace the same circle —
            # paired with formation_rotation_rate this gives each agent a
            # visibly distinct trajectory (fixes the "all robots move the
            # same way" symptom that plagued the rigid-offset formation).
            radius = self._formation_radius * (0.85 + 0.15 * (index % 3))
            self._formation_slots[f"robot_{index + 1}"] = (angle, radius)
            offset = (radius * math.cos(angle), radius * math.sin(angle))
            pose = Pose2D(offset[0], offset[1], angle)
            robot_id = f"robot_{index + 1}"
            self.robots[robot_id] = RobotState(
                robot_id=robot_id,
                pose=pose,
                velocity=(0.0, 0.0),
                formation_offset=offset,
                goal=offset,
                training_loss=1.15 + (index * 0.08),
                accuracy=45.0 + (index * 3.5),
            )
            self.robot_history[robot_id] = deque(maxlen=self.cfg.max_history)
            self.bus.publish("/fl/robot_status", robot_id, {"status": "registered"})

        # TOA estimator is recreated on reset so learnt dual variables don't
        # contaminate a fresh scenario. Skip quietly if NumPy isn't available.
        if self.cfg.enable_localization and _TOA_AVAILABLE and DistributedTOAEstimator is not None:
            self._toa_estimator = DistributedTOAEstimator(
                robot_ids=list(self.robots),
                config=TOAConfig(),
                seed=self._rng.randrange(2**31),
            )
            if self.cfg.toa_use_predictor and ConstantVelocityTargetPredictor is not None:
                # Seed the predictor at the configured target so the very
                # first tick's prior is already sensible.
                self._target_predictor = ConstantVelocityTargetPredictor(
                    x=self.target_position[0],
                    y=self.target_position[1],
                    config=PredictorConfig(dt=self.tick_interval)
                    if PredictorConfig is not None
                    else None,
                )
            else:
                self._target_predictor = None
        else:
            self._toa_estimator = None
            self._target_predictor = None

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

        # ── Rotating formation ──────────────────────────────────────
        # Each robot's formation slot rotates around the leader at
        # ``formation_rotation_rate`` rad/s, and we add a small
        # per-robot radial breathing so trajectories aren't just
        # translated copies of the leader. This is the fix for the
        # "robots all move identically" complaint.
        rot_angle = self.leader_phase * self.cfg.formation_rotation_rate
        breathing = 0.08 * math.sin(self.leader_phase * 0.9)
        for robot_id, robot in self.robots.items():
            base_angle, base_radius = self._formation_slots.get(
                robot_id, (0.0, self._formation_radius)
            )
            # Per-robot phase offset via base_angle means each robot
            # breathes in/out at a different time — visible diversity.
            radius = base_radius + breathing * math.cos(base_angle)
            theta = base_angle + rot_angle
            robot.formation_offset = (radius * math.cos(theta), radius * math.sin(theta))

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

        # Per-robot history — record every tick (even outside aggregation) so
        # the local-loss curves are dense.
        now = time.time()
        for robot in self.robots.values():
            buf = self.robot_history.get(robot.robot_id)
            if buf is None:
                buf = deque(maxlen=self.cfg.max_history)
                self.robot_history[robot.robot_id] = buf
            buf.append(
                RobotMetricPoint(
                    robot_id=robot.robot_id,
                    tick=self.tick_count,
                    round_id=self.current_round,
                    timestamp=now,
                    local_loss=robot.training_loss,
                    local_accuracy=robot.accuracy,
                )
            )

        # MPC diagnostics — ask the planner for last-solve timing/iters.
        if hasattr(self.planner, "diagnostics"):
            try:
                system_diag, per_robot_diag = self.planner.diagnostics(
                    self.tick_count, list(self.robots.values())
                )
            except Exception:  # pragma: no cover - defensive
                system_diag, per_robot_diag = None, []
            if system_diag is not None:
                self.last_mpc_system = system_diag
                self.last_mpc_per_robot = per_robot_diag
                self.mpc_system_history.append(system_diag)
                for d in per_robot_diag:
                    self.mpc_robot_history.append(d)

        # Distributed TOA target localization.
        if self._toa_estimator is not None:
            self._run_toa_locked(now)

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

    def _run_toa_locked(self, now: float) -> None:
        """Run one distributed TOA update against the ground-truth target.

        The target is **fixed** by default (``cfg.toa_target``) so the
        estimator's RMSE actually collapses to the noise floor — the
        previous figure-8 ground truth meant ADMM was forever chasing a
        moving target that the local cost never encoded. Set
        ``cfg.toa_target_moving`` to overlay a slow figure-8 and stress
        the constant-velocity predictor prior.
        """
        base_x, base_y = self.cfg.toa_target
        if self.cfg.toa_target_moving:
            self._target_phase += 0.4 * self.tick_interval
            phase = self._target_phase
            target = (
                base_x + 0.6 * math.sin(phase),
                base_y + 0.4 * math.sin(2.0 * phase),
            )
        else:
            target = (base_x, base_y)
        self.target_position = target

        positions = {rid: (r.pose.x, r.pose.y) for rid, r in self.robots.items()}
        # Noisy TOA / range measurements.
        measurements: dict[str, float] = {}
        for rid, (px, py) in positions.items():
            true_range = math.hypot(target[0] - px, target[1] - py)
            measurements[rid] = true_range + self._rng.gauss(0.0, self.cfg.toa_noise_std)

        # k-NN topology (k=2) — mirrors the paper's local-comms model.
        neighbors = self._build_knn_topology(positions, k=2)

        # Predictor step: advance the motion model, then hand the new
        # predicted position to the estimator as a Bayesian prior. The
        # predictor is updated *after* the TOA pass below with the
        # consensus mean so the next tick sees a better prior.
        predicted_xy: tuple[float, float] | None = None
        if self._target_predictor is not None:
            predicted_xy = self._target_predictor.predict(dt=self.tick_interval)

        assert self._toa_estimator is not None
        result = self._toa_estimator.update(
            sensor_positions=positions,
            measurements=measurements,
            neighbors=neighbors,
            ground_truth=target,
            predicted_target=predicted_xy,
        )

        # Close the loop on the predictor using the consensus mean
        # estimate as the "measurement" of the target position. Using
        # the mean (not any single robot's estimate) stops the predictor
        # from being biased by an outlier sensor.
        if self._target_predictor is not None and result.estimates:
            mean_x = sum(e[0] for e in result.estimates.values()) / len(result.estimates)
            mean_y = sum(e[1] for e in result.estimates.values()) / len(result.estimates)
            self._target_predictor.update((mean_x, mean_y), dt=self.tick_interval)

        estimates = [
            TOAEstimatePoint(
                robot_id=rid,
                x=xy[0],
                y=xy[1],
                residual=result.residuals[rid],
                error=result.errors[rid],
            )
            for rid, xy in result.estimates.items()
        ]
        snap = TOASnapshot(
            tick=self.tick_count,
            timestamp=now,
            target_x=target[0],
            target_y=target[1],
            mean_rmse=result.mean_rmse,
            consensus_gap=result.consensus_gap,
            estimates=estimates,
        )
        self.last_toa = snap
        self.toa_history.append(snap)
        payload: dict[str, Any] = {
            "target": {"x": target[0], "y": target[1]},
            "mean_rmse": result.mean_rmse,
            "consensus_gap": result.consensus_gap,
        }
        if predicted_xy is not None:
            payload["predicted"] = {"x": predicted_xy[0], "y": predicted_xy[1]}
        self.bus.publish("/localization/toa", "toa-estimator", payload)

    @staticmethod
    def _build_knn_topology(
        positions: dict[str, tuple[float, float]], k: int = 2
    ) -> dict[str, list[str]]:
        """Return a map ``i → k nearest neighbours by current Euclidean distance``."""
        out: dict[str, list[str]] = {}
        ids = list(positions)
        for i in ids:
            px, py = positions[i]
            scored = [(math.hypot(px - positions[j][0], py - positions[j][1]), j) for j in ids if j != i]
            scored.sort()
            out[i] = [j for _, j in scored[:k]]
        return out

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
            mean_loss = sum(r.training_loss for r in self.robots.values()) / len(self.robots)
            mean_accuracy = sum(r.accuracy for r in self.robots.values()) / len(self.robots)
            record = AggregationRecord(
                round_id=self.current_round,
                participants=len(self.robots),
                mean_loss=mean_loss,
                mean_accuracy=mean_accuracy,
                mean_divergence=divergence,
                formation_error=formation_error,
            )
            self.last_aggregation = record
            self.aggregation_history.append(record)

            # Validation-loss proxy: modest upward bias + a deterministic
            # divergence-linked bump so the "val" curve sits above "train"
            # the way a real FL run does. Documented in GlobalMetricPoint.
            val_loss = mean_loss * 1.05 + 0.02 * divergence
            val_accuracy = max(0.0, mean_accuracy - 0.4 - 0.5 * divergence)
            self.global_history.append(
                GlobalMetricPoint(
                    tick=self.tick_count,
                    round_id=self.current_round,
                    timestamp=time.time(),
                    mean_loss=mean_loss,
                    val_loss=val_loss,
                    mean_accuracy=mean_accuracy,
                    val_accuracy=val_accuracy,
                    mean_divergence=divergence,
                    formation_error=formation_error,
                )
            )

            for robot in self.robots.values():
                robot.training_round = self.current_round

            self.bus.publish("/fl/aggregation_metrics", "aggregator", record.as_dict())

    def _inject_disturbance_locked(self) -> None:
        for robot in self.robots.values():
            robot.pose.x += self._rng.uniform(-0.15, 0.15)
            robot.pose.y += self._rng.uniform(-0.15, 0.15)
        self.controller_state = "RECOVERING"
