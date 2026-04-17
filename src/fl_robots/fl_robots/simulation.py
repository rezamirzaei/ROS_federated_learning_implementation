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
from .mpc import DistributedMPCPlanner, MPCPlanner, _safe_atan2
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
from .utils.determinism import seed_everything

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

# Deterministic seeded PRNG for reproducible simulations (not cryptographic).
_SeededRNG = random.Random


class SimulationConfig(BaseModel):
    """Validated configuration for the standalone simulation engine."""

    model_config = ConfigDict(frozen=True)

    num_robots: int = Field(default=4, ge=1, le=20, description="Number of simulated robots")
    seed: int = Field(default=42, ge=0, le=2**31 - 1, description="Deterministic seed")
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
    capture_enabled: bool = Field(
        default=True,
        description=(
            "Hunt-the-target game mode: each robot drives toward *its own* "
            "distributed-TOA estimate of the target instead of a formation "
            "slot. When any robot lands within ``capture_radius`` of the "
            "actual target, its score increments and a new target is "
            "spawned inside ``capture_bounds``. Automatically disabled when "
            "``enable_localization`` is off (no estimator → no hunt)."
        ),
    )
    capture_radius: float = Field(
        default=0.25,
        gt=0.0,
        le=2.0,
        description="Distance (m) at which a robot is considered to have captured the target.",
    )
    capture_bounds: float = Field(
        default=1.8,
        gt=0.0,
        description=(
            "Half-extent of the square region (±bounds in x and y) where a new "
            "target spawns after each capture."
        ),
    )
    domain_bounds: float = Field(
        default=3.5,
        gt=0.0,
        description=(
            "Half-extent of the bounded game domain (±bounds in x and y). "
            "Robot positions are clamped to this region every tick."
        ),
    )
    max_capture_history: int = Field(default=30, ge=0)
    capture_win_score: int = Field(
        default=5,
        ge=1,
        description=(
            "First robot to reach this score wins the match — a "
            "``/localization/capture`` event with kind='win' is published "
            "and ``winner_id`` stays set until the next reset."
        ),
    )
    capture_grace_ticks: int = Field(
        default=0,
        ge=0,
        description=(
            "Ticks after a capture during which no new capture can fire. "
            "Set to 0 (default) so a new target is up-for-grabs the very "
            "next tick — keeps the game pace snappy. Raise this if you "
            "want displaced robots to have a beat before the next scramble."
        ),
    )
    capture_cooldown_ticks: int = Field(
        default=4,
        ge=0,
        description=(
            "Ticks during which the *most recent* capturer is ineligible "
            "to capture again. Keeps the game competitive — a robot that "
            "has just scored can't simply camp on the next target."
        ),
    )


class SimulationEngine:
    """ROS-inspired multi-agent system with federated rounds and distributed MPC."""

    #: Delegated to :mod:`fl_robots.controller` so the standalone web layer,
    #: the ROS dashboard, and the engine stay in lockstep.
    _VALID_COMMANDS = frozenset(COMMAND_NAMES)

    def __init__(
        self,
        num_robots: int = 4,
        tick_interval: float = 0.45,
        auto_start: bool = True,
        seed: int = 42,
    ) -> None:
        cfg = SimulationConfig(num_robots=num_robots, tick_interval=tick_interval, seed=seed)
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.num_robots = cfg.num_robots
        self.tick_interval = cfg.tick_interval
        self._formation_radius = cfg.formation_radius
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._rng = _SeededRNG(cfg.seed)
        self.bus = MessageBus(max_events=cfg.max_events)
        self.planner: MPCPlanner = DistributedMPCPlanner()
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
        #: Dedicated RNG for target respawns so captures are reproducible
        #: independent of the noise RNG.
        self._target_rng = _SeededRNG(cfg.seed + 1337)
        #: Recent capture events for UI / dashboards. Each entry is
        #: ``{"tick", "robot_id", "score", "target": {"x","y"}, "new_target": {"x","y"}}``.
        self.capture_events: deque[dict[str, Any]] = deque(maxlen=cfg.max_capture_history)
        self.total_captures: int = 0
        #: Robot that scored the most recent capture — ineligible again
        #: until ``_capture_cooldown_until`` (exclusive).
        self._last_capturer: str | None = None
        self._capture_cooldown_until: int = 0
        #: Tick count at which the global capture grace window ends.
        self._capture_grace_until: int = 0
        #: Winner of the current match, or ``None`` until a robot hits
        #: ``cfg.capture_win_score``. Set by :meth:`_maybe_capture_target_locked`.
        self.winner_id: str | None = None

        self.bus.subscribe("/system/command", self._handle_command_event)
        self._reset_locked()

        if auto_start:
            self.start()

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background simulation loop thread.

        No-op if the thread is already running.
        """
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """Signal the simulation loop to stop and join the thread."""
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
        """Execute a single simulation tick (thread-safe)."""
        with self._lock:
            self._tick_locked()

    # ── Snapshots / Export ───────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a complete JSON-serialisable snapshot of the simulation state."""
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
                "capture": {
                    "enabled": self.cfg.capture_enabled and self._toa_estimator is not None,
                    "target": {"x": self.target_position[0], "y": self.target_position[1]},
                    "radius": self.cfg.capture_radius,
                    "bounds": self.cfg.capture_bounds,
                    "total_captures": self.total_captures,
                    "win_score": self.cfg.capture_win_score,
                    "winner_id": self.winner_id,
                    "scoreboard": sorted(
                        (
                            {"robot_id": r.robot_id, "score": r.capture_score}
                            for r in self.robots.values()
                        ),
                        key=lambda e: (-int(e["score"]), str(e["robot_id"])),
                    ),
                    "events": list(self.capture_events)[:20],
                },
                "robots": robots,
                "messages": [evt.as_dict() for evt in self.bus.recent_events(limit=80)],
                "commands": list(self.command_history),
            }

    def export_results(self) -> dict[str, Any]:
        """Return the current snapshot augmented with an ``exported_at`` timestamp."""
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

    def _handle_command_event(self, event: Any) -> None:
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
        self.capture_events.clear()
        self.total_captures = 0
        self._target_rng = _SeededRNG(self.cfg.seed + 1337)
        self._last_capturer = None
        self._capture_cooldown_until = 0
        self._capture_grace_until = 0
        self.winner_id = None

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
        if (
            self.cfg.enable_localization
            and _TOA_AVAILABLE
            and DistributedTOAEstimator is not None
            and TOAConfig is not None
        ):
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

        # TOA runs at the *start* of the tick so the planner sees the
        # freshest estimates — critical in capture mode where each
        # robot's reference is its own estimate.
        now = time.time()
        if self._toa_estimator is not None:
            self._run_toa_locked(now)

        capture_mode = self.cfg.capture_enabled and self._toa_estimator is not None
        # Horizon used for per-step reference prediction. Both planners
        # expose ``horizon`` on the instance.
        horizon = int(getattr(self.planner, "horizon", 8))
        dt = float(getattr(self.planner, "dt", 0.35))

        refs: dict[str, list[tuple[float, float]]] = {}

        if capture_mode:
            # ── Hunt mode ───────────────────────────────────────────
            # Each robot steers toward its *own* distributed-TOA estimate
            # of the target. For the horizon we extrapolate the target
            # with the shared α-β motion model (fallback = static), so
            # chasing robots anticipate where the target is *going*, not
            # just where it is now. This fixes the characteristic
            # "one-tick lag" behaviour of a static-reference MPC.
            tgt_now = self.target_position
            tgt_traj: list[tuple[float, float]] = []
            if self._target_predictor is not None:
                # Snapshot predictor state, project forward, restore — so
                # the simulation's actual predictor isn't advanced by the
                # planner projection.
                px, py, vx, vy = self._target_predictor.state
                tgt_traj = [
                    (px + vx * dt * (k + 1), py + vy * dt * (k + 1)) for k in range(horizon)
                ]
            else:
                tgt_traj = [tgt_now] * horizon
            for robot_id, robot in self.robots.items():
                est = self._toa_estimator.estimate(robot_id)  # type: ignore[union-attr]
                # formation_offset describes the *current* slot (k=0)
                # relative to the planner leader — used by dashboards
                # and snapshots. The horizon-length lookahead lives in
                # ``refs[robot_id]`` instead, so observers stay stable.
                robot.formation_offset = (est[0] - tgt_now[0], est[1] - tgt_now[1])
                # Per-step reference = robot's private estimate shifted
                # by the *predicted* target motion at each step.
                refs[robot_id] = [
                    (est[0] + (tx - tgt_now[0]), est[1] + (ty - tgt_now[1]))
                    for (tx, ty) in tgt_traj
                ]
            planner_leader = tgt_now
        else:
            # ── Rotating formation ──────────────────────────────────
            # Each robot's formation slot rotates around the leader at
            # ``formation_rotation_rate`` rad/s. For the horizon we
            # extrapolate both the leader's circular motion *and* the
            # slot rotation, so followers plan against a *moving* target
            # rather than a static snapshot — a measurable tracking-
            # error improvement at nontrivial formation rotation rates.
            rot_rate = self.cfg.formation_rotation_rate
            leader_omega_a = 0.6 * 0.55
            leader_omega_b = 0.6 * 0.35
            for robot_id, robot in self.robots.items():
                base_angle, base_radius = self._formation_slots.get(
                    robot_id, (0.0, self._formation_radius)
                )
                per_robot_ref: list[tuple[float, float]] = []
                for k in range(horizon):
                    # Elapsed *world* time at horizon step k (k=0 is the
                    # control applied this tick, so planning happens in
                    # one-step-ahead land).
                    future_phase = self.leader_phase + (k + 1) * dt * (1.0 / self.tick_interval) * (
                        0.6 * self.tick_interval
                    )
                    leader_x = 1.4 * math.cos(future_phase * leader_omega_a / 0.6)
                    leader_y = 0.9 * math.sin(future_phase * leader_omega_b / 0.6)
                    theta = base_angle + future_phase * rot_rate
                    breathing = 0.08 * math.sin(future_phase * 0.9)
                    radius = base_radius + breathing * math.cos(base_angle)
                    per_robot_ref.append(
                        (leader_x + radius * math.cos(theta), leader_y + radius * math.sin(theta))
                    )
                refs[robot_id] = per_robot_ref
                # Keep formation_offset consistent with the k=0 step so
                # downstream consumers (snapshot, goal field, dashboards)
                # see a coherent "current slot".
                robot.formation_offset = (
                    per_robot_ref[0][0] - self.leader_position[0],
                    per_robot_ref[0][1] - self.leader_position[1],
                )
            planner_leader = self.leader_position

        plans = self.planner.solve_with_refs(list(self.robots.values()), refs)

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
            # Clamp robot positions to the bounded game domain.
            b = self.cfg.domain_bounds
            robot.pose.x = max(-b, min(b, robot.pose.x))
            robot.pose.y = max(-b, min(b, robot.pose.y))
            robot.goal = (
                planner_leader[0] + robot.formation_offset[0],
                planner_leader[1] + robot.formation_offset[1],
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

        # Capture check — runs whether or not capture mode is "officially"
        # on, so rigid-formation runs also get credit if a robot strays
        # into the target. Cheap (O(N)).
        if capture_mode:
            self._maybe_capture_target_locked()

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

        # Distributed TOA target localization: runs at the *start* of
        # each tick (see top of ``_tick_locked``) so the planner sees
        # fresh estimates. Nothing to do here.

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

        In capture mode the ground truth is driven by
        :meth:`_maybe_capture_target_locked` (respawns on each capture),
        so we simply use whatever ``self.target_position`` is instead of
        re-reading from config — otherwise every tick would clobber the
        freshly-spawned target.
        """
        capture_driven = self.cfg.capture_enabled and self._toa_estimator is not None
        if capture_driven:
            target = self.target_position
        else:
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

        if self._toa_estimator is None:
            msg = "TOA estimator is not initialised"
            raise RuntimeError(msg)
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

    def _maybe_capture_target_locked(self) -> None:
        """Award a point to the closest eligible robot inside the capture
        radius and spawn a fresh target.

        Eligibility rules (all configurable):

        * During the **grace window** after any capture (next
          ``cfg.capture_grace_ticks`` ticks) nothing fires — gives
          displaced robots time to notice the new target.
        * The **most recent capturer** is on cooldown for
          ``cfg.capture_cooldown_ticks`` ticks, so the same robot can't
          simply camp the next target.
        * The match ends when a robot reaches ``cfg.capture_win_score``;
          ``winner_id`` is set and a ``kind='win'`` event is emitted. New
          captures still count after that — useful for endless mode —
          but we stop re-announcing the win.
        """
        if self.winner_id is not None:
            # Auto-reset scores so the game continues indefinitely.
            for robot in self.robots.values():
                robot.capture_score = 0
            self.winner_id = None
            self._last_capturer = None
            self._capture_cooldown_until = 0
            self._capture_grace_until = self.tick_count + self.cfg.capture_grace_ticks + 5
        if self.tick_count < self._capture_grace_until:
            return

        tgt = self.target_position
        # Find the closest *eligible* robot and its distance.
        closest_id: str | None = None
        closest_dist = float("inf")
        on_cooldown = (
            self._last_capturer if self.tick_count < self._capture_cooldown_until else None
        )
        for rid, robot in self.robots.items():
            if rid == on_cooldown:
                continue
            d = math.hypot(robot.pose.x - tgt[0], robot.pose.y - tgt[1])
            if d < closest_dist:
                closest_dist = d
                closest_id = rid
        if closest_id is None or closest_dist > self.cfg.capture_radius:
            return

        robot = self.robots[closest_id]
        robot.capture_score += 1
        self.total_captures += 1
        self._last_capturer = closest_id
        self._capture_cooldown_until = self.tick_count + self.cfg.capture_cooldown_ticks
        self._capture_grace_until = self.tick_count + self.cfg.capture_grace_ticks

        new_target = self._spawn_target()
        self.target_position = new_target

        # Reset the estimator + predictor so they don't drag stale state
        # toward the old target. The predictor is re-seeded at the new
        # target so the first post-capture tick already has a sensible
        # prior, while the TOA estimator's duals are cleared and
        # per-sensor estimates re-randomised (keeps the ADMM well-conditioned).
        if self._toa_estimator is not None:
            self._toa_estimator.reset()
        if self._target_predictor is not None:
            self._target_predictor.reset(x=new_target[0], y=new_target[1])

        won = robot.capture_score >= self.cfg.capture_win_score
        if won:
            self.winner_id = closest_id

        event = {
            "tick": self.tick_count,
            "kind": "win" if won else "capture",
            "robot_id": closest_id,
            "score": robot.capture_score,
            "total_captures": self.total_captures,
            "target": {"x": tgt[0], "y": tgt[1]},
            "new_target": {"x": new_target[0], "y": new_target[1]},
            "win_score": self.cfg.capture_win_score,
            "winner_id": self.winner_id,
        }
        self.capture_events.appendleft(event)
        self.bus.publish("/localization/capture", "simulation", event)

    def _spawn_target(self) -> tuple[float, float]:
        """Sample a new target position inside ``±capture_bounds`` that is
        at least ``1.5·capture_radius`` away from every robot."""
        bounds = self.cfg.capture_bounds
        min_dist = 1.5 * self.cfg.capture_radius
        # Cap attempts so we never loop forever if robots saturate the arena.
        for _ in range(20):
            x = self._target_rng.uniform(-bounds, bounds)
            y = self._target_rng.uniform(-bounds, bounds)
            if all(
                math.hypot(r.pose.x - x, r.pose.y - y) >= min_dist for r in self.robots.values()
            ):
                return (x, y)
        # Fall back to an unconstrained sample — the next tick will sort it out.
        return (
            self._target_rng.uniform(-bounds, bounds),
            self._target_rng.uniform(-bounds, bounds),
        )

    @staticmethod
    def _build_knn_topology(
        positions: dict[str, tuple[float, float]], k: int = 2
    ) -> dict[str, list[str]]:
        """Return a map ``i → k nearest neighbours by current Euclidean distance``."""
        out: dict[str, list[str]] = {}
        ids = list(positions)
        for i in ids:
            px, py = positions[i]
            scored = [
                (math.hypot(px - positions[j][0], py - positions[j][1]), j) for j in ids if j != i
            ]
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
            # Add stochastic noise to prevent monotonic overfitting —
            # simulates real-world variance across local SGD rounds.
            noise = self._rng.gauss(0.0, 0.3)
            raw_acc = robot.accuracy + (1.4 - formation_error * 0.1) + safety_bonus + noise
            # Clamp accuracy to [0, 98.5] — never negative, never perfect.
            robot.accuracy = max(0.0, min(98.5, raw_acc))

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
            mean_accuracy = max(
                0.0, sum(r.accuracy for r in self.robots.values()) / len(self.robots)
            )
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
        b = self.cfg.domain_bounds
        for robot in self.robots.values():
            robot.pose.x = max(-b, min(b, robot.pose.x + self._rng.uniform(-0.15, 0.15)))
            robot.pose.y = max(-b, min(b, robot.pose.y + self._rng.uniform(-0.15, 0.15)))
        self.controller_state = "RECOVERING"
