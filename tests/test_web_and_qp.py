"""Tests for the auth layer, `/metrics` endpoint, and the QP MPC planner."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest
from fl_robots.simulation import SimulationEngine
from fl_robots.standalone_web import create_app

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def app_with_token(monkeypatch: Any) -> Iterator[Any]:
    """Flask app with a bearer token set. Simulation paused for determinism."""
    monkeypatch.setenv("FL_ROBOTS_API_TOKEN", "secret-xyz")
    sim = SimulationEngine(num_robots=3, auto_start=False)
    app = create_app(sim)
    yield app, sim
    sim.shutdown()


@pytest.fixture
def app_no_token() -> Iterator[Any]:
    sim = SimulationEngine(num_robots=3, auto_start=False)
    app = create_app(sim)
    yield app, sim
    sim.shutdown()


def test_command_requires_bearer_when_token_set(app_with_token: Any) -> None:
    app, _ = app_with_token
    client = app.test_client()
    rv = client.post("/api/command", json={"command": "step"})
    assert rv.status_code == 401


def test_command_accepts_valid_bearer(app_with_token: Any) -> None:
    app, _ = app_with_token
    client = app.test_client()
    rv = client.post(
        "/api/command",
        json={"command": "step"},
        headers={"Authorization": "Bearer secret-xyz"},
    )
    assert rv.status_code == 200
    assert rv.get_json()["ok"] is True


def test_command_rejects_wrong_bearer(app_with_token: Any) -> None:
    app, _ = app_with_token
    client = app.test_client()
    rv = client.post(
        "/api/command",
        json={"command": "step"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert rv.status_code == 401


def test_command_without_token_env_requires_csrf(app_no_token: Any) -> None:
    app, _ = app_no_token
    client = app.test_client()
    rv = client.post("/api/command", json={"command": "step"})
    assert rv.status_code == 403


def test_command_without_token_env_accepts_valid_csrf(app_no_token: Any, csrf_headers: Any) -> None:
    app, _ = app_no_token
    client = app.test_client()
    rv = client.post("/api/command", json={"command": "step"}, headers=csrf_headers(client))
    assert rv.status_code == 200


def test_metrics_endpoint_returns_prometheus_text(app_no_token: Any, csrf_headers: Any) -> None:
    app, sim = app_no_token
    sim.step_once()  # generate one tick so gauges are non-zero
    client = app.test_client()
    client.get("/api/status")
    client.post("/api/command", json={"command": "step"}, headers=csrf_headers(client))
    rv = client.get("/metrics")
    assert rv.status_code == 200
    body = rv.data.decode("utf-8")
    # Prometheus exposition format — HELP + TYPE lines plus values.
    for metric in (
        "fl_robot_count",
        "fl_avg_accuracy",
        "fl_tick_count",
        "fl_controller_state",
        "fl_training_active",
        "fl_aggregation_divergence",
        "fl_http_requests_total",
        "fl_http_request_duration_seconds_bucket",
        "fl_tracking_error_bucket",
        "fl_mpc_solve_time_ms_bucket",
    ):
        assert metric in body, f"missing metric {metric}"


def test_metrics_content_type(app_no_token: Any) -> None:
    app, _ = app_no_token
    client = app.test_client()
    rv = client.get("/metrics")
    assert rv.content_type.startswith("text/plain")


def test_status_exposes_full_capture_payload_for_ui(app_no_token: Any) -> None:
    """The standalone dashboard relies on a specific capture payload shape.
    Lock it in so a future refactor doesn't silently break the UI."""
    app, sim = app_no_token
    # Drive one capture so the ``events`` array is non-empty.
    import pytest as _pt

    _pt.importorskip("numpy")
    rid, robot = next(iter(sim.robots.items()))
    robot.pose.x, robot.pose.y = sim.target_position
    sim.step_once()

    client = app.test_client()
    rv = client.get("/api/status")
    assert rv.status_code == 200
    body = rv.get_json()

    assert "capture" in body
    cap = body["capture"]
    for key in (
        "enabled",
        "target",
        "radius",
        "total_captures",
        "win_score",
        "winner_id",
        "scoreboard",
        "events",
    ):
        assert key in cap, f"capture payload missing {key!r}"
    assert {"x", "y"} <= set(cap["target"])
    # Scoreboard ranked descending and includes every robot.
    ids = {e["robot_id"] for e in cap["scoreboard"]}
    assert ids == set(sim.robots)
    scores = [e["score"] for e in cap["scoreboard"]]
    assert scores == sorted(scores, reverse=True)
    # Per-robot payload must expose capture_score for the SVG badge.
    assert all("capture_score" in r for r in body["robots"])
    # At least one capture event is present and has the UI's required fields.
    assert cap["events"], "expected a capture event after driving a capture"
    evt = cap["events"][0]
    for key in ("tick", "robot_id", "score", "kind"):
        assert key in evt


# ── QP MPC ──────────────────────────────────────────────────────────

osqp = pytest.importorskip("osqp")
scipy = pytest.importorskip("scipy")


def test_qp_planner_drives_single_robot_toward_target() -> None:
    from fl_robots.mpc_qp import QPMPCPlanner
    from fl_robots.sim_models import Pose2D, RobotState

    planner = QPMPCPlanner(horizon=6, dt=0.3, max_speed=0.3)
    robot = RobotState(
        robot_id="r0",
        pose=Pose2D(0.0, 0.0),
        velocity=(0.0, 0.0),
        formation_offset=(1.5, 0.0),
        goal=(1.5, 0.0),
    )
    plan = planner.solve([robot], leader_position=(0.0, 0.0))["r0"]
    assert len(plan.path) == 6
    # Should make progress toward x=1.5.
    assert plan.path[-1].x > 0.0
    # Tracking error must shrink from initial 1.5.
    assert plan.tracking_error < 1.5


def test_qp_planner_preserves_separation_for_formation() -> None:
    from fl_robots.mpc_qp import QPMPCPlanner
    from fl_robots.sim_models import Pose2D, RobotState

    planner = QPMPCPlanner(horizon=5, dt=0.25)
    robots = []
    for i, angle in enumerate((0.0, 2 * math.pi / 3, 4 * math.pi / 3)):
        offset = (1.0 * math.cos(angle), 1.0 * math.sin(angle))
        robots.append(
            RobotState(
                robot_id=f"r{i}",
                pose=Pose2D(*offset),
                velocity=(0.0, 0.0),
                formation_offset=offset,
                goal=offset,
            )
        )

    plans = planner.solve(robots, leader_position=(0.0, 0.0))
    first_points = [(plans[r.robot_id].path[0].x, plans[r.robot_id].path[0].y) for r in robots]
    # Every pair should remain at a plausible distance (loose bound because the
    # QP penalty is soft).
    for i, pi in enumerate(first_points):
        for pj in first_points[i + 1 :]:
            assert math.dist(pi, pj) > 0.3


def test_qp_planner_shape_matches_grid_planner() -> None:
    """Drop-in compatibility with DistributedMPCPlanner."""
    from fl_robots.mpc import DistributedMPCPlanner, MPCPlan
    from fl_robots.mpc_qp import QPMPCPlanner
    from fl_robots.sim_models import Pose2D, RobotState

    robot = RobotState(
        robot_id="r0",
        pose=Pose2D(0.0, 0.0),
        velocity=(0.0, 0.0),
        formation_offset=(1.0, 0.5),
        goal=(1.0, 0.5),
    )
    grid = DistributedMPCPlanner(horizon=4)
    qp = QPMPCPlanner(horizon=4)
    plan_grid = grid.solve([robot], (0.0, 0.0))["r0"]
    plan_qp = qp.solve([robot], (0.0, 0.0))["r0"]
    for p in (plan_grid, plan_qp):
        assert isinstance(p, MPCPlan)
        assert len(p.path) == 4
        assert p.tracking_error >= 0.0


def test_qp_planner_respects_max_speed_box_constraint() -> None:
    """The L_∞ velocity bound ‖U_k‖_∞ ≤ u_max must hold for every step."""
    from fl_robots.mpc_qp import QPMPCPlanner
    from fl_robots.sim_models import Pose2D, RobotState

    u_max = 0.25
    planner = QPMPCPlanner(horizon=6, dt=0.3, max_speed=u_max)
    # Put the target *far* away — the unconstrained optimum would exceed u_max.
    robot = RobotState(
        robot_id="r0",
        pose=Pose2D(0.0, 0.0),
        velocity=(0.0, 0.0),
        formation_offset=(10.0, 10.0),
        goal=(10.0, 10.0),
    )
    plan = planner.solve([robot], leader_position=(0.0, 0.0))["r0"]

    # First velocity must lie inside the box.
    vx, vy = plan.first_velocity
    assert abs(vx) <= u_max + 1e-3, f"|vx|={abs(vx)} exceeds u_max={u_max}"
    assert abs(vy) <= u_max + 1e-3, f"|vy|={abs(vy)} exceeds u_max={u_max}"

    # Step-to-step displacement must stay within the implied position-change bound.
    prev = (0.0, 0.0)
    for p1 in plan.path:
        dx, dy = p1.x - prev[0], p1.y - prev[1]
        assert abs(dx) <= planner.dt * u_max + 1e-3
        assert abs(dy) <= planner.dt * u_max + 1e-3
        prev = (p1.x, p1.y)


def test_qp_warm_start_does_not_regress_iterations() -> None:
    """Re-solving a near-identical problem must not require more iterations."""
    from fl_robots.mpc_qp import QPMPCPlanner
    from fl_robots.sim_models import Pose2D, RobotState

    planner = QPMPCPlanner(horizon=8, dt=0.3)
    robot = RobotState(
        robot_id="r0",
        pose=Pose2D(0.0, 0.0),
        velocity=(0.0, 0.0),
        formation_offset=(1.5, 0.5),
        goal=(1.5, 0.5),
    )
    planner.solve([robot], leader_position=(0.0, 0.0))
    cold_iters = planner.last_iterations["r0"]
    first_solver = planner._solver_cache["r0"]["solver"]
    planner.solve([robot], leader_position=(0.0, 0.0))
    warm_iters = planner.last_iterations["r0"]
    assert planner._solver_cache["r0"]["solver"] is first_solver
    assert warm_iters <= cold_iters, f"warm start regressed: cold={cold_iters}, warm={warm_iters}"


def test_qp_warm_cache_tolerates_stale_shapes() -> None:
    """A pollutted cache (wrong shape) must not crash the next solve."""
    from fl_robots.mpc_qp import QPMPCPlanner
    from fl_robots.sim_models import Pose2D, RobotState

    planner = QPMPCPlanner(horizon=4, dt=0.3)
    robot = RobotState(
        robot_id="r0",
        pose=Pose2D(0.0, 0.0),
        velocity=(0.0, 0.0),
        formation_offset=(0.5, 0.0),
        goal=(0.5, 0.0),
    )
    planner.solve([robot], leader_position=(0.0, 0.0))
    u, y = planner._warm_cache["r0"]
    # Inject a truncated primal so the shape-guard must reject the warm start.
    planner._warm_cache["r0"] = (u[:-2], y)
    plan = planner.solve([robot], leader_position=(0.0, 0.0))["r0"]
    assert plan.tracking_error >= 0.0


def test_qp_planner_respects_slew_limit() -> None:
    """Step-to-step velocity change must stay within the configured slew."""
    from fl_robots.mpc_qp import QPMPCPlanner
    from fl_robots.sim_models import Pose2D, RobotState

    du = 0.04
    planner = QPMPCPlanner(horizon=6, dt=0.3, max_speed=0.3, slew_limit=du)
    robot = RobotState(
        robot_id="r0",
        pose=Pose2D(0.0, 0.0),
        velocity=(0.0, 0.0),
        formation_offset=(5.0, 0.0),
        goal=(5.0, 0.0),
    )
    plan = planner.solve([robot], leader_position=(0.0, 0.0))["r0"]
    # First-step slew: must be within du of zero starting velocity.
    assert abs(plan.first_velocity[0]) <= du + 1e-3
    assert abs(plan.first_velocity[1]) <= du + 1e-3


def test_rotating_formation_produces_diverse_motion() -> None:
    """Robots must not all move in lockstep — the rotating formation
    and per-robot breathing should spread their velocities."""
    sim = SimulationEngine(num_robots=4, tick_interval=0.4, auto_start=False)
    # Step a few ticks so motion develops.
    for _ in range(5):
        sim.step_once()
    velocities = [r.velocity for r in sim.robots.values()]
    # At least one pair of robots must have noticeably different velocity
    # vectors — the rigid-offset bug produced identical (vx,vy) for all.
    max_gap = 0.0
    for i, vi in enumerate(velocities):
        for vj in velocities[i + 1 :]:
            max_gap = max(max_gap, math.hypot(vi[0] - vj[0], vi[1] - vj[1]))
    assert max_gap > 0.02, f"robots moving in lockstep (max velocity gap={max_gap})"
    sim.shutdown()


def test_capture_mode_awards_score_and_respawns_target() -> None:
    """Place a robot on top of the target → it must capture, score++, target moves."""
    pytest.importorskip("numpy")  # TOA/capture needs numpy
    sim = SimulationEngine(num_robots=3, tick_interval=0.4, auto_start=False)
    # Move one robot directly onto the target so the capture radius hits.
    rid, robot = next(iter(sim.robots.items()))
    tgt = sim.target_position
    robot.pose.x = tgt[0]
    robot.pose.y = tgt[1]
    old_target = tgt

    # Snapshot should include the capture section.
    snap = sim.snapshot()
    assert "capture" in snap
    assert snap["capture"]["total_captures"] == 0

    sim.step_once()

    # Score incremented, target moved, history populated.
    assert sim.robots[rid].capture_score >= 1, "robot at the target failed to capture"
    assert sim.total_captures >= 1
    assert sim.target_position != old_target, "target did not respawn after capture"
    assert len(sim.capture_events) >= 1
    first = sim.capture_events[0]
    assert first["robot_id"] == rid
    # Scoreboard is ranked by score descending; winner must be on top.
    board = sim.snapshot()["capture"]["scoreboard"]
    assert board[0]["robot_id"] == rid
    assert board[0]["score"] >= 1
    sim.shutdown()


def test_capture_mode_drives_robots_toward_estimates() -> None:
    """In capture mode, each robot's formation_offset should encode its
    private TOA estimate (not a rigid formation slot)."""
    pytest.importorskip("numpy")
    sim = SimulationEngine(num_robots=3, tick_interval=0.4, auto_start=False)
    # Run enough ticks for the TOA estimator to start producing estimates.
    for _ in range(3):
        sim.step_once()
    assert sim._toa_estimator is not None
    tgt = sim.target_position
    for rid, robot in sim.robots.items():
        est = sim._toa_estimator.estimate(rid)
        expected_offset = (est[0] - tgt[0], est[1] - tgt[1])
        # formation_offset should match (est − target) up to floating point.
        assert math.isclose(robot.formation_offset[0], expected_offset[0], abs_tol=1e-6)
        assert math.isclose(robot.formation_offset[1], expected_offset[1], abs_tol=1e-6)
    sim.shutdown()


def test_capture_is_instant_by_default_next_tick_eligible() -> None:
    """With the default ``capture_grace_ticks=0``, a *different* robot can
    score on the tick immediately following someone else's capture.

    This locks in the "agents can start moving/scoring from the beginning"
    contract — the game must not pause between captures.
    """
    pytest.importorskip("numpy")
    sim = SimulationEngine(num_robots=3, tick_interval=0.4, auto_start=False)
    assert sim.cfg.capture_grace_ticks == 0, (
        "grace window must default to 0 so the game stays snappy"
    )
    ids = list(sim.robots)
    first, second = ids[0], ids[1]

    # Robot A parks on the target and captures.
    sim.robots[first].pose.x, sim.robots[first].pose.y = sim.target_position
    sim.step_once()
    assert sim.robots[first].capture_score == 1
    # On the VERY NEXT tick, a different robot (not on cooldown) must be
    # able to claim the freshly-spawned target.
    sim.robots[second].pose.x, sim.robots[second].pose.y = sim.target_position
    sim.step_once()
    assert sim.robots[second].capture_score == 1, (
        "grace window must not block a different robot on the next tick"
    )
    sim.shutdown()


def test_capture_cooldown_blocks_same_robot_from_double_scoring() -> None:
    """After a capture, the same robot is on cooldown for a few ticks."""
    pytest.importorskip("numpy")
    sim = SimulationEngine(num_robots=3, tick_interval=0.4, auto_start=False)
    _, robot = next(iter(sim.robots.items()))
    # Park on the target and capture once.
    robot.pose.x, robot.pose.y = sim.target_position
    sim.step_once()
    assert robot.capture_score == 1
    first_score = robot.capture_score
    # Try to capture again immediately — cooldown/grace should block.
    robot.pose.x, robot.pose.y = sim.target_position
    sim.step_once()
    assert robot.capture_score == first_score, "cooldown failed to block double-capture"
    sim.shutdown()


def test_capture_win_condition_sets_winner_id() -> None:
    """Cross the win_score threshold and winner_id should be set; the
    event payload flags kind='win'."""
    pytest.importorskip("numpy")
    sim = SimulationEngine(num_robots=2, tick_interval=0.4, auto_start=False)
    # Crank the scoreboard manually (bypass cooldown by driving both robots).
    ids = list(sim.robots)
    win_target = sim.cfg.capture_win_score
    steps = 0
    while sim.winner_id is None and steps < 200:
        # Whichever robot is *not* the last capturer parks on the target.
        candidate = next(r for r in ids if r != sim._last_capturer)
        sim.robots[candidate].pose.x, sim.robots[candidate].pose.y = sim.target_position
        sim.step_once()
        steps += 1
    assert sim.winner_id is not None, f"nobody won after {steps} steps"
    assert sim.robots[sim.winner_id].capture_score >= win_target
    # The latest capture event must be flagged as a win.
    assert sim.capture_events[0]["kind"] == "win"
    assert sim.capture_events[0]["winner_id"] == sim.winner_id
    sim.shutdown()


def test_planner_solve_with_refs_accepts_per_step_trajectories() -> None:
    """The new public API should drop-in-replace solve() with a list of refs."""
    from fl_robots.mpc import DistributedMPCPlanner
    from fl_robots.sim_models import Pose2D, RobotState

    planner = DistributedMPCPlanner(horizon=5, dt=0.3, max_speed=0.3)
    robot = RobotState(
        robot_id="r0",
        pose=Pose2D(0.0, 0.0),
        velocity=(0.0, 0.0),
        formation_offset=(0.0, 0.0),
        goal=(0.0, 0.0),
    )
    # A ramp along +x — robot should track it progressively.
    refs = {"r0": [(0.2 * (k + 1), 0.0) for k in range(5)]}
    plan = planner.solve_with_refs([robot], refs)["r0"]
    assert len(plan.path) == 5
    # Monotonic x progress toward the final ref.
    xs = [p.x for p in plan.path]
    assert xs[-1] > xs[0]


def test_qp_planner_solve_with_refs_tracks_moving_ref() -> None:
    """QP planner should track a time-varying reference. We pick a ramp
    slow enough that slew + max_speed can follow, then check tracking
    error is modest."""
    from fl_robots.mpc_qp import QPMPCPlanner
    from fl_robots.sim_models import Pose2D, RobotState

    planner = QPMPCPlanner(horizon=8, dt=0.3, max_speed=0.4, slew_limit=0.4)
    robot = RobotState(
        robot_id="r0",
        pose=Pose2D(0.0, 0.0),
        velocity=(0.0, 0.0),
        formation_offset=(0.0, 0.0),
        goal=(0.0, 0.0),
    )
    # Ramp at 0.1 m per step ⇒ ~0.33 m/s — well inside max_speed=0.4.
    moving = {"r0": [(0.1 * (k + 1), 0.0) for k in range(8)]}
    plan_moving = planner.solve_with_refs([robot], moving)["r0"]
    assert plan_moving.tracking_error < 0.3
    # Sanity: the plan progresses monotonically along +x.
    xs = [p.x for p in plan_moving.path]
    assert xs[-1] > xs[0]
