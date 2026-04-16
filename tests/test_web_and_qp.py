"""Tests for the auth layer, `/metrics` endpoint, and the QP MPC planner."""

from __future__ import annotations

import math

import pytest
from fl_robots.simulation import SimulationEngine
from fl_robots.standalone_web import create_app


@pytest.fixture
def app_with_token(monkeypatch):
    """Flask app with a bearer token set. Simulation paused for determinism."""
    monkeypatch.setenv("FL_ROBOTS_API_TOKEN", "secret-xyz")
    sim = SimulationEngine(num_robots=3, auto_start=False)
    app = create_app(sim)
    yield app, sim
    sim.shutdown()


@pytest.fixture
def app_no_token():
    sim = SimulationEngine(num_robots=3, auto_start=False)
    app = create_app(sim)
    yield app, sim
    sim.shutdown()


def test_command_requires_bearer_when_token_set(app_with_token):
    app, _ = app_with_token
    client = app.test_client()
    rv = client.post("/api/command", json={"command": "step"})
    assert rv.status_code == 401


def test_command_accepts_valid_bearer(app_with_token):
    app, _ = app_with_token
    client = app.test_client()
    rv = client.post(
        "/api/command",
        json={"command": "step"},
        headers={"Authorization": "Bearer secret-xyz"},
    )
    assert rv.status_code == 200
    assert rv.get_json()["ok"] is True


def test_command_rejects_wrong_bearer(app_with_token):
    app, _ = app_with_token
    client = app.test_client()
    rv = client.post(
        "/api/command",
        json={"command": "step"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert rv.status_code == 401


def test_command_without_token_env_is_open(app_no_token):
    app, _ = app_no_token
    client = app.test_client()
    rv = client.post("/api/command", json={"command": "step"})
    assert rv.status_code == 200


def test_metrics_endpoint_returns_prometheus_text(app_no_token):
    app, sim = app_no_token
    sim.step_once()  # generate one tick so gauges are non-zero
    client = app.test_client()
    rv = client.get("/metrics")
    assert rv.status_code == 200
    body = rv.data.decode("utf-8")
    # Prometheus exposition format — HELP + TYPE lines plus values.
    for metric in ("fl_robot_count", "fl_avg_accuracy", "fl_tick_count", "fl_controller_state"):
        assert metric in body, f"missing metric {metric}"


def test_metrics_content_type(app_no_token):
    app, _ = app_no_token
    client = app.test_client()
    rv = client.get("/metrics")
    assert rv.content_type.startswith("text/plain")


# ── QP MPC ──────────────────────────────────────────────────────────

osqp = pytest.importorskip("osqp")
scipy = pytest.importorskip("scipy")


def test_qp_planner_drives_single_robot_toward_target():
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


def test_qp_planner_preserves_separation_for_formation():
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


def test_qp_planner_shape_matches_grid_planner():
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


def test_qp_planner_respects_max_speed_box_constraint():
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


def test_qp_warm_start_does_not_regress_iterations():
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
    planner.solve([robot], leader_position=(0.0, 0.0))
    warm_iters = planner.last_iterations["r0"]
    assert warm_iters <= cold_iters, (
        f"warm start regressed: cold={cold_iters}, warm={warm_iters}"
    )


def test_qp_warm_cache_tolerates_stale_shapes():
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


def test_qp_planner_respects_slew_limit():
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


def test_rotating_formation_produces_diverse_motion():
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

