import math

from fl_robots.message_bus import MessageBus
from fl_robots.mpc import DistributedMPCPlanner
from fl_robots.sim_models import Pose2D, RobotState
from fl_robots.simulation import SimulationEngine
from fl_robots.standalone_web import create_app

# ── MessageBus ──────────────────────────────────────────────────────


def test_message_bus_records_and_dispatches_events():
    bus = MessageBus()
    received = []

    bus.subscribe("/demo/topic", received.append)
    event = bus.publish("/demo/topic", "robot_1", {"value": 7})

    assert event.topic == "/demo/topic"
    assert received[0].payload["value"] == 7
    assert bus.recent_events(limit=1)[0].source == "robot_1"


def test_message_bus_unsubscribe():
    bus = MessageBus()
    received = []
    bus.subscribe("/t", received.append)
    bus.publish("/t", "s", {"a": 1})
    assert len(received) == 1
    assert bus.unsubscribe("/t", received.append) is True
    bus.publish("/t", "s", {"a": 2})
    assert len(received) == 1  # no new events


def test_message_bus_wildcard_subscription():
    bus = MessageBus()
    received = []
    bus.subscribe("*", received.append)
    bus.publish("/a", "s", {})
    bus.publish("/b", "s", {})
    assert len(received) == 2


def test_message_bus_handler_exception_does_not_crash():
    bus = MessageBus()

    def bad_handler(event):
        raise RuntimeError("boom")

    bus.subscribe("/t", bad_handler)
    # Should not raise
    event = bus.publish("/t", "s", {"x": 1})
    assert event.topic == "/t"


def test_message_bus_subscriber_count():
    bus = MessageBus()
    assert bus.subscriber_count == 0
    bus.subscribe("/a", lambda e: None)
    bus.subscribe("/b", lambda e: None)
    assert bus.subscriber_count == 2


# ── MPC ─────────────────────────────────────────────────────────────


def test_distributed_mpc_produces_horizon_and_safe_first_step():
    planner = DistributedMPCPlanner(horizon=6)
    robots = [
        RobotState(
            robot_id="robot_1",
            pose=Pose2D(-1.0, 0.0),
            velocity=(0.0, 0.0),
            formation_offset=(-1.0, 0.0),
            goal=(-1.0, 0.0),
        ),
        RobotState(
            robot_id="robot_2",
            pose=Pose2D(1.0, 0.0),
            velocity=(0.0, 0.0),
            formation_offset=(1.0, 0.0),
            goal=(1.0, 0.0),
        ),
        RobotState(
            robot_id="robot_3",
            pose=Pose2D(0.0, 1.2),
            velocity=(0.0, 0.0),
            formation_offset=(0.0, 1.2),
            goal=(0.0, 1.2),
        ),
    ]

    plans = planner.solve(robots, leader_position=(0.4, 0.2))

    assert all(len(plan.path) == 6 for plan in plans.values())

    first_positions = [(plan.path[0].x, plan.path[0].y) for plan in plans.values()]
    min_separation = min(
        math.dist(first_positions[i], first_positions[j])
        for i in range(len(first_positions))
        for j in range(i + 1, len(first_positions))
    )
    assert min_separation > 0.3


def test_mpc_single_robot():
    planner = DistributedMPCPlanner(horizon=4)
    robots = [
        RobotState(
            robot_id="solo",
            pose=Pose2D(0.0, 0.0),
            velocity=(0.0, 0.0),
            formation_offset=(1.0, 0.0),
            goal=(1.0, 0.0),
        ),
    ]
    plans = planner.solve(robots, leader_position=(0.0, 0.0))
    assert len(plans["solo"].path) == 4
    assert plans["solo"].tracking_error >= 0.0


# ── Simulation ──────────────────────────────────────────────────────


def test_simulation_aggregates_after_five_training_steps():
    simulation = SimulationEngine(num_robots=4, auto_start=False)
    try:
        simulation.issue_command("start_training")
        for _ in range(5):
            simulation.step_once()

        snapshot = simulation.snapshot()
        assert snapshot["system"]["current_round"] == 1
        assert snapshot["metrics"]["last_aggregation"]["participants"] == 4
        assert any(
            message["topic"] == "/fl/aggregation_metrics" for message in snapshot["messages"]
        )
    finally:
        simulation.shutdown()


def test_simulation_reset_clears_state():
    sim = SimulationEngine(num_robots=2, auto_start=False)
    try:
        sim.issue_command("start_training")
        for _ in range(10):
            sim.step_once()
        assert sim.tick_count == 10

        sim.issue_command("reset")
        assert sim.tick_count == 0
        assert sim.current_round == 0
        assert sim.training_active is False
        assert len(sim.robots) == 2
    finally:
        sim.shutdown()


def test_simulation_disturbance_moves_robots():
    sim = SimulationEngine(num_robots=2, auto_start=False)
    try:
        positions_before = {rid: (r.pose.x, r.pose.y) for rid, r in sim.robots.items()}
        sim.issue_command("disturbance")
        positions_after = {rid: (r.pose.x, r.pose.y) for rid, r in sim.robots.items()}
        assert any(positions_before[rid] != positions_after[rid] for rid in positions_before)
        assert sim.controller_state == "RECOVERING"
    finally:
        sim.shutdown()


def test_simulation_export_has_timestamp():
    sim = SimulationEngine(num_robots=2, auto_start=False)
    try:
        export = sim.export_results()
        assert "exported_at" in export
        assert isinstance(export["exported_at"], float)
    finally:
        sim.shutdown()


# ── Web routes ──────────────────────────────────────────────────────


def test_web_routes_expose_status_and_validate_commands():
    simulation = SimulationEngine(num_robots=3, auto_start=False)
    app = create_app(simulation)
    client = app.test_client()

    try:
        status_response = client.get("/api/status")
        assert status_response.status_code == 200
        assert status_response.get_json()["system"]["robot_count"] == 3

        command_response = client.post("/api/command", json={"command": "step"})
        assert command_response.status_code == 200
        assert command_response.get_json()["ok"] is True

        invalid_response = client.post("/api/command", json={"command": "unknown"})
        assert invalid_response.status_code == 400
    finally:
        simulation.shutdown()


def test_standalone_template_wires_capture_panel_ids():
    """The capture panel in the dashboard relies on specific DOM IDs that
    ``standalone.js`` looks up at boot — if any go missing the UI silently
    breaks. Lock them in."""
    simulation = SimulationEngine(num_robots=2, auto_start=False)
    app = create_app(simulation)
    try:
        client = app.test_client()
        rv = client.get("/")
        assert rv.status_code == 200
        html = rv.data.decode("utf-8")
        required_ids = [
            'id="capture-hint"',
            'id="capture-total"',
            'id="capture-target-pos"',
            'id="capture-radius"',
            'id="capture-win-score"',
            'id="capture-winner-banner"',
            'id="capture-winner-text"',
            'id="scoreboard"',
            'id="capture-events"',
        ]
        for marker in required_ids:
            assert marker in html, f"template missing {marker}"
        # Panel + CSS class hooks
        assert 'class="panel capture-panel"' in html
    finally:
        simulation.shutdown()


def test_web_health_endpoint():
    simulation = SimulationEngine(num_robots=2, auto_start=False)
    app = create_app(simulation)
    client = app.test_client()

    try:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True
        assert "uptime_ticks" in body
    finally:
        simulation.shutdown()


def test_web_results_endpoint():
    simulation = SimulationEngine(num_robots=2, auto_start=False)
    app = create_app(simulation)
    client = app.test_client()

    try:
        resp = client.get("/api/results")
        assert resp.status_code == 200
        assert "attachment" in resp.headers.get("Content-Disposition", "")
    finally:
        simulation.shutdown()


def test_web_missing_command_returns_400():
    simulation = SimulationEngine(num_robots=2, auto_start=False)
    app = create_app(simulation)
    client = app.test_client()

    try:
        resp = client.post("/api/command", json={})
        assert resp.status_code == 400
        assert resp.get_json()["ok"] is False
    finally:
        simulation.shutdown()


def test_web_index_returns_html():
    simulation = SimulationEngine(num_robots=2, auto_start=False)
    app = create_app(simulation)
    client = app.test_client()

    try:
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Digital Twin" in resp.data
    finally:
        simulation.shutdown()
