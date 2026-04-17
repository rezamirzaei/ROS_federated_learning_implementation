"""Packaged CLI entrypoint for the standalone FL + MPC dashboard."""

from __future__ import annotations

import argparse
import logging
import sys

from fl_robots.message_bus import MessageBus
from fl_robots.mpc import DistributedMPCPlanner
from fl_robots.sim_models import BusEvent, Pose2D, RobotState
from fl_robots.simulation import SimulationEngine
from fl_robots.standalone_web import create_app

__all__ = ["build_parser", "main", "run_tests"]

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the ROS-inspired federated learning and distributed MPC dashboard.",
    )
    sub = parser.add_subparsers(dest="subcommand")

    run_p = sub.add_parser("run", help="Start the web dashboard (default)")
    run_p.add_argument("--host", default="127.0.0.1", help="HTTP host to bind")
    run_p.add_argument("--port", default=5000, type=int, help="HTTP port to bind")
    run_p.add_argument("--robots", default=4, type=int, help="Number of robot agents to simulate")
    run_p.add_argument(
        "--manual",
        action="store_true",
        help="Start with autopilot disabled.",
    )

    sub.add_parser("test", help="Run quick component tests and exit")
    return parser


def _check(condition: bool, msg: str) -> None:
    """Raise RuntimeError if *condition* is False."""
    if not condition:
        raise RuntimeError(msg)


def run_tests() -> None:
    """Quick smoke-tests that validate simulation core, message bus, and MPC planner."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    log.info("=" * 55)
    log.info("  QUICK COMPONENT TEST  (no Docker / ROS2 required)")
    log.info("=" * 55)

    pose = Pose2D(1.0, 2.0, 0.5)
    _check(pose.as_dict() == {"x": 1.0, "y": 2.0, "heading": 0.5}, "Pose2D.as_dict mismatch")
    log.info("  ✅ Pose2D OK")

    robot = RobotState(
        robot_id="r1",
        pose=pose,
        velocity=(0.1, 0.2),
        formation_offset=(1.0, 0.0),
        goal=(2.0, 0.0),
    )
    _check(robot.as_dict()["robot_id"] == "r1", "RobotState.robot_id mismatch")
    log.info("  ✅ RobotState OK")

    bus = MessageBus(max_events=10)
    received: list[BusEvent] = []
    bus.subscribe("/test", received.append)
    bus.publish("/test", "tester", {"key": "value"})
    _check(len(received) == 1 and received[0].payload["key"] == "value", "MessageBus pub failed")
    _check(bus.unsubscribe("/test", received.append) is True, "MessageBus unsub failed")
    log.info("  ✅ MessageBus pub/sub + unsubscribe OK")

    planner = DistributedMPCPlanner()
    robots = [
        RobotState(
            robot_id="r0",
            pose=Pose2D(1.0, 0.0),
            velocity=(0.0, 0.0),
            formation_offset=(1.0, 0.0),
            goal=(1.0, 0.0),
        ),
        RobotState(
            robot_id="r1",
            pose=Pose2D(-1.0, 0.0),
            velocity=(0.0, 0.0),
            formation_offset=(-1.0, 0.0),
            goal=(-1.0, 0.0),
        ),
    ]
    plans = planner.solve(robots, leader_position=(0.0, 0.0))
    _check("r0" in plans and len(plans["r0"].path) == planner.horizon, "MPC solve failed")
    log.info("  ✅ MPC planner OK — horizon=%d, cost=%.3f", planner.horizon, plans["r0"].cost)

    sim = SimulationEngine(num_robots=3, auto_start=False)
    sim.step_once()
    _check(sim.tick_count == 1, "tick_count != 1 after step_once")
    snap = sim.snapshot()
    _check(len(snap["robots"]) == 3, "snapshot robot count mismatch")
    log.info("  ✅ SimulationEngine single step + snapshot OK")

    sim.issue_command("start_training")
    _check(sim.training_active is True, "training_active not True")
    sim.issue_command("reset")
    _check(sim.tick_count == 0, "tick_count != 0 after reset")
    log.info("  ✅ Command dispatch OK")

    export = sim.export_results()
    _check("exported_at" in export, "export missing exported_at")
    log.info("  ✅ Export results OK")

    try:
        sim.issue_command("invalid_cmd")
        msg = "Should have raised ValueError"
        raise RuntimeError(msg)
    except ValueError:
        pass
    log.info("  ✅ Invalid command rejection OK")

    log.info("-" * 55)
    log.info("  All tests passed ✅")
    log.info("=" * 55)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand == "test":
        run_tests()
        return

    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"
    port = getattr(args, "port", 5000) or 5000
    robots = getattr(args, "robots", 4) or 4
    manual = getattr(args, "manual", False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    simulation = SimulationEngine(num_robots=robots, auto_start=True)
    if manual:
        simulation.issue_command("toggle_autopilot")

    app = create_app(simulation)
    try:
        log.info("🤖 Dashboard: http://%s:%d", host, port)
        app.run(host=host, port=port, debug=False)
    finally:
        simulation.shutdown()
