#!/usr/bin/env python3
"""Repository entrypoint for the standalone ROS showcase web app.

Usage::

    python main.py              # Start web dashboard (default)
    python main.py run          # Same as above
    python main.py run --robots 6 --manual
    python main.py test         # Quick component smoke-tests
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the fl_robots package is importable without pip install
_src_dir = str(Path(__file__).resolve().parent / "src" / "fl_robots")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the ROS-inspired federated learning and distributed MPC dashboard.",
    )
    sub = parser.add_subparsers(dest="subcommand")

    # Default / run mode
    run_p = sub.add_parser("run", help="Start the web dashboard (default)")
    run_p.add_argument("--host", default="127.0.0.1", help="HTTP host to bind")
    run_p.add_argument("--port", default=5000, type=int, help="HTTP port to bind")
    run_p.add_argument("--robots", default=4, type=int, help="Number of robot agents to simulate")
    run_p.add_argument(
        "--manual",
        action="store_true",
        help="Start with autopilot disabled.",
    )

    # Quick component test (no Docker / ROS2 required)
    sub.add_parser("test", help="Run quick component tests and exit")

    return parser


def run_tests() -> None:
    """Quick smoke-tests that validate simulation core, message bus, and MPC planner."""
    from fl_robots.simulation import SimulationEngine
    from fl_robots.mpc import DistributedMPCPlanner
    from fl_robots.message_bus import MessageBus
    from fl_robots.sim_models import Pose2D, RobotState, AggregationRecord, BusEvent

    print("=" * 55)
    print("  QUICK COMPONENT TEST  (no Docker / ROS2 required)")
    print("=" * 55)

    # 1. Model data classes
    pose = Pose2D(1.0, 2.0, 0.5)
    assert pose.as_dict() == {"x": 1.0, "y": 2.0, "heading": 0.5}
    print("  ✅ Pose2D OK")

    robot = RobotState(
        robot_id="r1", pose=pose, velocity=(0.1, 0.2),
        formation_offset=(1.0, 0.0), goal=(2.0, 0.0),
    )
    assert robot.as_dict()["robot_id"] == "r1"
    print("  ✅ RobotState OK")

    # 2. Message bus round-trip
    bus = MessageBus(max_events=10)
    received: list[BusEvent] = []
    bus.subscribe("/test", received.append)
    bus.publish("/test", "tester", {"key": "value"})
    assert len(received) == 1 and received[0].payload["key"] == "value"
    assert bus.unsubscribe("/test", received.append) is True
    print("  ✅ MessageBus pub/sub + unsubscribe OK")

    # 3. MPC planner
    planner = DistributedMPCPlanner()
    robots = [
        RobotState(robot_id="r0", pose=Pose2D(1.0, 0.0), velocity=(0.0, 0.0),
                   formation_offset=(1.0, 0.0), goal=(1.0, 0.0)),
        RobotState(robot_id="r1", pose=Pose2D(-1.0, 0.0), velocity=(0.0, 0.0),
                   formation_offset=(-1.0, 0.0), goal=(-1.0, 0.0)),
    ]
    plans = planner.solve(robots, leader_position=(0.0, 0.0))
    assert "r0" in plans and len(plans["r0"].path) == planner.horizon
    print(f"  ✅ MPC planner OK — horizon={planner.horizon}, cost={plans['r0'].cost:.3f}")

    # 4. Simulation engine
    sim = SimulationEngine(num_robots=3, auto_start=False)
    sim.step_once()
    assert sim.tick_count == 1
    snap = sim.snapshot()
    assert len(snap["robots"]) == 3
    print("  ✅ SimulationEngine single step + snapshot OK")

    # 5. Commands
    sim.issue_command("start_training")
    assert sim.training_active is True
    sim.issue_command("reset")
    assert sim.tick_count == 0
    print("  ✅ Command dispatch OK")

    # 6. Export
    export = sim.export_results()
    assert "exported_at" in export
    print("  ✅ Export results OK")

    # 7. Invalid command raises
    try:
        sim.issue_command("invalid_cmd")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ✅ Invalid command rejection OK")

    print("-" * 55)
    print("  All tests passed ✅")
    print("=" * 55)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand == "test":
        run_tests()
        return

    # Resolve defaults for 'run' subcommand or no subcommand
    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"
    port = getattr(args, "port", 5000) or 5000
    robots = getattr(args, "robots", 4) or 4
    manual = getattr(args, "manual", False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    from fl_robots.simulation import SimulationEngine
    from fl_robots.standalone_web import create_app

    simulation = SimulationEngine(num_robots=robots, auto_start=True)
    if manual:
        simulation.issue_command("toggle_autopilot")

    app = create_app(simulation)
    try:
        print(f"\n🤖 Dashboard: http://{host}:{port}\n")
        app.run(host=host, port=port, debug=False)
    finally:
        simulation.shutdown()


if __name__ == "__main__":
    main()
