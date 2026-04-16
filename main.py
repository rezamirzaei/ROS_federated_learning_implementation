#!/usr/bin/env python3
"""Repository entrypoint for the standalone ROS showcase web app."""

from __future__ import annotations

import argparse

from ros_web.simulation import SimulationEngine
from ros_web.web import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the ROS-inspired federated learning and distributed MPC dashboard."
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host to bind")
    parser.add_argument("--port", default=5000, type=int, help="HTTP port to bind")
    parser.add_argument("--robots", default=4, type=int, help="Number of robot agents to simulate")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Start with autopilot disabled so the simulation only advances on explicit step commands.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    simulation = SimulationEngine(num_robots=args.robots, auto_start=True)
    if args.manual:
        simulation.issue_command("toggle_autopilot")

    app = create_app(simulation)
    try:
        app.run(host=args.host, port=args.port, debug=False)
    finally:
        simulation.shutdown()


if __name__ == "__main__":
    main()
