#!/usr/bin/env python3
"""Repository entrypoint for the standalone ROS showcase web app."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the ROS-inspired federated learning and distributed MPC dashboard."
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
        help="Start with autopilot disabled so the simulation only advances on explicit step commands.",
    )

    # Quick component test (no Docker / ROS2 required)
    sub.add_parser("test", help="Run quick component tests and exit")

    # Also support top-level flags for backward compat
    parser.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument("--port", default=5000, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--robots", default=4, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--manual", action="store_true", help=argparse.SUPPRESS)
    return parser


def run_tests() -> None:
    """Quick smoke-tests that validate models, FedAvg, and data generation."""
    import numpy as np
    import torch

    from ros_web.models import (
        SimpleNavigationNet,
        federated_averaging,
        compute_gradient_divergence,
    )

    print("=" * 55)
    print("  QUICK COMPONENT TEST  (no Docker / ROS2 required)")
    print("=" * 55)

    # 1. Model creation + forward pass
    model = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)
    x = torch.randn(8, 12)
    out = model(x)
    assert out.shape == (8, 4), f"Expected (8,4), got {out.shape}"
    print(f"  ✅ Model forward pass OK — params: {model.count_parameters()}")

    # 2. Weight serialisation round-trip
    w = model.get_weights()
    model2 = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)
    model2.set_weights(w)
    for k in w:
        np.testing.assert_array_almost_equal(w[k], model2.get_weights()[k])
    print("  ✅ Weight serialisation round-trip OK")

    # 3. FedAvg
    models = [SimpleNavigationNet(12, 32, 4) for _ in range(3)]
    wl = [m.get_weights() for m in models]
    avg = federated_averaging(wl, [100, 150, 200])
    global_m = SimpleNavigationNet(12, 32, 4)
    global_m.set_weights(avg)
    assert global_m(torch.randn(1, 12)).shape == (1, 4)
    print("  ✅ Federated Averaging OK")

    # 4. Gradient divergence
    divs = compute_gradient_divergence(wl, avg)
    assert all(d >= 0 for d in divs)
    print(f"  ✅ Gradient divergence OK — values: {[f'{d:.4f}' for d in divs]}")

    # 5. Simulation engine quick tick
    from ros_web.simulation import SimulationEngine
    sim = SimulationEngine(num_robots=2, auto_start=False)
    sim._step()
    assert sim.total_aggregations == 1
    print("  ✅ SimulationEngine single step OK")

    # 6. MPC planner
    from ros_web.mpc import DistributedMPCPlanner
    planner = DistributedMPCPlanner()
    planner.update_pose("r0", 1.0, 0.0, 0.0)
    lin, ang = planner.plan("r0", (0.0, 0.0))
    assert isinstance(lin, float)
    print(f"  ✅ MPC planner OK — cmd=({lin:.2f}, {ang:.2f})")

    print("-" * 55)
    print("  All tests passed ✅")
    print("=" * 55)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand == "test":
        run_tests()
        return

    # Default: run the web dashboard
    from ros_web.simulation import SimulationEngine
    from ros_web.web import create_app

    simulation = SimulationEngine(num_robots=args.robots, auto_start=True)
    if args.manual:
        simulation.issue_command("toggle_autopilot")

    app = create_app(simulation)
    try:
        print(f"\n🤖 Dashboard: http://{args.host}:{args.port}\n")
        app.run(host=args.host, port=args.port, debug=False)
    finally:
        simulation.shutdown()


if __name__ == "__main__":
    main()
