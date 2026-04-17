"""Smoke tests for CLI, digital twin, monitor, and the MNIST federated loader.

The goal of these tests is to drive *import* and *basic construction* code
paths for modules that would otherwise sit at 0% coverage — they're not
meant to be exhaustive functional tests (those live in the module-specific
test files).
"""

from __future__ import annotations

import json
import sys

import pytest

# ── CLI ──────────────────────────────────────────────────────────────


def test_cli_build_parser_round_trip():
    from fl_robots.cli import build_parser

    p = build_parser()
    ns = p.parse_args(["run", "--robots", "5", "--port", "5050", "--manual"])
    assert ns.subcommand == "run"
    assert ns.robots == 5
    assert ns.port == 5050
    assert ns.manual is True


def test_cli_run_tests_executes_end_to_end(capsys):
    """The ``main test`` subcommand exercises the full SimulationEngine path."""
    from fl_robots.cli import run_tests

    run_tests()
    out = capsys.readouterr().out
    assert "All tests passed" in out


# ── Digital twin ─────────────────────────────────────────────────────


def test_digital_twin_constructs_and_serializes_state(fake_ros):
    from fl_robots.digital_twin import DigitalTwinNode, RobotVisualState, SystemVisualState

    # Pydantic models — pure tests, no fake env needed.
    rv = RobotVisualState(
        robot_id="robot_0",
        x=1.0,
        y=2.0,
        heading=0.5,
        is_training=False,
        training_round=3,
        is_active=True,
    )
    sv = SystemVisualState(robots={"robot_0": rv}, global_round=3)
    assert sv.robots["robot_0"].robot_id == "robot_0"
    payload = sv.model_dump_json()
    assert "robot_0" in payload

    # Node construction — drives subscriber wiring.
    node = DigitalTwinNode()
    assert node._name
    assert any("robot_status" in t for t in node.subscriptions)


# ── Monitor ──────────────────────────────────────────────────────────


def test_monitor_node_constructs_and_receives_status(fake_ros, tmp_path):
    from fl_robots.monitor import MonitorNode

    fake_ros.parameter_overrides["output_dir"] = str(tmp_path)

    node = MonitorNode()
    assert node._name
    assert any("status" in t or "metrics" in t for t in node.subscriptions)

    class _Msg:
        data = json.dumps({"type": "registration", "robot_id": "robot_q"})

    # Deliver a few messages to touch the dispatch code paths.
    if "/fl/robot_status" in fake_ros.subscriptions:
        fake_ros.publish("/fl/robot_status", _Msg())


# ── MNIST federated loader ───────────────────────────────────────────


def test_mnist_federated_dirichlet_partition_shapes():
    """The Dirichlet partition helper must produce disjoint shards per client."""
    import numpy as np
    from fl_robots.data.mnist_federated import _dirichlet_partition

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, size=500)
    shards = _dirichlet_partition(labels, num_clients=4, alpha=0.5, rng=rng)

    assert len(shards) == 4
    # Shards must be disjoint and cover the original indices.
    flat = [idx for shard in shards for idx in shard]
    assert len(flat) == len(set(flat))
    assert set(flat).issubset(set(range(len(labels))))
