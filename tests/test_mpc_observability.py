"""Tests for MPC observability (diagnostics + snapshot integration)."""

from __future__ import annotations

import pytest

from fl_robots.simulation import SimulationEngine


def _run_sim(ticks: int = 6) -> SimulationEngine:
    sim = SimulationEngine(num_robots=3, tick_interval=0.5, auto_start=False)
    for _ in range(ticks):
        sim.step_once()
    return sim


def test_grid_planner_reports_diagnostics() -> None:
    sim = _run_sim(3)
    system, per_robot = sim.planner.diagnostics(42, list(sim.robots.values()))
    assert system.tick == 42
    assert system.planner_kind == "grid-search"
    assert system.n_robots == 3
    assert system.horizon >= 1
    assert system.n_variables >= 1
    assert len(per_robot) == 3
    for d in per_robot:
        assert d.qp_status == "grid-search"
        assert d.qp_solve_time_ms >= 0
        assert d.tracking_error >= 0
        assert d.control_effort >= 0


def test_snapshot_contains_mpc_section() -> None:
    sim = _run_sim(3)
    snap = sim.snapshot()
    assert "mpc" in snap
    mpc = snap["mpc"]
    assert mpc["system"] is not None
    assert mpc["system"]["planner_kind"] == "grid-search"
    assert isinstance(mpc["per_robot"], list) and len(mpc["per_robot"]) == 3
    assert isinstance(mpc["history"], list) and len(mpc["history"]) >= 3


def test_mpc_history_appends_per_tick() -> None:
    sim = _run_sim(8)
    # One record per robot per tick → 8 * 3 = 24.
    assert len(sim.mpc_robot_history) >= 24
    ticks_present = {d.tick for d in sim.mpc_robot_history}
    assert len(ticks_present) >= 8


def test_qp_planner_diagnostics() -> None:
    osqp = pytest.importorskip("osqp")
    scipy = pytest.importorskip("scipy")
    from fl_robots.mpc_qp import QPMPCPlanner

    sim = SimulationEngine(num_robots=3, tick_interval=0.5, auto_start=False)
    sim.planner = QPMPCPlanner(horizon=6)
    for _ in range(3):
        sim.step_once()

    system, per_robot = sim.planner.diagnostics(99, list(sim.robots.values()))
    assert system.planner_kind == "qp-osqp"
    assert system.n_variables == 2 * 6 * 3  # 2N per robot * 3 robots
    # Constraints per robot: 2N box + 2N slew + 2N·(n_robots−1) keep-outs.
    expected_cons_per_robot = 2 * 6 * (1 + 1 + (3 - 1))
    assert system.n_constraints == expected_cons_per_robot * 3
    for d in per_robot:
        # OSQP reports a status string and (usually) ≥ 0 iterations.
        assert d.qp_status
        assert d.qp_iterations >= 0
