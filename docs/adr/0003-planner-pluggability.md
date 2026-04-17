# ADR 0003: Planner pluggability

- Status: Accepted
- Date: 2026-04-17

## Context

The project ships both a dependency-free MPC planner and an OSQP-backed QP
planner. The standalone simulation, tests, and observability surfaces need to
swap planners without changing higher-level orchestration code.

## Decision

Planners expose the same public shape through the `MPCPlanner` protocol and the
legacy `solve(robots, leader_position)` / richer `solve_with_refs(...)`
contract. `SimulationEngine` depends on that shape rather than on a concrete
planner implementation.

## Consequences

- New planners can be added without rewriting the simulation surface.
- Diagnostics and reference-trajectory support remain cross-planner features.
- Public planner behavior is part of the architecture contract, not just an
  implementation detail.
