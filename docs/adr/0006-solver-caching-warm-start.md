# ADR-0006: OSQP Solver Caching and Warm-Start

**Status:** Accepted
**Date:** 2026-04-17

## Context

The QP-based MPC planner (`mpc_qp.py`) solves a small quadratic program per
robot per simulation tick. Naively creating a fresh OSQP solver each tick wastes
the KKT factorisation that dominates setup cost. Additionally, adjacent ticks
produce similar solutions, so warm-starting the ADMM iterates from the previous
solution can cut iteration counts by 3–5×.

## Decision

1. **Solver cache** — We maintain a `_solver_cache` dict keyed by `robot_id`.
   When the constraint-matrix sparsity pattern (shape + nnz) matches the
   previous tick, we call `solver.update(Px, Ax, q, l, u)` instead of
   rebuilding. If the pattern changes (e.g. a robot joins/leaves), we rebuild.

2. **Warm-start cache** — We store the previous primal (`u`) and dual (`y`)
   solution per robot. Before solving, we inject them via `solver.warm_start()`.
   Between ticks we apply a *shift-and-pad* receding-horizon initialisation.

3. **Parallel solves** — Per-robot QPs are solved in a `ThreadPoolExecutor`.
   OSQP releases the GIL during its C-level ADMM iterations.

## Consequences

- Typical solve time drops from ~2 ms to ~0.3 ms per robot after warm-up.
- The cache is invalidated safely on shape mismatch; fallback is a full rebuild.
- Thread-level parallelism scales to ~4 robots before Python overhead dominates.
