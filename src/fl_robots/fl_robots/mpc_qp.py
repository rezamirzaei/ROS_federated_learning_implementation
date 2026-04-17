"""QP-based distributed MPC planner.

Solves a small quadratic program per robot each tick::

    min_U   ½‖U‖²_R + ‖X - X_ref‖²_Q
    s.t.    X_{k+1} = X_k + dt · U_k        (double integrator kinematics)
            ‖U_k‖_∞ ≤ u_max
            (soft collision avoidance handled outside the QP via penalty terms)

We use OSQP — a sparse ADMM QP solver — which is ~50× faster and dramatically
better-conditioned than the discrete velocity-grid search in :mod:`fl_robots.mpc`.

If ``osqp`` / ``scipy`` aren't installed, :func:`get_qp_planner` raises a
friendly ``ImportError`` at construction time, and :class:`DistributedMPCPlanner`
from :mod:`fl_robots.mpc` remains the default planner.
"""
# noinspection PyTypeChecker
# Optional deps (numpy / osqp / scipy) are typed as ``Any`` below so PyCharm
# doesn't flag every ``_np.foo(...)`` call when they resolve to ``None`` in
# minimal installs. Runtime gating lives in ``get_qp_planner``.

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from .mpc import MPCConfig, MPCPlan, _safe_atan2
from .sim_models import (
    MPCRobotDiagnostic,
    MPCSystemDiagnostic,
    Pose2D,
    RobotState,
    TrajectoryPoint,
)

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = ["OSQP_AVAILABLE", "QPMPCPlanner", "get_qp_planner"]


ArrayLike = Any
WarmStartEntry = tuple[ArrayLike, ArrayLike]
SolverCacheEntry = dict[str, Any]


# ``Any`` fallbacks keep PyCharm / pyright happy: they see ``_np.eye(...)``
# and ``sparse.csc_matrix(...)`` as well-typed regardless of whether the
# optional deps resolved, and runtime gating lives in ``get_qp_planner``.
_np: Any
osqp: Any
sparse: Any

try:
    import numpy as _np  # type: ignore[no-redef]
    import osqp  # type: ignore[no-redef]
    from scipy import sparse  # type: ignore[no-redef]

    OSQP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _np = None
    osqp = None
    sparse = None
    OSQP_AVAILABLE = False


class QPMPCConfig(BaseModel):
    """Additional knobs specific to the QP planner."""

    model_config = ConfigDict(frozen=True)

    q_tracking: float = Field(default=10.0, gt=0.0, description="Tracking cost weight")
    r_control: float = Field(default=1.0, gt=0.0, description="Control effort weight")
    collision_radius: float = Field(default=0.55, gt=0.0)
    collision_penalty: float = Field(default=80.0, ge=0.0)
    slew_limit: float = Field(
        default=0.18,
        gt=0.0,
        description=(
            "Max step-to-step change ‖u_{k+1} − u_k‖_∞ (m/s per tick). Keeps "
            "the control signal smooth — without it OSQP routinely bangs "
            "against the box bounds every tick, which is what made earlier "
            "runs look stiff and synchronous."
        ),
    )
    terminal_weight: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "Extra tracking weight on the final horizon step. Ties the tail "
            "of the plan down so the receding-horizon shift doesn't drift."
        ),
    )
    neighbor_margin: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Extra slack (m) on top of ``safe_distance`` for the linearised "
            "keep-out half-planes added against each neighbour."
        ),
    )


class QPMPCPlanner:
    """OSQP-backed distributed MPC planner with the same public shape as
    :class:`fl_robots.mpc.DistributedMPCPlanner`.
    """

    def __init__(
        self,
        horizon: int = 8,
        dt: float = 0.35,
        max_speed: float = 0.32,
        q_tracking: float = 10.0,
        r_control: float = 1.0,
        slew_limit: float | None = None,
        terminal_weight: float | None = None,
    ) -> None:
        if not OSQP_AVAILABLE:
            raise ImportError(
                "osqp + scipy not installed. Install the qp extra: "
                "`uv sync --extra qp` or `pip install 'fl-robots[qp]'`."
            )
        cfg = MPCConfig(horizon=horizon, dt=dt, max_speed=max_speed)
        qcfg_kwargs: dict[str, float] = {"q_tracking": q_tracking, "r_control": r_control}
        if slew_limit is not None:
            qcfg_kwargs["slew_limit"] = slew_limit
        if terminal_weight is not None:
            qcfg_kwargs["terminal_weight"] = terminal_weight
        qcfg = QPMPCConfig(**qcfg_kwargs)
        self.horizon = cfg.horizon
        self.dt = cfg.dt
        self.max_speed = cfg.max_speed
        self.safe_distance = cfg.safe_distance
        self.q_tracking = qcfg.q_tracking
        self.r_control = qcfg.r_control
        self.collision_penalty = qcfg.collision_penalty
        self.slew_limit = qcfg.slew_limit
        self.terminal_weight = qcfg.terminal_weight
        self.neighbor_margin = qcfg.neighbor_margin
        # Cached OSQP workspaces. Reusing the solver instance preserves the KKT
        # factorisation, which is where OSQP's warm-start performance actually
        # comes from. We rebuild only when the sparsity pattern changes.
        self._solver_cache: dict[str, SolverCacheEntry] = {}
        # Per-robot warm-start cache: previous primal solution ``u`` and dual
        # ``y``. Re-seeding OSQP with these typically cuts iteration count by
        # 3–5× once the formation has settled.
        self._warm_cache: dict[str, WarmStartEntry] = {}
        # Iteration stats for observability and tests.
        self.last_iterations: dict[str, int] = {}
        self.last_status: dict[str, str] = {}
        self.last_solve_time_ms: dict[str, float] = {}
        self.last_control_effort: dict[str, float] = {}
        self.last_tracking_error: dict[str, float] = {}

    # Public API identical to the grid planner — drop-in replacement.
    def solve(
        self,
        robots: list[RobotState],
        leader_position: tuple[float, float],
    ) -> dict[str, MPCPlan]:
        """Solve with a single static reference point per robot. Backward
        compatible; delegates to :meth:`solve_with_refs` internally."""
        refs: dict[str, list[tuple[float, float]]] = {}
        for robot in robots:
            tgt = (
                leader_position[0] + robot.formation_offset[0],
                leader_position[1] + robot.formation_offset[1],
            )
            refs[robot.robot_id] = [tgt] * self.horizon
        return self.solve_with_refs(robots, refs)

    def solve_with_refs(
        self,
        robots: list[RobotState],
        references: dict[str, list[tuple[float, float]]],
    ) -> dict[str, MPCPlan]:
        """Solve with per-robot, per-step horizon-length references.

        Each robot's reference is a list of ``(x, y)`` tuples of length
        ``horizon``. Short sequences are padded with their last point;
        long ones are truncated. This lets callers express time-varying
        goals (e.g. an anticipated leader trajectory or a moving-target
        extrapolation from the α-β predictor) without touching the QP
        formulation.
        """
        plans: dict[str, MPCPlan] = {}
        predicted_neighbors: dict[str, list[TrajectoryPoint]] = {}
        for robot in robots:
            ref_seq = list(references.get(robot.robot_id, ()))
            if not ref_seq:
                ref_seq = [(robot.pose.x, robot.pose.y)]
            if len(ref_seq) < self.horizon:
                ref_seq = ref_seq + [ref_seq[-1]] * (self.horizon - len(ref_seq))
            elif len(ref_seq) > self.horizon:
                ref_seq = ref_seq[: self.horizon]

            t0 = time.perf_counter()
            plan = self._plan_robot(robot, ref_seq, predicted_neighbors, robots)
            self.last_solve_time_ms[robot.robot_id] = (time.perf_counter() - t0) * 1e3
            self.last_control_effort[robot.robot_id] = math.hypot(*plan.first_velocity)
            self.last_tracking_error[robot.robot_id] = plan.tracking_error
            plans[robot.robot_id] = plan
            predicted_neighbors[robot.robot_id] = plan.path
        return plans

    def diagnostics(
        self, tick: int, robots: list[RobotState]
    ) -> tuple[MPCSystemDiagnostic, list[MPCRobotDiagnostic]]:
        """Return last-solve diagnostics tagged with ``tick``."""
        per_robot: list[MPCRobotDiagnostic] = []
        for robot in robots:
            per_robot.append(
                MPCRobotDiagnostic(
                    tick=tick,
                    robot_id=robot.robot_id,
                    tracking_error=self.last_tracking_error.get(robot.robot_id, 0.0),
                    control_effort=self.last_control_effort.get(robot.robot_id, 0.0),
                    qp_iterations=int(self.last_iterations.get(robot.robot_id, 0)),
                    qp_solve_time_ms=self.last_solve_time_ms.get(robot.robot_id, 0.0),
                    qp_status=self.last_status.get(robot.robot_id, "unknown"),
                )
            )
        times = list(self.last_solve_time_ms.values()) or [0.0]
        # Per-robot QP has 2N decision vars (vx,vy stacked over the horizon).
        # Constraint rows: 2N box + 2N slew + 2N·(R−1) neighbour keep-outs.
        n_robots = max(len(robots), 1)
        per_robot_vars = 2 * self.horizon
        per_robot_cons = 2 * self.horizon * (1 + 1 + max(n_robots - 1, 0))
        system = MPCSystemDiagnostic(
            tick=tick,
            planner_kind="qp-osqp",
            n_robots=n_robots,
            horizon=self.horizon,
            nu=2,
            n_variables=per_robot_vars * n_robots,
            n_constraints=per_robot_cons * n_robots,
            mean_solve_time_ms=sum(times) / len(times),
        )
        return system, per_robot

    # ── Internals ────────────────────────────────────────────────────

    def _nominal_ego_positions(
        self, robot_id: str, x0: ArrayLike, v_prev: ArrayLike
    ) -> list[ArrayLike]:
        """Roll out a nominal ego trajectory for keep-out linearisation.

        Prefer the previous warm-started control sequence when available; fall
        back to a constant-velocity rollout from the current measured velocity.
        """
        cached = self._warm_cache.get(robot_id)
        if cached is not None and cached[0].shape == (2 * self.horizon,):
            u_nominal = _np.clip(cached[0], -self.max_speed, self.max_speed)
        else:
            u_nominal = _np.tile(v_prev, self.horizon)

        pos = x0.copy()
        rollout: list[ArrayLike] = []
        for k in range(self.horizon):
            pos = pos + self.dt * u_nominal[2 * k : 2 * k + 2]
            rollout.append(pos.copy())
        return rollout

    def _plan_robot(
        self,
        robot: RobotState,
        ref_seq: list[tuple[float, float]],
        predicted_neighbors: dict[str, list[TrajectoryPoint]],
        all_robots: list[RobotState],
    ) -> MPCPlan:
        """Build and solve a QP for a single robot.

        Decision variable: stacked velocities ``u = [vx₀, vy₀, …, vx_{N-1}, vy_{N-1}]``.
        Positions are reconstructed as ``x_k = x_0 + dt · Σ u_j`` so the QP has
        ``2N`` variables, matching the horizon.

        Cost
        ----
        * Stage tracking      — ``Q · Σ_{k<N-1} ‖p_k − p_ref_k‖²``
        * Terminal tracking   — ``(Q + Q_f) · ‖p_{N-1} − p_ref_{N-1}‖²``
        * Control effort      — ``R · Σ ‖u_k‖²``

        Constraints
        -----------
        * Velocity box        — ``‖u_k‖_∞ ≤ u_max``
        * Slew                — ``‖u_k − u_{k-1}‖_∞ ≤ du_max`` (with
          ``u_{-1} := robot.velocity``)
        * Neighbour keep-out  — one linearised half-plane per neighbour
          per horizon step: ``n̂_k^T · p_k ≥ n̂_k^T · p_nbr_k + safe``
        """
        N = self.horizon
        dt = self.dt
        x0 = _np.array([robot.pose.x, robot.pose.y], dtype=float)
        v_prev = _np.array(robot.velocity, dtype=float)
        # Stack the reference sequence into a flat (2N,) vector.
        tgt = _np.array(ref_seq[-1], dtype=float)  # last point — for tracking_error only
        p_ref = _np.asarray(ref_seq, dtype=float).reshape(-1)
        if p_ref.shape[0] != 2 * N:  # defensive; solve_with_refs already pads
            p_ref = _np.tile(tgt, N)

        # Build the cumulative-sum matrix S such that positions_k = x0 + dt · S_k · u
        #   where S_k is the k-th row of a lower-triangular block of 2×2 identities.
        # Shape of S: (2N, 2N). Positions: P = (1 ⊗ x0) + dt · S · u
        S = _np.zeros((2 * N, 2 * N))
        for k in range(N):
            for j in range(k + 1):
                S[2 * k, 2 * j] = 1.0
                S[2 * k + 1, 2 * j + 1] = 1.0

        # Reference comes from the per-step ``ref_seq`` assembled above;
        # ``x0_tile`` broadcasts the current position across the horizon.
        x0_tile = _np.tile(x0, N)

        # Stage weights: per-step Q, with a heavier terminal block on the
        # last position — ties the tail of the horizon down so the
        # warm-start shift doesn't leak error forward.
        w_stage = _np.ones(2 * N) * self.q_tracking
        w_stage[2 * (N - 1) : 2 * N] += self.terminal_weight
        W = _np.diag(w_stage)

        # Quadratic cost (½ uᵀ P u + qᵀ u):
        #   ½ (x0_tile + dt·S·u − p_ref)ᵀ W (x0_tile + dt·S·u − p_ref) + R‖u‖²
        # expands to P = 2·(dt² Sᵀ W S + R I), q = 2·dt Sᵀ W (x0_tile − p_ref).
        R = self.r_control
        P: Any = 2.0 * ((dt * dt) * (S.T @ W @ S) + R * _np.eye(2 * N))
        q: Any = 2.0 * dt * (S.T @ W) @ (x0_tile - p_ref)

        # ── Inequality constraints ──────────────────────────────────
        # Assemble all rows into a single (A, l, u) block.
        A_rows: list[Any] = []
        l_rows: list[float] = []
        u_rows: list[float] = []

        # (a) Velocity box on every step component.
        A_box = _np.eye(2 * N)
        A_rows.append(A_box)
        l_rows.extend([-self.max_speed] * (2 * N))
        u_rows.extend([self.max_speed] * (2 * N))

        # (b) Slew: |u_0 − v_prev|_∞ ≤ du_max and |u_k − u_{k-1}|_∞ ≤ du_max.
        du = self.slew_limit
        for i in range(2):
            row = _np.zeros(2 * N)
            row[i] = 1.0
            A_rows.append(row.reshape(1, -1))
            l_rows.append(-du + float(v_prev[i]))
            u_rows.append(du + float(v_prev[i]))
        for k in range(1, N):
            for i in range(2):
                row = _np.zeros(2 * N)
                row[2 * k + i] = 1.0
                row[2 * (k - 1) + i] = -1.0
                A_rows.append(row.reshape(1, -1))
                l_rows.append(-du)
                u_rows.append(du)

        # (c) Neighbour keep-out half-planes (convex linearisation).
        # For each neighbour, at each horizon step k, we ensure the
        # predicted ego position p_k = x0 + dt·S_k·u lies on the
        # half-plane n̂^T · p_k ≥ n̂^T · p_nbr_k + safe, where n̂ is the
        # unit vector from the neighbour to the ego at the *current*
        # time (good when robots are already separated). If the robots
        # have collapsed on top of each other we pick an arbitrary axis
        # so the QP stays feasible — the slew limit prevents it from
        # rushing into the chosen direction.
        safe = self.safe_distance + self.neighbor_margin
        nominal_ego_positions = self._nominal_ego_positions(robot.robot_id, x0, v_prev)
        for other in all_robots:
            if other.robot_id == robot.robot_id:
                continue
            pred: Any = predicted_neighbors.get(other.robot_id)
            for k in range(N):
                if pred and k < len(pred):
                    p_nbr = _np.array([pred[k].x, pred[k].y])
                else:
                    p_nbr = _np.array([other.pose.x, other.pose.y])
                offset_k = nominal_ego_positions[k] - p_nbr
                dist_k = float(_np.linalg.norm(offset_k))
                if dist_k < 1e-6:
                    n_hat = _np.array([1.0, 0.0])
                else:
                    n_hat = offset_k / dist_k
                # Row of S for step k picks out the first (k+1) velocity
                # blocks; extract it directly to keep the row sparse-ish.
                S_row = S[2 * k : 2 * k + 2, :]  # shape (2, 2N)
                # Constraint: n_hat^T (x0 + dt S_row u) ≥ n_hat^T p_nbr + safe
                # ⇒ (dt · n_hatᵀ · S_row) · u ≥ n_hatᵀ (p_nbr − x0) + safe
                row = dt * (n_hat @ S_row)  # shape (2N,)
                rhs = float(n_hat @ (p_nbr - x0)) + safe
                A_rows.append(row.reshape(1, -1))
                l_rows.append(rhs)
                u_rows.append(_np.inf)

        A = _np.vstack(A_rows)
        lb = _np.asarray(l_rows, dtype=float)
        ub = _np.asarray(u_rows, dtype=float)

        # OSQP requires sparse CSC matrices and triu for P.
        P_csc = sparse.triu(sparse.csc_matrix(P)).tocsc()
        A_csc = sparse.csc_matrix(A)

        prob: Any | None = None
        cache = self._solver_cache.get(robot.robot_id)
        cache_matches = bool(
            cache
            and cache.get("a_shape") == A_csc.shape
            and cache.get("a_nnz") == A_csc.nnz
            and cache.get("p_nnz") == P_csc.nnz
        )
        if cache_matches and cache is not None:
            solver = cache["solver"]
            prob = solver
            try:
                solver.update(Px=P_csc.data, Ax=A_csc.data, q=q, l=lb, u=ub)
            except Exception:  # pragma: no cover - defensive fallback
                cache_matches = False
                self._solver_cache.pop(robot.robot_id, None)

        if not cache_matches:
            assert osqp is not None
            solver = osqp.OSQP()
            solver.setup(
                P=P_csc,
                q=q,
                A=A_csc,
                l=lb,
                u=ub,
                verbose=False,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=400,
            )
            prob = solver
            self._solver_cache[robot.robot_id] = {
                "solver": solver,
                "a_shape": A_csc.shape,
                "a_nnz": A_csc.nnz,
                "p_nnz": P_csc.nnz,
            }

        # Warm-start from the previous tick's primal/dual if available.
        # OSQP's ADMM iterates converge dramatically faster near the previous
        # optimum when the formation is tracking steadily.
        cached = self._warm_cache.get(robot.robot_id)
        if cached is not None:
            prev_u, prev_y = cached
            if prob is not None and prev_u.shape == (2 * N,):
                prev_u = _np.clip(prev_u, -self.max_speed, self.max_speed)
                try:
                    if prev_y.shape == (A_csc.shape[0],):
                        prob.warm_start(x=prev_u, y=prev_y)
                    else:
                        prob.warm_start(x=prev_u)
                except Exception:  # pragma: no cover - defensive
                    pass

        assert prob is not None
        result = prob.solve()
        self.last_iterations[robot.robot_id] = int(getattr(result.info, "iter", 0))
        self.last_status[robot.robot_id] = str(getattr(result.info, "status", "unknown"))

        if result.info.status_val not in (1, 2):  # solved / solved-inaccurate
            # Fall back to "drive toward target" velocity, respecting
            # both the box and slew bound.
            to_tgt = tgt - x0
            norm = float(_np.linalg.norm(to_tgt)) or 1.0
            v_desired = self.max_speed * to_tgt / norm
            # Enforce slew: first step can only change v_prev by ±du.
            delta = v_desired - v_prev
            for i in range(2):
                if abs(delta[i]) > du:
                    delta[i] = du if delta[i] > 0 else -du
            v0 = v_prev + delta
            u = _np.tile(v0, N)
            self._warm_cache.pop(robot.robot_id, None)
        else:
            u = result.x
            # Shift-and-pad warm start: drop the first control, pad the
            # tail with the last value. Classic receding-horizon trick —
            # gives a better initial guess than the raw previous solution.
            shifted = _np.concatenate([u[2:], u[-2:]])
            self._warm_cache[robot.robot_id] = (
                shifted.copy(),
                _np.asarray(result.y, dtype=float).copy(),
            )

        # Reconstruct trajectory.
        path: list[TrajectoryPoint] = []
        pos = x0.copy()
        for k in range(N):
            pos = pos + dt * u[2 * k : 2 * k + 2]
            path.append(TrajectoryPoint(float(pos[0]), float(pos[1])))

        first_velocity = (float(u[0]), float(u[1]))
        tracking_error = math.dist((path[-1].x, path[-1].y), ref_seq[-1])
        total_cost = float(0.5 * u @ P @ u + q @ u)

        return MPCPlan(
            path=path,
            first_velocity=first_velocity,
            cost=total_cost / max(N, 1),
            tracking_error=tracking_error,
        )


def get_qp_planner(*args, **kwargs) -> QPMPCPlanner:
    """Factory that returns a :class:`QPMPCPlanner` or raises on missing deps."""
    return QPMPCPlanner(*args, **kwargs)


# Silence unused-import warnings when the optional deps are missing.
_ = Pose2D, _safe_atan2
