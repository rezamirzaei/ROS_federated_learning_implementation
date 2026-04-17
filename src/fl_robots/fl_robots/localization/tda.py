"""Distributed Time-of-Arrival (TOA) target localization.

Each sensor (robot) *i* observes a noisy TOA / range ``d_i = ‖p* − x_i‖ + v_i``
to a single target ``p*``. The goal is to have every sensor estimate ``p*``
collaboratively, using **only neighbour-to-neighbour consensus** (no fusion
centre).

Algorithm
---------
Following the distributed ADMM / dual-ascent scheme described in
*"Distributed Localization of Sensor Networks …"* (arXiv:2308.16737), each
sensor *i* maintains its own estimate ``p̂_i ∈ R²`` and a dual variable
``λ_{ij}`` for every neighbour. At every tick:

1. **Local primal step** — each sensor minimises a local cost combining
   its own range residual and a quadratic penalty on disagreement with
   neighbours::

       p̂_i ← p̂_i − η · ∇f_i(p̂_i)
       f_i(p) = ½ (‖p − x_i‖ − d_i)²
              + (ρ/2) Σ_{j∈N(i)} ‖p − p̂_j + λ_{ij}/ρ‖²

   We take ``max_inner_iters`` cheap gradient steps per tick — plenty for
   convergence between consecutive TOA samples, which arrive at most once
   per simulation tick.

2. **Dual ascent** — update multipliers to push neighbours toward
   agreement::

       λ_{ij} ← λ_{ij} + ρ · (p̂_i − p̂_j)

The gradient of the range term linearises around the current estimate::

    ∇ ½(‖p − x_i‖ − d_i)² = (‖p − x_i‖ − d_i) · (p − x_i) / ‖p − x_i‖

which is the classic Jacobi-style update used in range-based sensor
network localization. No matrix inverses, no solver dependency — pure NumPy.

Convergence properties (verified by ``tests/test_localization.py``):

* Static target, zero noise → mean RMSE → 0 as iterations → ∞.
* Noisy, moving target → RMSE stays bounded by O(σ / √k) after warm-up.
* Pairwise consensus gap ``max_ij ‖p̂_i − p̂_j‖`` decreases monotonically
  on a static target with a connected topology.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by optional-deps CI
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["DistributedTOAEstimator", "TOAConfig", "TOAUpdateResult"]


class TOAConfig(BaseModel):
    """Tunable knobs for the distributed TOA estimator."""

    model_config = ConfigDict(frozen=True)

    step_size: float = Field(default=0.18, gt=0.0, le=2.0, description="Primal gradient step η")
    rho: float = Field(default=0.6, gt=0.0, le=5.0, description="ADMM penalty weight ρ")
    max_inner_iters: int = Field(default=4, ge=1, le=50)
    init_radius: float = Field(
        default=1.5, ge=0.0, description="Initial estimate sampled in a disk of this radius"
    )
    prior_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=10.0,
        description=(
            "Weight μ on the predicted-target prior. When the caller supplies a "
            "motion-model prediction p̂_pred, the local cost gets an extra "
            "term ½ μ ‖p − p̂_pred‖², i.e. gradient μ·(p − p̂_pred). Setting 0 "
            "disables the prior and recovers the pure range/consensus ADMM."
        ),
    )


@dataclass
class TOAUpdateResult:
    """One tick of the distributed update, ready for snapshotting."""

    estimates: dict[str, tuple[float, float]]
    residuals: dict[str, float]
    errors: dict[str, float]
    mean_rmse: float
    consensus_gap: float
    #: How many inner gradient steps were taken this tick.
    inner_iters: int = field(default=0)


class DistributedTOAEstimator:
    """Jacobi ADMM / dual-ascent distributed TOA localizer.

    Each sensor keeps its own estimate; consensus is enforced through
    neighbour-wise dual variables that ride on the current topology. The
    estimator is oblivious to ground truth — the ``error`` field in
    :class:`TOAUpdateResult` is computed by the caller when it has access
    to the true target for scoring purposes.
    """

    def __init__(
        self,
        robot_ids: Sequence[str],
        config: TOAConfig | None = None,
        seed: int = 0,
    ) -> None:
        if not NUMPY_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "DistributedTOAEstimator requires NumPy. Install the 'ml' extra: "
                "`uv sync --extra ml`."
            )
        if not robot_ids:
            raise ValueError("robot_ids must be non-empty")
        self.cfg = config or TOAConfig()
        self._rng = np.random.default_rng(seed)
        # Random init inside a disk — never start with identical estimates
        # or the consensus gap is trivially zero.
        self._estimates: dict[str, np.ndarray] = {rid: self._random_init() for rid in robot_ids}
        #: Dual variables, keyed by the ordered pair (i, j) — one per directed
        #: neighbour edge. Allocated lazily as topology evolves.
        self._duals: dict[tuple[str, str], np.ndarray] = {}

    # ── Public API ───────────────────────────────────────────────────

    def estimate(self, robot_id: str) -> tuple[float, float]:
        """Current estimate for ``robot_id`` (x, y)."""
        arr = self._estimates[robot_id]
        return float(arr[0]), float(arr[1])

    def all_estimates(self) -> dict[str, tuple[float, float]]:
        return {rid: (float(v[0]), float(v[1])) for rid, v in self._estimates.items()}

    def update(
        self,
        sensor_positions: Mapping[str, tuple[float, float]],
        measurements: Mapping[str, float],
        neighbors: Mapping[str, Iterable[str]],
        ground_truth: tuple[float, float] | None = None,
        predicted_target: tuple[float, float] | None = None,
    ) -> TOAUpdateResult:
        """Run one tick of the distributed update.

        Parameters
        ----------
        sensor_positions
            Current position of each sensor (robot). Missing entries are
            interpreted as "sensor dropped out this tick" and skipped.
        measurements
            Noisy range ``d_i`` observed by each sensor.  Missing entries
            mean the sensor didn't contribute a measurement this tick
            (but may still participate in consensus).
        neighbors
            Adjacency map ``i → [j₁, j₂, …]``. Edges need not be symmetric
            — the estimator treats them as directed and the symmetric half
            is added implicitly via the dual updates.
        ground_truth
            Optional true target position for RMSE bookkeeping (the
            estimator itself never uses it).
        predicted_target
            Optional motion-model prediction of the target (e.g. from a
            constant-velocity α-β tracker). When provided, every sensor's
            local cost gets an extra ½ μ ‖p − p̂_pred‖² term — this keeps
            the consensus anchored between TOA updates even when range
            measurements are noisy or partially missing, and accelerates
            convergence when the target moves.
        """
        eta = self.cfg.step_size
        rho = self.cfg.rho
        mu = self.cfg.prior_weight if predicted_target is not None else 0.0
        pred_arr = (
            np.asarray(predicted_target, dtype=float) if predicted_target is not None else None
        )
        inner = self.cfg.max_inner_iters

        # Ensure every sensor in the current tick has an estimate.
        for rid in sensor_positions:
            if rid not in self._estimates:
                self._estimates[rid] = self._random_init()

        # Snapshot neighbours once per tick so inner iterations see a stable
        # graph (matches the paper's synchronous update model).
        nbrs: dict[str, tuple[str, ...]] = {
            rid: tuple(j for j in neighbors.get(rid, ()) if j in self._estimates)
            for rid in self._estimates
        }

        for _ in range(inner):
            # Copy current estimates so the inner step is parallel (Jacobi,
            # not Gauss-Seidel) — matches what a real distributed system
            # would get with one-hop message passing.
            snapshot = {rid: est.copy() for rid, est in self._estimates.items()}
            for rid, est in self._estimates.items():
                pos = sensor_positions.get(rid)
                if pos is None:
                    continue
                x_i = np.asarray(pos, dtype=float)
                grad = np.zeros(2, dtype=float)

                # Range term (only if we got a measurement this tick).
                d_i = measurements.get(rid)
                if d_i is not None:
                    diff = est - x_i
                    r = float(np.linalg.norm(diff))
                    if r > 1e-9:
                        grad += (r - float(d_i)) * diff / r

                # Consensus term.
                for j in nbrs[rid]:
                    lam = self._duals.setdefault((rid, j), np.zeros(2, dtype=float))
                    grad += rho * (est - snapshot[j] + lam / rho)

                # Motion-model prior — pulls the estimate toward the
                # predicted target supplied by the caller. Crucially this
                # is the *same* p̂_pred for every sensor, so it also
                # tightens consensus without needing any extra comms.
                if pred_arr is not None and mu > 0.0:
                    grad += mu * (est - pred_arr)

                self._estimates[rid] = est - eta * grad

        # Dual-ascent step after the primal block — classic ADMM ordering.
        for rid in self._estimates:
            for j in nbrs[rid]:
                key = (rid, j)
                lam = self._duals.setdefault(key, np.zeros(2, dtype=float))
                self._duals[key] = lam + rho * (self._estimates[rid] - self._estimates[j])

        return self._summarise(ground_truth, inner, measurements, sensor_positions)

    def reset(self) -> None:
        """Re-randomise estimates and clear dual variables."""
        for rid in list(self._estimates):
            self._estimates[rid] = self._random_init()
        self._duals.clear()

    # ── Internals ────────────────────────────────────────────────────

    def _random_init(self) -> np.ndarray:
        r = float(self._rng.uniform(0.0, self.cfg.init_radius))
        theta = float(self._rng.uniform(0.0, 2.0 * math.pi))
        return np.asarray([r * math.cos(theta), r * math.sin(theta)], dtype=float)

    def _summarise(
        self,
        ground_truth: tuple[float, float] | None,
        inner_iters: int,
        measurements: Mapping[str, float],
        sensor_positions: Mapping[str, tuple[float, float]],
    ) -> TOAUpdateResult:
        estimates: dict[str, tuple[float, float]] = {}
        errors: dict[str, float] = {}
        residuals: dict[str, float] = {}

        gt = None if ground_truth is None else np.asarray(ground_truth, dtype=float)
        errs_sq: list[float] = []
        for rid, est in self._estimates.items():
            estimates[rid] = (float(est[0]), float(est[1]))
            if gt is not None:
                err = float(np.linalg.norm(est - gt))
                errors[rid] = err
                errs_sq.append(err * err)
            else:
                errors[rid] = 0.0

            d_i = measurements.get(rid)
            pos = sensor_positions.get(rid)
            if d_i is not None and pos is not None:
                residuals[rid] = abs(float(np.linalg.norm(est - np.asarray(pos))) - float(d_i))
            else:
                residuals[rid] = 0.0

        mean_rmse = math.sqrt(sum(errs_sq) / len(errs_sq)) if errs_sq else 0.0
        consensus_gap = self._consensus_gap()
        return TOAUpdateResult(
            estimates=estimates,
            residuals=residuals,
            errors=errors,
            mean_rmse=mean_rmse,
            consensus_gap=consensus_gap,
            inner_iters=inner_iters,
        )

    def _consensus_gap(self) -> float:
        values = list(self._estimates.values())
        if len(values) < 2:
            return 0.0
        gap = 0.0
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                gap = max(gap, float(np.linalg.norm(values[i] - values[j])))
        return gap
