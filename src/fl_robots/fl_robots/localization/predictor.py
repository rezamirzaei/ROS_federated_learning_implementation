"""Lightweight motion-model predictor for TOA target tracking.

A classic α-β tracker (degenerate Kalman filter for the constant-velocity
model) — enough to anchor the distributed TOA estimator between noisy
range updates without pulling SciPy in as a hard dependency.

State: ``(x, y, vx, vy)`` in the world frame. The predict step advances
by ``dt`` under constant velocity; the update step blends the current
state toward a new measurement with fixed gains ``α`` (position) and
``β`` (velocity). The tracker is intentionally stateless in between
``predict`` / ``update`` pairs so the TOA driver can call them in either
order on the same tick.

See: Brookner, *Tracking and Kalman Filtering Made Easy* (1998), §2 for
the derivation of α-β gains from steady-state Kalman responses.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ConstantVelocityTargetPredictor", "PredictorConfig"]


class PredictorConfig(BaseModel):
    """Validated config for :class:`ConstantVelocityTargetPredictor`."""

    model_config = ConfigDict(frozen=True)

    dt: float = Field(default=0.45, gt=0.0, le=5.0, description="Nominal step (seconds)")
    alpha: float = Field(
        default=0.55,
        gt=0.0,
        le=1.0,
        description="Position gain — higher trusts the measurement more.",
    )
    beta: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description="Velocity gain — higher lets the tracker react faster.",
    )


class ConstantVelocityTargetPredictor:
    """α-β constant-velocity tracker for a single 2-D target.

    Parameters
    ----------
    x, y:
        Initial target position guess. Defaults to the origin.
    config:
        Optional validated :class:`PredictorConfig`. When omitted, uses
        the α-β gains tuned for the simulation's ~0.45 s tick.
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        config: PredictorConfig | None = None,
    ) -> None:
        self.cfg = config or PredictorConfig()
        self._x = float(x)
        self._y = float(y)
        self._vx = 0.0
        self._vy = 0.0

    # ── Public API ───────────────────────────────────────────────────

    def predict(self, dt: float | None = None) -> tuple[float, float]:
        """Advance the internal state by ``dt`` (default: cfg.dt) and
        return the predicted ``(x, y)``.

        The predictor is a *pure* constant-velocity model — no process
        noise — so two successive ``predict`` calls without an
        intervening ``update`` just extrapolate along the current
        velocity. That's the intended behaviour: the caller is expected
        to run ``predict`` once per tick, followed by ``update`` with
        whatever measurement is available.
        """
        step = self.cfg.dt if dt is None else float(dt)
        self._x += self._vx * step
        self._y += self._vy * step
        return self._x, self._y

    def update(self, measurement: tuple[float, float], dt: float | None = None) -> None:
        """Blend the predicted state with a new position measurement.

        The α-β update is::

            r = measurement − predicted
            x ← x + α · r
            v ← v + (β / dt) · r

        with gains from ``cfg``. This is the steady-state form of the
        Kalman filter for a constant-velocity model with uncorrelated
        position measurement noise — fast, deterministic, and free of
        any matrix algebra.
        """
        step = self.cfg.dt if dt is None else float(dt)
        step = step if step > 1e-9 else 1e-9
        mx, my = float(measurement[0]), float(measurement[1])
        rx = mx - self._x
        ry = my - self._y
        self._x += self.cfg.alpha * rx
        self._y += self.cfg.alpha * ry
        self._vx += (self.cfg.beta / step) * rx
        self._vy += (self.cfg.beta / step) * ry

    def reset(self, x: float = 0.0, y: float = 0.0) -> None:
        """Re-initialise the tracker at ``(x, y)`` with zero velocity."""
        self._x = float(x)
        self._y = float(y)
        self._vx = 0.0
        self._vy = 0.0

    @property
    def state(self) -> tuple[float, float, float, float]:
        """Current ``(x, y, vx, vy)`` — exposed for tests and snapshots."""
        return self._x, self._y, self._vx, self._vy

