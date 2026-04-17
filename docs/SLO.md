# Service-Level Objectives

These SLOs apply to the **standalone** Flask dashboard (`standalone_web.py`)
served behind the `/api/health` and `/api/ready` probes. They are tracked via
the Prometheus metrics exposed on `/metrics` and alerted on by the rules in
[`docs/prometheus-rules.yml`](prometheus-rules.yml).

| # | Indicator | Target | Measurement window |
|---|---|---|---|
| 1 | **Readiness availability** — fraction of time `/api/ready` returns 200 | ≥ 99.5 % | 30 d rolling |
| 2 | **Round latency p95** — wall time between `start_training` and the first aggregation event | ≤ 2.5 s | 24 h rolling |
| 3 | **MPC solve time p99** — per-robot OSQP solve time | ≤ 25 ms | 5 min rolling |
| 4 | **Training progress** — rounds completed while `training_active` is true | ≥ 1 / min | 10 min rolling |
| 5 | **API error rate** — non-2xx response ratio on `/api/*` | ≤ 0.5 % | 1 h rolling |
| 6 | **Aggregation divergence** — gradient divergence metric | ≤ 0.5 | per round |

## Error budgets

- SLO 1 allows **≈ 3.6 h of unreadiness per 30 d**. Burning more than 10 % of
  the budget in a single hour triggers the `FLRobotsBudgetBurn` alert (see
  Prometheus rules).
- SLO 4 has no strict error budget but a 5-minute stall fires
  `FLRobotsTrainingStalled`.

## What we do not promise

- **ROS 2 mode** SLOs are governed separately by the `web_dashboard.py`
  service and DDS transport reliability; they are out of scope for this
  document.
- **Benchmark reproducibility** — seeds are controlled in
  `scripts/benchmark.py`, but wheel-level non-determinism
  (torch/cuDNN/BLAS) can shift accuracies by ±0.5 pp.

## Review cadence

SLOs are reviewed quarterly. Adjustments land via a PR to this file and the
Prometheus rules file.

