# Observability

This document maps the main standalone metrics to their alerts and the runbook
sections that tell you what to do when those alerts fire.

## Signal map

| Metric / signal | Meaning | Alert(s) | Runbook |
|---|---|---|---|
| `up{job="fl-robots"}` and `/api/ready` | Readiness of the standalone dashboard | `FLRobotsReadinessDown`, `FLRobotsBudgetBurn` | [Readiness flapping](RUNBOOK.md#readiness-flapping) |
| `fl_aggregation_rounds_total` and `fl_training_active` | Completed FL rounds and whether training is currently active | `FLRobotsTrainingStalled` | [Aggregation stalled at round N](RUNBOOK.md#aggregation-stalled-at-round-n) |
| `fl_aggregation_divergence` | Client disagreement after aggregation | `FLRobotsAggregationDivergenceHigh` | [Aggregation stalled at round N](RUNBOOK.md#aggregation-stalled-at-round-n) |
| `fl_tracking_error_*` | Formation tracking quality | `FLRobotsTrackingErrorSpike` | [OSQP infeasibility or solve-time spike](RUNBOOK.md#osqp-infeasibility-or-solve-time-spike) |
| `fl_mpc_solve_time_ms_*` | MPC/OSQP per-robot solve latency | `FLRobotsQpSolveSlow` | [OSQP infeasibility or solve-time spike](RUNBOOK.md#osqp-infeasibility-or-solve-time-spike) |
| `fl_controller_state{state="ERROR"}` | Controller health state | `FLRobotsControllerError` | [Readiness flapping](RUNBOOK.md#readiness-flapping) |

## Key endpoints

- `/metrics` — Prometheus scrape endpoint.
- `/api/status` — full point-in-time snapshot for debugging.
- `/api/history/global` — round-level FL metrics.
- `/api/history/mpc` — per-robot MPC diagnostics.
- `/api/history/localization` — TOA localization history plus `enabled` flag.

## Dashboards and rules

- Import [`grafana-dashboard.json`](grafana-dashboard.json) into Grafana.
- Load [`prometheus-rules.yml`](prometheus-rules.yml) into Prometheus or the
  Prometheus Operator.

## Gaps to be aware of

- HTTP request RED metrics are not yet emitted, so API-level latency/error SLOs
  are inferred indirectly from readiness and application behavior.
- ROS mode reuses core metrics logic but does not yet expose the same full
  standalone endpoint surface.
