# Runbook

This runbook covers the common operator-facing failure modes for the standalone
dashboard and the shared FL/MPC logic that feeds both standalone and ROS views.

## Aggregation stalled at round N

**Symptoms**

- `FLRobotsTrainingStalled` fires.
- `/api/status` shows `training_active=true` but `current_round` is flat.
- `/api/history/global` stops growing.

**Checks**

1. Call `/api/status` and confirm the controller is still `RUNNING`.
2. Check `/api/history/robots/<robot_id>` for recent local loss/accuracy points.
3. Inspect logs for stale-round drops or missing robot registrations.

**Likely causes**

- Too few active participants for the aggregator threshold.
- A robot is stuck before publishing local weights.
- Rate-limited/manual operator commands paused or reset the simulation.

**Actions**

1. Trigger a controlled reset if the system is wedged: `POST /api/command {"command":"reset"}`.
2. In ROS mode, inspect `/fl/robot_status` and `/fl/{robot_id}/model_weights`.
3. Verify the aggregator is still publishing `/fl/global_model` after the next round.

## Readiness flapping

**Symptoms**

- `FLRobotsReadinessDown` or `FLRobotsBudgetBurn` fires.
- `/api/ready` alternates between 200 and 503.

**Checks**

1. Read `/api/ready` and compare `last_tick_age_s` with `stale_threshold_s`.
2. Check whether autopilot is enabled in `/api/status`.
3. Look for repeated restarts or blocked command handlers in logs.

**Likely causes**

- Simulation thread not stepping.
- CPU starvation or long blocking work on the request path.
- Container probe thresholds tighter than the runtime can sustain.

**Actions**

1. Confirm the process is alive with `/api/health`.
2. If manual mode was enabled intentionally, either toggle autopilot back on or
   adjust probe expectations for that environment.
3. Restart the workload if the background simulation thread is hung.

## Memory creep

**Symptoms**

- Resident memory rises steadily over a long run.
- Dashboard history becomes sluggish.

**Checks**

1. Confirm history ring buffers are bounded in `/api/status`.
2. Inspect result artifacts for unexpected growth.
3. Review whether optional features (TOA/capture/FedProx) are enabled.

**Likely causes**

- Unexpected accumulation outside the bounded histories.
- Repeated result export or artifact retention in a mounted volume.
- Long-running debug sessions with additional tooling attached.

**Actions**

1. Verify `summary.json` / `aggregation_history.json` are rotated or archived by
   the deployment, not appended indefinitely.
2. Capture a heap/profile snapshot before restart if the trend is reproducible.
3. Restart the process to restore the bounded in-memory baseline.

## OSQP infeasibility or solve-time spike

**Symptoms**

- `FLRobotsQpSolveSlow` or `FLRobotsTrackingErrorSpike` fires.
- `/api/history/mpc` shows rising solve time or degraded tracking.

**Checks**

1. Inspect `/api/history/mpc` for per-robot `qp_status` and solve-time trends.
2. Check if the system is in recovery after a disturbance.
3. Compare robot count and horizon against the current SLO budget in `docs/SLO.md`.

**Likely causes**

- Constraint density increased after a disturbance or crowded formation.
- OSQP fell back to a slower path or required more iterations.
- The grid-search fallback is active because optional QP deps are unavailable.

**Actions**

1. Confirm whether the QP planner is actually installed in the target runtime.
2. Reduce robot count or horizon for the affected deployment profile.
3. Reset the simulation if infeasibility is persistent after the disturbance clears.
