# Architecture

## Dual-mode design

The codebase deliberately runs in two modes that share one package:

```
+---------------------------------+         +---------------------------------+
|   Standalone (no ROS2)          |         |   ROS2 Humble                   |
|                                 |         |                                 |
|   fl_robots.simulation          |         |   fl_robots.robot_agent         |
|   fl_robots.message_bus         |<-same-->|   fl_robots.aggregator          |
|   fl_robots.controller          |  code   |   fl_robots.coordinator         |
|   fl_robots.standalone_web      |         |   fl_robots.web_dashboard       |
|                                 |         |                                 |
|   transport: in-process         |         |   transport: DDS + Actions      |
|   launcher : python main.py     |         |   launcher : ros2 launch ...    |
+---------------------------------+         +---------------------------------+
                        ^                                     ^
                        +----------- ros_compat.py -----------+
                           (shim that exports real rclpy types
                            when available, otherwise stubs)
```

All ROS node modules import their ROS dependencies **only** via
`fl_robots.ros_compat`, so the package imports cleanly in environments where
`rclpy` is not installed. The standalone path uses
`fl_robots.message_bus.MessageBus` — a thread-safe, in-process pub/sub bus
that mirrors ROS topic semantics.

## Components (ROS mode)

| Component       | Module                       | Kind            | Responsibilities |
|-----------------|------------------------------|-----------------|------------------|
| Robot Agent     | `fl_robots.robot_agent`      | Node + Action   | Local SGD rounds, publish model weights, serve `TrainRound` action |
| Aggregator      | `fl_robots.aggregator`       | Lifecycle Node  | FedAvg, global-model broadcast, services |
| Coordinator     | `fl_robots.coordinator`      | Node            | Round orchestration state machine |
| Monitor         | `fl_robots.monitor`          | Node            | Metrics persistence + pretty-printing |
| Digital Twin    | `fl_robots.digital_twin`     | Node            | Matplotlib view of formation + predictions |
| Web Dashboard   | `fl_robots.web_dashboard`    | Node + Flask    | Socket.IO real-time UI |

### Shared cross-cutting modules

| Module | Purpose |
|---|---|
| `fl_robots.controller`           | Canonical command vocabulary + Pydantic payload model. |
| `fl_robots.ros_compat`           | Re-exports or stubs `rclpy` / std msgs. |
| `fl_robots.message_bus`          | In-process ROS-topic mimic for standalone mode. |
| `fl_robots.observability.metrics`| Prometheus registry + snapshot-to-gauge bridge at `/metrics`. |
| `fl_robots.observability.logging`| JSON/structured logs via `structlog`. |
| `fl_robots.persistence`          | SQLite store for aggregation history and per-robot metrics. |
| `fl_robots.utils.retry`          | Pure-stdlib exponential-backoff decorator with jitter. |
| `fl_robots.data.mnist_federated` | Dirichlet non-IID MNIST shards for the benchmark script. |
| `fl_robots.mpc_qp`               | OSQP-based QP planner (opt-in via `--extra qp`). |

## FedAvg round sequence

```
 Robot 1        Robot 2        Robot 3       Aggregator           Coordinator
   |               |              |              |                     |
   |--TrainRound-> |              |              |                     |
   |               |--TrainRound->|              |                     |
   |               |              |--TrainRound->|                     |
   |  local SGD    |  local SGD   |  local SGD   |                     |
   |--weights-------------------------------->   |                     |
   |               |--weights----------------->  |                     |
   |               |              |--weights-->  |                     |
   |               |              |              | FedAvg              |
   |               |              |              |--AggregationResult->|
   |<--global_model-----------------------------|                      |
```

## Data flow (ROS topics, services, actions)

The runtime is deliberately split into small, single-purpose nodes that
communicate only over named DDS topics or declared services/actions. The
table below is the single source of truth — every publisher, subscriber,
service server and action server lives in one of the modules listed.

### Topics

| Topic                                  | Msg type               | QoS                           | Publisher(s)                              | Subscriber(s)                              | Purpose                                       |
|----------------------------------------|------------------------|-------------------------------|-------------------------------------------|--------------------------------------------|-----------------------------------------------|
| `/fl/{robot_id}/model_weights`         | `std_msgs/String` (JSON)| `RELIABLE`, `TRANSIENT_LOCAL` | `robot_agent`                             | `aggregator`                               | Post-SGD local weights for one round          |
| `/fl/{robot_id}/typed_status`          | `RobotStatus`          | `RELIABLE`, `VOLATILE`        | `robot_agent` (when interfaces available) | _reserved for a future typed subscriber_   | Strongly-typed heartbeat — migrating from JSON |
| `/fl/{robot_id}/typed_metrics`         | `TrainingMetrics`      | `BEST_EFFORT`, `VOLATILE`     | `robot_agent`                             | _reserved_                                 | Strongly-typed metrics — migrating from JSON  |
| `/fl/{robot_id}/metrics`               | `std_msgs/String` (JSON)| `BEST_EFFORT`, `VOLATILE`     | `robot_agent`                             | `monitor`, `web_dashboard`                 | Loss / accuracy / round counter               |
| `/fl/{robot_id}/mpc_plan`              | `std_msgs/String` (JSON)| `BEST_EFFORT`, `VOLATILE`     | standalone `simulation`                   | dashboards                                 | Per-tick MPC plan summary                     |
| `/fl/{robot_id}/telemetry`             | `std_msgs/String` (JSON)| `BEST_EFFORT`, `VOLATILE`     | standalone `simulation`                   | dashboards                                 | Pose + velocity                               |
| `/{robot_id}/cmd_vel`                  | `geometry_msgs/Twist`  | `RELIABLE`, `VOLATILE`        | controller / MPC driver                   | robot base                                 | Velocity command                              |
| `/fl/robot_status`                     | `std_msgs/String` (JSON)| `RELIABLE`, `VOLATILE`        | `robot_agent`, standalone `simulation`    | `coordinator`                              | Registration + lifecycle events               |
| `/fl/training_command`                 | `std_msgs/String` (JSON)| `RELIABLE`, `KEEP_LAST(1)`    | `coordinator`, `web_dashboard`            | `robot_agent`                              | start / stop / reset / step / disturbance     |
| `/fl/coordinator_status`               | `std_msgs/String` (JSON)| `RELIABLE`, `VOLATILE`        | `coordinator`, standalone `simulation`    | dashboards, monitor                        | Current FSM state, round id                   |
| `/fl/aggregation_metrics`              | `std_msgs/String` (JSON)| `RELIABLE`, `KEEP_LAST(20)`   | `aggregator`, standalone `simulation`     | `coordinator`, dashboards                  | Per-round FedAvg summary                      |
| `/fl/global_model`                     | `std_msgs/String` (JSON)| `RELIABLE`, `TRANSIENT_LOCAL` | `aggregator`                              | `robot_agent`                              | Broadcast of aggregated weights               |
| `/localization/toa`                    | `std_msgs/String` (JSON)| `BEST_EFFORT`, `VOLATILE`     | standalone `simulation`                   | dashboards                                 | Ground-truth + predicted target + RMSE        |
| `/localization/capture`                | `std_msgs/String` (JSON)| `RELIABLE`, `KEEP_LAST(10)`   | standalone `simulation`                   | dashboards                                 | Capture events: winner robot, score, new target |

Key conventions:

* **Per-robot topics** are namespaced under `/fl/{robot_id}/...`. The
  helpers in `fl_robots.ros_compat` centralise this so a typo can't
  split the pub/sub pair.
* **Payloads on `std_msgs/String` topics are JSON-serialised** Pydantic
  models from `fl_robots.sim_models`. The `Typed*` topics publish the
  same data using the custom interface definitions — the JSON-String
  path will be retired once every consumer has migrated.
* **QoS choices** are deliberate, not defaults:
  - weights/global_model → `TRANSIENT_LOCAL` so late-joiners pick up
    the last known value without waiting for the next round.
  - metrics / telemetry → `BEST_EFFORT` because a drop is always
    preferable to back-pressure on the training loop.
  - training commands → `KEEP_LAST(1)` **without** transient-local so a
    rebooting agent doesn't replay a stale `start_training` from last
    week.

### Services (defined in `fl_robots_interfaces/srv`)

| Service                             | Server         | Callers                  | Purpose                                          |
|-------------------------------------|----------------|--------------------------|--------------------------------------------------|
| `GetModelInfo`                      | `aggregator`   | dashboards, `robot_agent`| Fetch global round id + weight-vector hash       |
| `RegisterRobot`                     | `coordinator`  | `robot_agent` (startup)  | Announce capacity + claim a participant slot     |
| `TriggerAggregation`                | `aggregator`   | `coordinator`, web UI    | Manually force a FedAvg round                    |
| `UpdateHyperparameters`             | `aggregator`   | `coordinator`, web UI    | Change learning rate / FedProx μ / client split  |

### Actions (defined in `fl_robots_interfaces/action`)

| Action                              | Server         | Client                   | Purpose                                          |
|-------------------------------------|----------------|--------------------------|--------------------------------------------------|
| `TrainRound` @ `/fl/{robot_id}/train_round` | `robot_agent` | `coordinator`   | One async local-SGD round with streaming feedback |

### Standalone (`MessageBus`) parity

`fl_robots.message_bus.MessageBus` mimics ROS pub/sub with an in-process
thread-safe bus. Topic names and payload schemas are kept intentionally
identical to the ROS topics above so the same dashboards and tests run
in both modes — the bus is effectively a type-erased ROS executor that
publishes `BusEvent(topic, source, payload)` records into a bounded
ring buffer.

## MPC planners

Two interchangeable planners implement the same public API:

* `fl_robots.mpc.DistributedMPCPlanner` — discrete velocity-grid search; no
  external deps. Default and always available.
* `fl_robots.mpc_qp.QPMPCPlanner` — OSQP-backed QP with a double-integrator
  model. Available when `[qp]` extra is installed (`uv sync --extra qp`).

The QP formulation in detail:

* **Decision variable** — stacked velocities
  `u = [vx₀, vy₀, …, vx_{N-1}, vy_{N-1}]` over the horizon `N`.
* **Dynamics** — double integrator reconstructed analytically as
  `p_k = x₀ + dt · Σ_{j≤k} u_j` so the QP stays at 2N variables.
* **Cost** — stage tracking `Q‖p_k − p_ref_k‖²` + terminal bump
  `Q_f‖p_{N-1} − p_ref_{N-1}‖²` + control effort `R‖u‖²`.
* **Constraints** — (a) velocity box `‖u_k‖_∞ ≤ u_max`, (b) slew
  `‖u_k − u_{k-1}‖_∞ ≤ du_max` (with `u_{-1} := v_current`), (c) one
  linearised keep-out half-plane per neighbour per horizon step:
  `n̂^T · p_k ≥ n̂^T · p_nbr + safe_distance` — this is a *real* convex
  inequality, not the soft-penalty ridge used by the grid planner.
* **Warm-start** — previous solution is shift-and-padded so steady-state
  solves converge in 2–3 ADMM iterations.

Formation diversity is produced upstream of the planner: the simulation
rotates each robot's formation slot around the leader at
`cfg.formation_rotation_rate` and adds per-robot radial breathing, so
every robot ends up with a *distinct* reference trajectory. Without
this, a rigid leader-offset formation would make every robot's motion a
translated copy of the leader's.

### Capture ("hunt-the-target") mode

When `cfg.capture_enabled` is on (default) and the TOA estimator is
available, the planner reference for each robot is **its own
distributed TOA estimate** of the target — not a formation slot. The
true target stays fixed until one of the robots closes to within
`cfg.capture_radius`; at that moment the winner's `capture_score` is
incremented, a fresh target is spawned uniformly inside
`[-capture_bounds, capture_bounds]²` (and at least `1.5·capture_radius`
from every robot), and both the TOA estimator and the constant-velocity
predictor are reset around the new target. Every capture is also
published on the `/localization/capture` topic and appended to the
`capture_events` ring buffer for the dashboard.

## Observability pipeline

```
  Simulation snapshot
          |
          v
  observability.metrics.update_from_snapshot(snap)
          |
          v
  prometheus_client.REGISTRY  --/metrics-->  Prometheus  -->  Grafana
                                                      ^
                                              docs/grafana-dashboard.json
```

Import `docs/grafana-dashboard.json` into Grafana for a four-panel view
of controller state, robot count, loss/accuracy, and MPC tracking error.

## Persistence flow

`MetricsStore` writes to a single SQLite file in WAL mode. Any component
(aggregator, simulation, benchmark script) can call `record_round(...)` /
`record_event(...)`. Schema is a single inline `CREATE TABLE IF NOT EXISTS`
string — no ORM, no migrations.

## Security

* `FL_ROBOTS_API_TOKEN=<token>` enables bearer-token auth on mutating
  endpoints. When unset, the dashboard is open — intended for local demos.
  See `docs/SECURITY.md`.

## CI

See `.github/workflows/ci.yml`. Five jobs:

1. **lint** — ruff + ruff-format + mypy.
2. **standalone-tests** — pytest matrix on Python 3.10 / 3.11 / 3.12 with coverage.
3. **docker-standalone** — builds `standalone-runtime` then `standalone-test`
   and runs the suite inside.
4. **ros-build** — builds `ros-runtime` (colcon compiles `fl_robots` and
   `fl_robots_interfaces`) and does a post-source import smoke.
5. **benchmark** — runs a 3-round MNIST FedAvg smoke so regressions in FedAvg
   math surface as accuracy drops.

## Extension points

* **Add a command**: append to `COMMAND_NAMES` in `controller.py`; both the
  Flask routes and `_handle_command_event` pick it up.
* **Add a dataset**: create `src/fl_robots/fl_robots/data/<name>.py` that
  exposes `make_federated_shards(cfg)` returning `list[(X, y)]`, and swap it
  in `scripts/benchmark.py`.
* **Add a planner**: mirror
  `DistributedMPCPlanner.solve(robots, leader_position) -> dict[str, MPCPlan]`.
