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

## MPC planners

Two interchangeable planners implement the same public API:

* `fl_robots.mpc.DistributedMPCPlanner` — discrete velocity-grid search; no
  external deps. Default and always available.
* `fl_robots.mpc_qp.QPMPCPlanner` — OSQP-backed QP with a double-integrator
  model. Available when `[qp]` extra is installed (`uv sync --extra qp`).

Cost terms: goal tracking, control effort, speed, soft pairwise collision
avoidance using neighbours' predicted first-step positions. Each robot plans
sequentially against the previous plan's predictions — the "distributed"
flavour.

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
