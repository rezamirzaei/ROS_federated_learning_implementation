# ROS2 Federated Learning Multi-Robot System

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](.github/workflows/ci.yml)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.12-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Multi%E2%80%91stage-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

A dual-mode ROS2 / standalone project demonstrating **distributed federated learning** across simulated robot agents. Runs **without ROS2** as a pure-Python simulation, or as a full ROS2 Humble system with custom interfaces, lifecycle nodes, and actions.

* **Standalone** — `python main.py` spins up a Flask dashboard, in-process message bus, distributed MPC planner, and FedAvg rounds. No ROS dependency.
* **ROS2** — `./run.sh ros` launches the same system as a proper ROS2 workspace with per-agent action servers, lifecycle aggregator, and Socket.IO dashboard.
* **Benchmark** — `scripts/benchmark.py` runs reproducible MNIST FedAvg with Dirichlet non-IID splits; see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for real numbers.

Reproducible MNIST FedAvg headline (seed=42, 4 clients, 400 samples / client, CPU):

| α (Dirichlet) | Rounds | Final test accuracy |
|---:|---:|---:|
| 10.0 (≈ IID)        | 15 | **91.18 %** |
|  0.5 (moderate skew) | 10 | **88.78 %** |
|  0.1 (heavy skew)    | 15 | **83.46 %** |

## ✨ Features

### ROS2 Depth
- **📡 Topics** — Model weight broadcasting, status heartbeats, training commands
- **🔧 Services** — `RegisterRobot`, `TriggerAggregation`, `GetModelInfo`, `UpdateHyperparameters`
- **🎬 Actions** — `TrainRound` with real-time feedback and cancellation
- **♻️ Lifecycle Nodes** — Aggregator with configure/activate/deactivate/shutdown
- **📦 Custom Interfaces** — `fl_robots_interfaces` (msg/srv/action definitions)
- **⚙️ Dynamic Parameters** — Runtime reconfiguration of learning rate, batch size, epochs
- **📊 QoS Profiles** — Reliable + Transient Local for critical data, Best Effort for metrics
- **🔀 Callback Groups** — Reentrant + Mutually Exclusive for concurrent execution
- **🧵 Multi-threaded Executor** — 4-thread executors on all nodes

### Application
- **🤖 Multi-Robot Federated Learning** — 3+ robot agents learning collaboratively
- **🧠 FedAvg *and* FedProx** — baseline + Li et al. 2020 proximal-term variant for non-IID stability
- **🎲 Multi-seed benchmarks** — `--num-seeds N` reports mean ± std across seeds, not cherry-picked numbers
- **🌐 MVC Web Dashboard** — standalone Flask UI with templates, static assets, and controller routes
- **📨 Topic-Level Message Passing** — in-process ROS-style message bus for commands, telemetry, plans, and aggregation events
- **🎯 Distributed MPC (OSQP, warm-started)** — coordinated formation control with receding-horizon QP, ADMM warm starts between ticks, and verified ‖u‖∞ ≤ u_max box constraints
- **🌐 Interactive Digital Twin** — browser-rendered formation view with predicted trajectories and leader motion
- **🧪 Liveness + Readiness probes** — `/api/health` and `/api/ready` so Kubernetes/Compose only routes traffic to a stepping simulation
- **📜 OpenAPI 3.1 schema** — machine-readable API contract at `/api/openapi.json`
- **🚦 Per-IP rate limiting** — sliding-window limiter on mutating endpoints (configurable via env)
- **🛑 Graceful SIGTERM shutdown** — background simulation thread joins cleanly on Docker stop
- **🐳 Docker Ready** — lightweight dashboard container by default, with the full ROS2 stack still available on demand
- **🧪 Comprehensive Testing** — 86 tests, 66 % coverage; ROS nodes driven through a fake-rclpy harness so aggregator/coordinator/robot_agent reach real code paths without a ROS install

## 🚀 Quick Start

### Using the Run Script (Recommended)

```bash
# Navigate to project
cd ROS

# Start the lightweight Docker dashboard (fastest path)
./run.sh

# Start the original full ROS2 multi-container stack
./run.sh ros

# Or use specific commands:
./run.sh build lite   # Build lightweight dashboard image only
./run.sh build ros    # Build full ROS image only
./run.sh start lite   # Start lightweight dashboard
./run.sh start ros    # Start full ROS stack
./run.sh stop         # Stop all containers
./run.sh logs         # View all logs
./run.sh logs dashboard # View lightweight dashboard logs
./run.sh dashboard    # Show current metrics
./run.sh status       # Check container status
./run.sh test         # Run test suite
./run.sh clean        # Remove all containers/images
```

The default Docker path now launches the standalone FL + MPC dashboard in a single slim container at **http://localhost:5000**. Use `./run.sh ros` when you specifically want the original ROS2 multi-node deployment.

You can also start the default lightweight container directly from the repository root with:

```bash
docker compose up --build
```

### Standalone Dashboard (No Docker/ROS2)

```bash
# Install core + ML + dev deps with uv
#   core        → Flask dashboard & simulation only
#   --extra ml  → torch / numpy / scikit-learn (needed for aggregation tests)
#   --extra dev → pytest, linters
uv sync --extra ml --extra dev

# Launch the standalone Flask dashboard with the built-in multi-agent simulation
uv run python main.py

# Optional flags
uv run python main.py run --robots 6 --port 5050
uv run python main.py run --manual
```

> **Note**: The standalone dashboard uses plain Flask (no Socket.IO). The
> Socket.IO-based real-time dashboard (`flask-socketio`, `eventlet`) is
> provided by the ROS2 `web_dashboard` node and is installed via the
> `--extra ros` extra, which the `./run.sh ros` Docker profile uses.

### Web Dashboard

Once running, access the interactive dashboard at:

**http://localhost:5000**

Features:
- 📊 Real-time coordinator, robot, and aggregation status
- 🌐 Browser-based digital twin with predicted MPC horizons
- 📨 Live topic/message stream for commands, telemetry, and plans
- 🎮 Start/stop training, step the world, toggle autopilot, inject disturbances
- 📥 Download the current system snapshot as JSON

## 🏗️ Architecture

```
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │  Robot 1    │  │  Robot 2    │  │  Robot 3    │
     │ Action Srv  │  │ Action Srv  │  │ Action Srv  │
     │ Services    │  │ Services    │  │ Services    │
     └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
            │                │                │
            │  Topics + Services + Actions    │
            └────────────────┼────────────────┘
                             ▼
                    ┌────────────────┐
                    │   Aggregator   │
                    │ (Lifecycle)    │
                    │   FedAvg +     │
                    │   Services     │
                    └────────┬───────┘
                             │ Global Model
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │  Robot 1    │  │  Robot 2    │  │  Robot 3    │
     │ (Updated)   │  │ (Updated)   │  │ (Updated)   │
     └─────────────┘  └─────────────┘  └─────────────┘
                          ↻ Repeat
```

## 📦 Custom Interfaces

The `fl_robots_interfaces` package defines:

| Type | Name | Description |
|------|------|-------------|
| **msg** | `ModelWeights` | Serialized neural network weights |
| **msg** | `TrainingMetrics` | Real-time training progress |
| **msg** | `RobotStatus` | Robot heartbeat with status enum |
| **msg** | `AggregationResult` | FedAvg round result |
| **srv** | `RegisterRobot` | Register robot with federation |
| **srv** | `TriggerAggregation` | Force federated averaging |
| **srv** | `GetModelInfo` | Query model information |
| **srv** | `UpdateHyperparameters` | Dynamic hyperparameter tuning |
| **action** | `TrainRound` | Training with feedback & cancellation |

## 📊 ROS2 Topics

| Topic | Description | QoS |
|-------|-------------|-----|
| `/fl/robot_status` | Robot registration & heartbeat | Reliable |
| `/fl/{robot_id}/model_weights` | Local model weights | Reliable |
| `/fl/global_model` | Aggregated global model | Reliable |
| `/fl/training_command` | Training orchestration | Reliable |
| `/fl/aggregation_metrics` | Performance metrics | Reliable |
| `/fl/{robot_id}/typed_status` | Typed robot status | Reliable |
| `/fl/{robot_id}/typed_metrics` | Typed training metrics | Best Effort |

## ⚙️ Configuration

Key parameters in `config/params.yaml`:

```yaml
aggregator:
  min_robots: 2
  aggregation_timeout: 30.0
  auto_aggregate: true

robot_agent:
  learning_rate: 0.001
  local_epochs: 5
  samples_per_round: 256

coordinator:
  total_rounds: 20
  round_timeout: 60.0
```

## 🧪 Testing

```bash
# Run all tests using the run script
./run.sh test

# Or run tests directly (requires the ml + dev extras installed above)
uv run python -m pytest tests/ -v

# Run specific test
uv run python -m pytest tests/test_aggregation.py -v

# Quick component test (no Docker/ROS2)
uv run python main.py test
```

## 📈 Training Results

Real reproducible numbers from `scripts/benchmark.py` (seed=42, CPU-only) —
see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the full story and raw JSON.

| Experiment | Clients | α | Rounds | Final acc | Best acc | Wall |
|---|--:|--:|--:|--:|--:|--:|
| Near-IID | 4 | 10.0 | 15 | **91.18 %** | 91.76 % | 0.85 s |
| Moderate non-IID | 4 | 0.5 | 10 | 88.78 % | 88.78 % | 0.62 s |
| Heavy non-IID | 4 | 0.1 | 15 | 83.46 % | 85.42 % | 1.00 s |
| Moderate non-IID (8 clients) | 8 | 0.5 | 20 | **92.97 %** | 92.97 % | 2.60 s |

Reproduce:

```bash
# Single run
uv run python scripts/benchmark.py --rounds 15 --clients 4 --alpha 0.5

# FedProx — the canonical non-IID fix (Li et al., 2020)
uv run python scripts/benchmark.py --rounds 15 --alpha 0.1 \
    --algorithm fedprox --proximal-mu 0.01

# Multi-seed — reports mean ± std (the right way to publish FL numbers)
uv run python scripts/benchmark.py --rounds 15 --alpha 0.1 --num-seeds 5
```

## 📊 Observability

The standalone dashboard exposes a Prometheus scrape endpoint:

```bash
curl http://127.0.0.1:5000/metrics
```

Import [`docs/grafana-dashboard.json`](docs/grafana-dashboard.json) into
Grafana to get pre-built panels for loss/accuracy, controller state, round
latency, and MPC tracking error.

Structured JSON logs: `FL_ROBOTS_JSON_LOGS=1 uv run python main.py`.

## 🔒 Security

* **API auth** — Set `FL_ROBOTS_API_TOKEN=<token>` in the environment to
  require `Authorization: Bearer <token>` on mutating endpoints. Unset by
  default (local-demo mode).
* **Threat model & hardening checklist** — [`docs/SECURITY.md`](docs/SECURITY.md).
* **Secrets hygiene** — `gitleaks` runs in pre-commit; install with
  `uv run pre-commit install`.

## 📚 Documentation

| Document | What's in it |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)   | Dual-mode design, component map, sequence diagrams, extension points. |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)       | Reproducible FedAvg benchmarks on MNIST with Dirichlet non-IID splits. |
| [`docs/SECURITY.md`](docs/SECURITY.md)           | Threat model, hardening checklist, responsible disclosure. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md)             | How to develop, test, and extend the project. |
| [`docs/grafana-dashboard.json`](docs/grafana-dashboard.json) | Import-ready Grafana dashboard for the `/metrics` endpoint. |

## 📖 References

- [FedAvg Paper (McMahan et al., 2017)](https://arxiv.org/abs/1610.05492)
- [Hsu et al. 2019 — Measuring the effects of non-identical data distribution for federated visual classification](https://arxiv.org/abs/1909.06335) (Dirichlet split benchmark)
- [OSQP — Operator Splitting QP Solver](https://osqp.org/)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
- [ROS2 Lifecycle Nodes](https://design.ros2.org/articles/node_lifecycle.html)

## 📝 License

MIT License
