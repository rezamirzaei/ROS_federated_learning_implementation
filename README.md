# ROS2 Federated Learning Multi-Robot System

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](https://docker.com)

A comprehensive ROS2 project demonstrating **distributed federated learning** across multiple simulated robot agents. Each robot maintains a local ML model for navigation, participates in collaborative learning through federated averaging, and benefits from the collective intelligence of the swarm.

> **Note**: Training runs continuously until manually stopped. The robots will keep learning and improving their models indefinitely.

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
- **🧠 FedAvg Algorithm** — Industry-standard federated averaging implementation
- **🌐 MVC Web Dashboard** — Standalone Flask UI with templates, static assets, and controller routes
- **📨 Topic-Level Message Passing** — In-process ROS-style message bus for commands, telemetry, plans, and aggregation events
- **🎯 Distributed MPC Showcase** — Coordinated formation control with receding-horizon predictions and collision-aware motion planning
- **🌐 Interactive Digital Twin** — Browser-rendered formation view with predicted trajectories and leader motion
- **🐳 Docker Ready** — lightweight dashboard container by default, with the full ROS2 stack still available on demand
- **🧪 Comprehensive Testing** — Unit tests now cover the new message bus, MPC planner, simulation engine, and web routes

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
# Install runtime + dev dependencies with uv
uv sync --extra dev --extra viz

# Launch the full web dashboard with the built-in multi-agent simulation
uv run python main.py

# Optional flags
uv run python main.py run --robots 6 --port 5050
uv run python main.py run --manual
```

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

# Or run tests directly
uv run python -m pytest tests/ -v

# Run specific test
uv run python -m pytest tests/test_aggregation.py -v

# Quick component test (no Docker/ROS2)
uv run python main.py test
```

## 📈 Training Results

| Metric | Initial | After 20 Rounds | Continuous |
|--------|---------|-----------------|------------|
| **Loss** | ~1.29 | ~0.80 | Continues improving |
| **Accuracy** | ~40% | ~70% | Keeps increasing |
| **Divergence** | ~0.99 | ~0.58 | Models converging |

## 📚 Documentation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## 📖 References

- [FedAvg Paper (McMahan et al., 2017)](https://arxiv.org/abs/1610.05492)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
- [ROS2 Lifecycle Nodes](https://design.ros2.org/articles/node_lifecycle.html)

## 📝 License

MIT License
