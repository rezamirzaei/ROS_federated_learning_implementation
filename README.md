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
- **🌐 MVC Web Dashboard** — Flask + Socket.IO with real-time WebSocket updates
- **📈 Interactive Charts** — Chart.js loss/accuracy graphs, Canvas topology visualization
- **🎮 Interactive Controls** — Start/stop training, force aggregation, tune hyperparameters
- **🔬 Digital Twin** — Matplotlib network visualization
- **🐳 Docker Ready** — 7-container orchestration with healthchecks
- **🧪 Comprehensive Testing** — Unit tests + integration tests covering custom interfaces

## 🚀 Quick Start

### Using the Run Script (Recommended)

```bash
# Navigate to project
cd ROS

# Run the complete system (builds if needed)
./run.sh

# Or use specific commands:
./run.sh build        # Build Docker image only
./run.sh start        # Start containers
./run.sh stop         # Stop all containers
./run.sh logs         # View all logs
./run.sh logs monitor # View training dashboard
./run.sh dashboard    # Show current metrics
./run.sh status       # Check container status
./run.sh test         # Run test suite
./run.sh clean        # Remove all containers/images
```

### Quick Component Test (No Docker/ROS2)

```bash
# Test model, FedAvg, data generation, MPC, simulation engine
python main.py test
```

### Standalone Dashboard (No Docker/ROS2)

```bash
# Launch the full web dashboard with in-process simulation
python main.py
# or
python main.py run --robots 4 --port 5000
```

### Web Dashboard

Once running, access the interactive dashboard at:

**http://localhost:5000** (or http://localhost:8080 via Docker)

Features:
- 📊 Real-time system status with WebSocket updates
- 📈 Loss/accuracy charts (Chart.js)
- 🌐 Live network topology (Canvas)
- 🎮 Start/stop training, force aggregation
- ⚙️ Interactive hyperparameter tuning
- 📥 Download results as ZIP

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
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_aggregation.py -v

# Quick component test (no Docker/ROS2)
python main.py test
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
