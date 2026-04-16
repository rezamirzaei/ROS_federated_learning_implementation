# ROS2 Federated Learning Multi-Robot System — Architecture

A comprehensive ROS2 project demonstrating distributed federated learning across multiple simulated robot agents. This project showcases deep proficiency in ROS2 concepts, distributed machine learning, and containerized deployment.

## 🎯 Project Overview

This system implements **Federated Averaging (FedAvg)** across multiple robot agents, where each robot:
- Trains a local navigation/obstacle avoidance model
- Shares model weights with a central aggregator
- Receives updated global model for improved performance

### ROS2 Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| **Topics** | Model weights, status heartbeats, training commands, metrics |
| **Services** | `RegisterRobot`, `TriggerAggregation`, `GetModelInfo`, `UpdateHyperparameters` |
| **Actions** | `TrainRound` with real-time feedback and cancellation support |
| **Lifecycle Nodes** | Aggregator: configure → activate → deactivate → cleanup → shutdown |
| **Custom Interfaces** | `fl_robots_interfaces` package (msg/srv/action definitions) |
| **Parameters** | Dynamic reconfiguration of learning rate, batch size, epochs |
| **QoS Profiles** | Reliable + Transient Local for critical data; Best Effort for metrics |
| **Callback Groups** | Reentrant (actions, services) + Mutually Exclusive (timers) |
| **Multi-threaded Executor** | All nodes use `MultiThreadedExecutor` with 4 threads |
| **Launch System** | Parameterized multi-node launch with staggered startup |
| **Web Dashboard** | Flask + Socket.IO MVC architecture with real-time WebSocket |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ROS2 DDS Network                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Custom Interfaces                            │   │
│  │  fl_robots_interfaces:                                           │   │
│  │    msg: ModelWeights, TrainingMetrics, RobotStatus, Aggregation  │   │
│  │    srv: RegisterRobot, TriggerAggregation, GetModelInfo,         │   │
│  │         UpdateHyperparameters                                    │   │
│  │    action: TrainRound (goal/feedback/result)                     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Robot 1    │    │   Robot 2    │    │   Robot 3    │              │
│  │              │    │              │    │              │              │
│  │ Action Srv   │    │ Action Srv   │    │ Action Srv   │              │
│  │ Services     │    │ Services     │    │ Services     │              │
│  │ Local Model  │    │ Local Model  │    │ Local Model  │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         │    Topics + Services + Actions        │                       │
│         └───────────────────┼───────────────────┘                       │
│                             │                                           │
│                             ▼                                           │
│                    ┌─────────────────┐                                  │
│                    │   Aggregator    │                                  │
│                    │ (Lifecycle Node)│                                  │
│                    │                 │                                  │
│                    │ FedAvg Server   │                                  │
│                    │ Services:       │                                  │
│                    │  RegisterRobot  │                                  │
│                    │  TriggerAgg     │                                  │
│                    │  GetModelInfo   │                                  │
│                    └────────┬────────┘                                  │
│                             │                                           │
│         ┌───────────────────┼───────────────────┐                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Robot 1    │    │   Robot 2    │    │   Robot 3    │              │
│  │  (Updated)   │    │  (Updated)   │    │  (Updated)   │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Coordinator │  │   Monitor    │  │ Digital Twin │  │    Web     │ │
│  │  State Mach  │  │  Metrics &   │  │ Matplotlib   │  │ Dashboard  │ │
│  │  Orchestrate │  │  Persistence │  │ Visualization│  │ MVC+WS    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
ROS/
├── main.py                     # Entry point with quick test
├── run.sh                      # Docker management script
├── pyproject.toml              # Python project config
├── docker/
│   ├── Dockerfile              # ROS2 Humble + PyTorch + Flask
│   ├── docker-compose.yaml     # 7-container orchestration with healthchecks
│   └── ros_entrypoint.sh       # Container entrypoint
├── src/
│   ├── fl_robots_interfaces/   # Custom ROS2 interfaces (CMake package)
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   ├── msg/                # ModelWeights, TrainingMetrics, RobotStatus, AggregationResult
│   │   ├── srv/                # RegisterRobot, TriggerAggregation, GetModelInfo, UpdateHyperparameters
│   │   └── action/             # TrainRound (goal/feedback/result)
│   └── fl_robots/              # Main ROS2 Python package
│       ├── fl_robots/
│       │   ├── robot_agent.py  # Action Server + Services + Parameters
│       │   ├── aggregator.py   # Lifecycle Node + FedAvg + Services
│       │   ├── coordinator.py  # State machine orchestration
│       │   ├── monitor.py      # Metrics collection & persistence
│       │   ├── digital_twin.py # Matplotlib visualization
│       │   ├── web_dashboard.py# MVC + WebSocket (Flask + Socket.IO)
│       │   ├── web/            # Web assets
│       │   │   ├── templates/  # Jinja2 HTML templates
│       │   │   └── static/     # CSS + JS (Chart.js, Socket.IO)
│       │   └── models/
│       │       └── simple_nn.py# Neural network + FedAvg + divergence
│       ├── launch/
│       │   └── fl_system.launch.py
│       └── config/
│           └── params.yaml
├── tests/
│   ├── test_aggregation.py     # Unit tests (model, FedAvg, divergence)
│   └── test_ros_integration.py # Integration tests (interfaces, pub/sub)
├── scripts/
│   ├── visualize.py            # Post-training visualization
│   └── run.sh                  # Alternative run script
├── docs/
│   └── ARCHITECTURE.md         # This file
├── results/                    # Training outputs
└── logs/                       # Runtime logs
```

## 🔧 Custom Interfaces (fl_robots_interfaces)

### Messages
| Message | Description |
|---------|-------------|
| `ModelWeights` | Serialized neural network weights with metadata |
| `TrainingMetrics` | Real-time training progress (epoch, loss, accuracy) |
| `RobotStatus` | Robot heartbeat with status enum (IDLE/TRAINING/UPLOADING/WAITING/ERROR) |
| `AggregationResult` | FedAvg round result with per-robot metrics |

### Services
| Service | Description |
|---------|-------------|
| `RegisterRobot` | Register a robot agent with the federation |
| `TriggerAggregation` | Manually trigger federated averaging (force option) |
| `GetModelInfo` | Query model information from robot or aggregator |
| `UpdateHyperparameters` | Dynamically update training hyperparameters per robot |

### Actions
| Action | Description |
|--------|-------------|
| `TrainRound` | Execute a training round with real-time feedback and cancellation |

## 📊 Topics

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `/fl/robot_status` | `String` | Reliable/TransientLocal | Robot registration & heartbeat |
| `/fl/{robot_id}/model_weights` | `String` | Reliable/TransientLocal | Local model weights |
| `/fl/global_model` | `String` | Reliable/TransientLocal | Aggregated global model |
| `/fl/training_command` | `String` | Reliable/TransientLocal | Training orchestration |
| `/fl/aggregation_metrics` | `String` | Reliable/TransientLocal | Performance metrics |
| `/fl/coordinator_status` | `String` | Reliable/TransientLocal | Coordinator state |
| `/fl/{robot_id}/metrics` | `String` | BestEffort | Training progress |
| `/fl/{robot_id}/typed_status` | `RobotStatus` | Reliable/TransientLocal | Typed status |
| `/fl/{robot_id}/typed_metrics` | `TrainingMetrics` | BestEffort | Typed metrics |

## 🌐 Web Dashboard (MVC Architecture)

### Architecture
- **Model**: ROS2 subscriber callbacks collect real-time state from all nodes
- **View**: Jinja2 HTML templates + Chart.js (loss/accuracy graphs) + Canvas (topology)
- **Controller**: Flask routes + Socket.IO for bidirectional WebSocket communication

### Features
- Real-time WebSocket push updates (1.5s interval)
- Interactive hyperparameter tuning (learning rate, batch size, epochs)
- Force aggregation trigger via ROS2 service client
- Live network topology canvas
- Loss/accuracy charts with Chart.js
- Per-robot status cards with progress bars
- Event log stream
- Results download (ZIP)

### Access
```
http://localhost:5000      # Direct
http://localhost:8080      # Docker port mapping
```

## 📊 Algorithm: Federated Averaging (FedAvg)

McMahan et al., 2017:

1. **Initialize** global model W₀
2. **For each round t:**
   - Server sends W_t to all robots
   - Each robot k trains on local data for E epochs
   - Robots publish updated weights W_k^{t+1}
   - Server aggregates: W_{t+1} = Σ (n_k/n) × W_k^{t+1}

### Non-IID Data
Each robot generates synthetic sensor data with robot-specific biases, simulating
different operating environments and sensor characteristics.

## 🧪 Testing

```bash
# Unit tests (no ROS2 required)
python -m pytest tests/test_aggregation.py -v

# Integration tests (requires ROS2 + custom interfaces)
python -m pytest tests/test_ros_integration.py -v

# Quick component test
python main.py test

# Full test suite via Docker
./run.sh test
```

## 📚 References

- [FedAvg Paper (McMahan et al., 2017)](https://arxiv.org/abs/1610.05492)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
- [ROS2 Lifecycle Nodes](https://design.ros2.org/articles/node_lifecycle.html)
- [ROS2 Actions](https://docs.ros.org/en/humble/Tutorials/Intermediate/Writing-an-Action-Server-Client/Py.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)

## 📝 License

MIT License
