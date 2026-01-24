# ROS2 Federated Learning Multi-Robot System

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](https://docker.com)

A comprehensive ROS2 project demonstrating **distributed federated learning** across multiple simulated robot agents. Each robot maintains a local ML model for navigation, participates in collaborative learning through federated averaging, and benefits from the collective intelligence of the swarm.

> **Note**: Training runs continuously until manually stopped. The robots will keep learning and improving their models indefinitely, demonstrating real-world federated learning scenarios.

## ✨ Features

- **🤖 Multi-Robot Federated Learning**: 3+ robot agents learning collaboratively
- **🧠 FedAvg Algorithm**: Industry-standard federated averaging implementation
- **📡 Full ROS2 Integration**: Topics, Services, Parameters, QoS profiles
- **🐳 Docker Ready**: One-command deployment for macOS Intel
- **📊 Real-time Monitoring**: Training metrics visualization
- **🌐 Web Dashboard**: Interactive browser-based control panel
- **🎮 Digital Twin**: Real-time 2D visualization of the robot network
- **🧪 Comprehensive Testing**: Unit and integration tests
- **🔄 Continuous Training**: System runs indefinitely for ongoing model improvement

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
./run.sh help         # Show all commands
```

### Using Docker Compose Directly

```bash
# Clone and navigate to project
cd ROS

# Build and start all containers
cd docker
docker-compose up --build

# View training progress
docker-compose logs -f monitor

# Stop the system
docker-compose down
```

### Local Development

```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Build workspace
cd ROS
colcon build --symlink-install
source install/setup.bash

# Launch complete system
ros2 launch fl_robots fl_system.launch.py
```

## 📁 Project Structure

```
ROS/
├── run.sh                  # Main run script
├── docker/                 # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yaml
├── src/fl_robots/         # ROS2 package
│   ├── fl_robots/         # Python nodes
│   │   ├── robot_agent.py # Robot learning agent
│   │   ├── aggregator.py  # FedAvg server
│   │   ├── coordinator.py # Training orchestrator
│   │   └── monitor.py     # Metrics dashboard
│   ├── launch/            # Launch files
│   └── config/            # Parameters
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## 🏗️ Architecture

```
     ┌─────────┐   ┌─────────┐   ┌─────────┐
     │ Robot 1 │   │ Robot 2 │   │ Robot 3 │
     │ (Agent) │   │ (Agent) │   │ (Agent) │
     └────┬────┘   └────┬────┘   └────┬────┘
          │             │             │
          │   Local Weights (Topics)  │
          └─────────────┼─────────────┘
                        ▼
               ┌────────────────┐
               │   Aggregator   │
               │   (FedAvg)     │
               └────────┬───────┘
                        │ Global Model
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     ┌─────────┐   ┌─────────┐   ┌─────────┐
     │ Robot 1 │   │ Robot 2 │   │ Robot 3 │
     │(Updated)│   │(Updated)│   │(Updated)│
     └─────────┘   └─────────┘   └─────────┘
          │             │             │
          └─────────────┼─────────────┘
                        │
                   ↻ Repeat
```

## 📊 ROS2 Topics

| Topic | Description |
|-------|-------------|
| `/fl/robot_status` | Robot registration & heartbeat |
| `/fl/{robot_id}/model_weights` | Local model weights |
| `/fl/global_model` | Aggregated global model |
| `/fl/training_command` | Training orchestration |
| `/fl/aggregation_metrics` | Performance metrics |
| `/fl/coordinator_status` | Coordinator state |

## ⚙️ Configuration

Key parameters in `config/params.yaml`:

```yaml
aggregator:
  min_robots: 2              # Minimum robots for aggregation
  aggregation_timeout: 30.0  # Timeout in seconds

robot_agent:
  learning_rate: 0.001       # Local learning rate
  local_epochs: 5            # Epochs per round
  samples_per_round: 256     # Training samples

coordinator:
  total_rounds: 20           # Initial training rounds (continues after)
```

## 🧪 Testing

```bash
# Run all tests using the run script
./run.sh test

# Or run tests directly
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_aggregation.py -v
```

## 📈 Training Results

The system continuously trains and improves. Typical results after initial rounds:

| Metric | Initial | After 20 Rounds | Continuous |
|--------|---------|-----------------|------------|
| **Loss** | ~1.29 | ~0.80 | Continues improving |
| **Accuracy** | ~40% | ~70% | Keeps increasing |
| **Divergence** | ~0.99 | ~0.58 | Models converging |

### Results Location
After training, find results in `/results/` (inside container) or mounted volume:
- `aggregation_history.csv` - Per-round metrics
- `robot_metrics.json` - Per-robot statistics
- `training_summary.json` - Overall summary

### View Live Dashboard
```bash
./run.sh dashboard
# or
./run.sh logs monitor
```

## 🌐 Web Dashboard & Digital Twin

Once the system is running, access the **Web Dashboard** at:

**http://localhost:5000**

### Features:
- **Real-time System Status**: Coordinator state, current round, aggregations
- **Training Metrics**: Average loss, accuracy, and divergence tracking
- **Robot Status Cards**: Individual robot metrics and training progress
- **Digital Twin Visualization**: 2D network topology showing robots and aggregator
- **Control Panel**: Start/stop training, refresh data, download results
- **Event Log**: Live stream of system events

### Screenshot:
The dashboard shows:
- 📊 System status panel with coordinator state
- 📈 Training metrics with progress bars
- 🤖 Individual robot cards with loss/accuracy
- 🌐 Digital twin visualization (updates every 5 seconds)
- 📋 Real-time event log

## 📚 Documentation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed documentation.

## 🔧 Extending

### Add More Robots
Modify `docker-compose.yaml` or launch file parameters.

### Custom Model
Implement in `fl_robots/models/` with `get_weights()`/`set_weights()` methods.

### Real Sensors
Connect to actual ROS2 sensor topics instead of synthetic data.

### Adjust Training Duration
Modify `coordinator.py` to change training behavior, or set `total_rounds` parameter.

## 📖 References

- [FedAvg Paper](https://arxiv.org/abs/1610.05492)
- [ROS2 Humble Docs](https://docs.ros.org/en/humble/)

## 📝 License

MIT License
