# ROS2 Federated Learning Multi-Robot System

A comprehensive ROS2 project demonstrating distributed federated learning across multiple simulated robot agents. This project showcases proficiency in ROS2 concepts, distributed machine learning, and containerized deployment.

## 🎯 Project Overview

This system implements **Federated Averaging (FedAvg)** across multiple robot agents, where each robot:
- Trains a local navigation/obstacle avoidance model
- Shares model weights with a central aggregator
- Receives updated global model for improved performance

### Key Features
- **Distributed Learning**: Multiple robots learn collaboratively without sharing raw data
- **Non-IID Data Handling**: Each robot has a slightly different data distribution
- **ROS2 Best Practices**: Demonstrates topics, services, actions, parameters, and QoS
- **Docker Deployment**: Fully containerized for easy deployment on macOS Intel

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ROS2 DDS Network                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Robot 1    │    │   Robot 2    │    │   Robot 3    │     │
│  │              │    │              │    │              │     │
│  │ Local Model  │    │ Local Model  │    │ Local Model  │     │
│  │   Training   │    │   Training   │    │   Training   │     │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
│         │                   │                   │              │
│         │    /fl/robot_X/model_weights         │              │
│         └───────────────────┼───────────────────┘              │
│                             │                                  │
│                             ▼                                  │
│                    ┌────────────────┐                          │
│                    │   Aggregator   │                          │
│                    │                │                          │
│                    │   FedAvg       │                          │
│                    │   Algorithm    │                          │
│                    └────────┬───────┘                          │
│                             │                                  │
│                             │ /fl/global_model                 │
│                             │                                  │
│         ┌───────────────────┼───────────────────┐              │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Robot 1    │    │   Robot 2    │    │   Robot 3    │     │
│  │  (Updated)   │    │  (Updated)   │    │  (Updated)   │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                 │
│  ┌──────────────┐                       ┌──────────────┐       │
│  │  Coordinator │ ◄─────────────────────│   Monitor    │       │
│  │              │    Status/Metrics     │              │       │
│  │ Orchestrates │                       │ Visualizes   │       │
│  │  Training    │                       │  Metrics     │       │
│  └──────────────┘                       └──────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
ROS/
├── docker/
│   ├── Dockerfile              # ROS2 Humble + PyTorch image
│   ├── docker-compose.yaml     # Multi-container orchestration
│   └── ros_entrypoint.sh       # Container entrypoint
│
├── src/
│   └── fl_robots/
│       ├── fl_robots/
│       │   ├── __init__.py
│       │   ├── robot_agent.py      # Robot agent node
│       │   ├── aggregator.py       # FedAvg aggregator node
│       │   ├── coordinator.py      # Training orchestrator
│       │   ├── monitor.py          # Metrics visualization
│       │   └── models/
│       │       ├── __init__.py
│       │       └── simple_nn.py    # Neural network models
│       │
│       ├── launch/
│       │   └── fl_system.launch.py # System launch file
│       │
│       ├── config/
│       │   └── params.yaml         # Configuration parameters
│       │
│       ├── package.xml
│       ├── setup.py
│       └── setup.cfg
│
├── tests/
│   ├── test_aggregation.py         # Unit tests
│   └── test_ros_integration.py     # Integration tests
│
├── docs/
│   └── ARCHITECTURE.md             # This file
│
├── logs/                           # Runtime logs
├── models/                         # Saved models
└── results/                        # Training results
```

## 🚀 Quick Start

### Prerequisites
- Docker Desktop for Mac (Intel)
- At least 8GB RAM available for Docker

### Running the System

1. **Build and start all containers:**
   ```bash
   cd docker
   docker-compose up --build
   ```

2. **View logs from specific container:**
   ```bash
   docker-compose logs -f aggregator
   docker-compose logs -f robot_1
   docker-compose logs -f coordinator
   ```

3. **Stop the system:**
   ```bash
   docker-compose down
   ```

### Running Locally (Development)

1. **Build the ROS2 workspace:**
   ```bash
   cd /path/to/ROS
   source /opt/ros/humble/setup.bash
   colcon build --symlink-install
   source install/setup.bash
   ```

2. **Launch the complete system:**
   ```bash
   ros2 launch fl_robots fl_system.launch.py
   ```

3. **Launch with custom parameters:**
   ```bash
   ros2 launch fl_robots fl_system.launch.py total_rounds:=30 learning_rate:=0.0005
   ```

## 🔧 ROS2 Concepts Demonstrated

### Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/fl/robot_status` | `std_msgs/String` | Robot registration and status |
| `/fl/{robot_id}/model_weights` | `std_msgs/String` | Local model weights |
| `/fl/global_model` | `std_msgs/String` | Aggregated global model |
| `/fl/training_command` | `std_msgs/String` | Training commands |
| `/fl/aggregation_metrics` | `std_msgs/String` | Aggregation statistics |
| `/fl/coordinator_status` | `std_msgs/String` | Coordinator state |
| `/fl/{robot_id}/metrics` | `std_msgs/String` | Training metrics |

### Parameters
Each node has configurable parameters:

**Robot Agent:**
- `robot_id`: Unique identifier
- `learning_rate`: Local SGD learning rate
- `batch_size`: Training batch size
- `local_epochs`: Epochs per round
- `samples_per_round`: Training samples per round

**Aggregator:**
- `min_robots`: Minimum robots for aggregation
- `aggregation_timeout`: Timeout for weight collection
- `participation_threshold`: Required participation ratio

**Coordinator:**
- `total_rounds`: Total federated rounds
- `round_timeout`: Per-round timeout
- `evaluation_interval`: Evaluation frequency

### QoS Profiles
- **Reliable + Transient Local**: For critical messages (weights, commands)
- **Best Effort**: For high-frequency metrics

### Multi-threading
All nodes use `MultiThreadedExecutor` with callback groups for concurrent operation.

## 📊 Algorithm: Federated Averaging

The FedAvg algorithm (McMahan et al., 2017):

1. **Initialize** global model W₀
2. **For each round t = 1, 2, ..., T:**
   - Server sends W_t to all robots
   - Each robot k:
     - Initializes local model with W_t
     - Trains on local data for E epochs
     - Sends updated weights W_k^{t+1} to server
   - Server aggregates: W_{t+1} = Σ (n_k/n) × W_k^{t+1}

### Non-IID Data Handling
Each robot generates synthetic data with a robot-specific bias, simulating:
- Different sensor characteristics
- Different operating environments
- Non-identical data distributions

## 🧪 Testing

### Run Unit Tests
```bash
cd /ros2_ws
python -m pytest tests/test_aggregation.py -v
```

### Run Integration Tests
```bash
# Requires ROS2 environment
python -m pytest tests/test_ros_integration.py -v
```

### Manual Testing
```bash
# Terminal 1: Start aggregator
ros2 run fl_robots aggregator

# Terminal 2: Start robot
ros2 run fl_robots robot_agent --ros-args -p robot_id:=robot_0

# Terminal 3: Monitor topics
ros2 topic echo /fl/robot_status

# Terminal 4: Check node status
ros2 node list
ros2 topic list
ros2 param list /aggregator
```

## 📈 Results and Metrics

After training, results are saved to `/results/`:

- `aggregation_history.json`: Per-round aggregation metrics
- `aggregation_history.csv`: CSV format for analysis
- `robot_metrics.json`: Per-robot training history
- `training_summary.json`: Overall training summary

### Key Metrics
- **Gradient Divergence**: Measures how different local models are from global
- **Participation Rate**: Fraction of robots contributing per round
- **Local Loss/Accuracy**: Per-robot training performance

## 🔬 Extending the Project

### Adding More Robots
1. Add new service in `docker-compose.yaml`
2. Or modify `launch/fl_system.launch.py`

### Custom Model
1. Implement new model in `fl_robots/models/`
2. Ensure `get_weights()` and `set_weights()` methods
3. Update `robot_agent.py` to use new model

### Real Sensor Data
1. Subscribe to actual sensor topics (`/scan`, `/camera/image`)
2. Implement data preprocessing in `robot_agent.py`
3. Remove synthetic data generator

### Gazebo Simulation
1. Add Gazebo plugins to Dockerfile
2. Create world file and robot URDF
3. Connect nodes to simulated sensors

## 📚 References

- [Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/abs/1610.05492)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)

## 📝 License

MIT License - feel free to use for learning and projects.
