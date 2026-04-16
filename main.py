#!/usr/bin/env python3
"""
ROS2 Federated Learning Multi-Robot System — Entry Point

This script provides a convenient way to launch the FL system or run
quick tests without Docker. For full deployment, use:

    ./run.sh              # Docker-based deployment
    ros2 launch fl_robots fl_system.launch.py   # ROS2 launch

ROS2 Concepts Demonstrated in this project:
- Topics (pub/sub for model weights, status, commands)
- Services (RegisterRobot, TriggerAggregation, GetModelInfo, UpdateHyperparameters)
- Actions (TrainRound with real-time feedback and cancellation)
- Lifecycle Nodes (Aggregator: configure → activate → deactivate → shutdown)
- Parameters (dynamic reconfiguration of learning rate, batch size, etc.)
- QoS Profiles (reliable + transient local for critical data, best effort for metrics)
- Callback Groups (reentrant + mutually exclusive for concurrent execution)
- Multi-threaded Executors
- Custom Interfaces (msg/srv/action definitions in fl_robots_interfaces)
- Launch System (parameterized multi-node launch with staggered startup)
- Web Dashboard (Flask + Socket.IO MVC architecture)
"""

import sys
import os

# Add source to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'fl_robots'))


def print_banner():
    """Print project banner."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        🤖 ROS2 Federated Learning Multi-Robot System 🤖        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Architecture:                                                   ║
║    • Robot Agents  — Local training + Action Server              ║
║    • Aggregator    — FedAvg (Lifecycle Node) + Services          ║
║    • Coordinator   — State machine orchestration                 ║
║    • Monitor       — Metrics collection & persistence            ║
║    • Digital Twin  — Real-time visualization                     ║
║    • Web Dashboard — MVC + WebSocket (http://localhost:5000)     ║
║                                                                  ║
║  ROS2 Features:                                                  ║
║    Topics · Services · Actions · Lifecycle · Parameters          ║
║    QoS · Callback Groups · Multi-threaded Executor               ║
║    Custom msg/srv/action · Launch System                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


def run_quick_test():
    """Run a quick standalone test of the FL components (no ROS2 required)."""
    print("\n── Quick Component Test (no ROS2) ─────────────────────────")

    from fl_robots.models import SimpleNavigationNet, federated_averaging, compute_gradient_divergence
    from fl_robots.robot_agent import SyntheticDataGenerator
    import torch
    import numpy as np

    # 1. Model creation
    model = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)
    print(f"✅ Model created: {model.count_parameters()} parameters")

    # 2. Synthetic data generation
    gen = SyntheticDataGenerator("test_robot")
    X, y = gen.generate_batch(64)
    print(f"✅ Generated batch: X={X.shape}, y={y.shape}")

    # 3. Forward pass
    x_tensor = torch.tensor(X)
    output = model(x_tensor)
    print(f"✅ Forward pass: output={output.shape}")

    # 4. Training step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, torch.tensor(y))
    loss.backward()
    optimizer.step()
    print(f"✅ Training step: loss={loss.item():.4f}")

    # 5. Weight serialization
    weights = model.get_weights()
    print(f"✅ Weights extracted: {len(weights)} layers")

    # 6. Federated averaging (simulate 3 robots)
    models = [SimpleNavigationNet(12, 64, 4) for _ in range(3)]
    weights_list = [m.get_weights() for m in models]
    avg_weights = federated_averaging(weights_list, [100, 150, 200])
    print(f"✅ Federated averaging: {len(avg_weights)} layers averaged")

    # 7. Gradient divergence
    divergences = compute_gradient_divergence(weights_list, avg_weights)
    print(f"✅ Divergence: mean={np.mean(divergences):.4f}")

    # 8. Inference
    model.set_weights(avg_weights)
    model.eval()
    with torch.no_grad():
        sample = torch.randn(1, 12)
        probs = torch.softmax(model(sample), dim=-1)
        action = torch.argmax(probs).item()
        print(f"✅ Inference: action={action}, confidence={probs[0, action]:.3f}")

    print("\n✅ All component tests passed!\n")


def main():
    print_banner()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'test':
            run_quick_test()
        elif cmd == 'help':
            print("Usage:")
            print("  python main.py          Show project info")
            print("  python main.py test     Run quick component test")
            print("  python main.py help     Show this help")
            print()
            print("For full deployment:")
            print("  ./run.sh                Docker-based deployment")
            print("  ros2 launch fl_robots fl_system.launch.py")
        else:
            print(f"Unknown command: {cmd}")
            print("Run 'python main.py help' for usage.")
    else:
        print("Run 'python main.py test' for a quick component test.")
        print("Run './run.sh' to deploy with Docker.")
        print("Run 'ros2 launch fl_robots fl_system.launch.py' for ROS2 launch.")


if __name__ == '__main__':
    main()
