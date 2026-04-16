"""
Federated Learning Multi-Robot Coordination System
===================================================

ROS2 package for distributed federated learning across multiple robot agents.

**ROS2 mode** — requires ``rclpy``:
    Robot agents, aggregator, coordinator, monitor, digital twin, web dashboard.

**Standalone mode** — no ROS2 required:
    ``SimulationEngine``, ``MessageBus``, ``DistributedMPCPlanner``, ``create_app``.

    Run: ``python main.py`` or ``python main.py test``

Architecture
------------
- Robot Agent Nodes  – local model training & weight publishing
- Aggregator Node    – FedAvg server for weight aggregation
- Coordinator Node   – training orchestration state machine
- Monitor Node       – real-time metrics persistence
- Digital Twin Node  – matplotlib-based system visualisation
- Web Dashboard Node – Flask + Socket.IO real-time UI

All simulation-side data models are Pydantic-validated; see :mod:`sim_models`.
"""

__version__ = "1.0.0"
__author__ = "Developer"
