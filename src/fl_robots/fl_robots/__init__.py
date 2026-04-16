"""
Federated Learning Multi-Robot Coordination System

This ROS2 package implements distributed federated learning across multiple
simulated robot agents. It demonstrates:
- ROS2 Topics for model weight broadcasting
- ROS2 Services for robot registration and aggregation
- ROS2 Actions for training rounds with progress feedback
- ROS2 Parameters for configurable hyperparameters

Standalone mode (no ROS2 required):
- SimulationEngine: Full multi-agent simulation with MPC + FL (ros_web.simulation)
- MessageBus: In-process pub/sub mirroring ROS topics (ros_web.message_bus)
- DistributedMPCPlanner: Formation control planner (ros_web.mpc)
- create_app: Flask dashboard factory (ros_web.web)
- Run: ``python main.py`` or ``python main.py test``

Architecture:
- Robot Agent Nodes: Local model training and weight publishing
- Aggregator Node: FedAvg server for weight aggregation
- Coordinator Node: Training orchestration
- Monitor Node: Real-time metrics visualization
"""

__version__ = '1.0.0'
__author__ = 'Developer'
