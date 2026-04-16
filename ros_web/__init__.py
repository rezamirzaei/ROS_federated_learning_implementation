"""
Standalone web package for the Federated Learning system.

Provides an in-process simulation of the full ROS2 multi-robot FL pipeline
without requiring a ROS2 installation.  Start via ``python main.py``.

Components:
- MessageBus:         In-process pub/sub mirroring ROS2 topics
- SimulationEngine:   Multi-agent FL + MPC simulation loop
- DistributedMPCPlanner: Formation-control planner
- create_app:         Flask application factory
"""

__all__ = [
    "MessageBus",
    "SimulationEngine",
    "DistributedMPCPlanner",
    "create_app",
]

