"""Compatibility helpers for importing ROS-dependent modules outside a ROS environment.

This project supports two modes:
- standalone simulation/web mode on regular Python
- ROS2 node mode with ``rclpy`` and generated message packages available

Importing ROS node modules should not crash local tooling, tests, or non-ROS runs.
This module exposes the ROS symbols when available and lightweight placeholders
otherwise. Call :func:`require_ros` before starting any real ROS node logic.
"""

from __future__ import annotations

from typing import Any

ROS_IMPORT_ERROR: ImportError | None = None
ROS_AVAILABLE = False

try:  # pragma: no cover - exercised only in a real ROS environment
    import rclpy as rclpy
    from geometry_msgs.msg import Twist as Twist
    from rcl_interfaces.msg import SetParametersResult as SetParametersResult
    from rclpy.action import ActionServer as ActionServer
    from rclpy.action import CancelResponse as CancelResponse
    from rclpy.action import GoalResponse as GoalResponse
    from rclpy.callback_groups import (
        MutuallyExclusiveCallbackGroup as MutuallyExclusiveCallbackGroup,
    )
    from rclpy.callback_groups import ReentrantCallbackGroup as ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor as MultiThreadedExecutor
    from rclpy.node import Node as Node
    from rclpy.qos import DurabilityPolicy as DurabilityPolicy
    from rclpy.qos import HistoryPolicy as HistoryPolicy
    from rclpy.qos import QoSProfile as QoSProfile
    from rclpy.qos import ReliabilityPolicy as ReliabilityPolicy
    from std_msgs.msg import String as String

    ROS_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - exercised in local non-ROS dev
    ROS_IMPORT_ERROR = exc

    class _RclpyStub:
        def init(self, *args: Any, **kwargs: Any) -> None:
            require_ros()

        def shutdown(self, *args: Any, **kwargs: Any) -> None:
            require_ros()

    rclpy = _RclpyStub()

    class _RosStub:
        """Base stub that raises a clear error when instantiated or called."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_ros()

    class _Logger:
        def debug(self, *args: Any, **kwargs: Any) -> None:
            require_ros()

        info = warning = error = debug

    class _Parameter:
        def __init__(self, value: Any = None) -> None:
            self.value = value

    class _Publisher:
        def publish(self, *args: Any, **kwargs: Any) -> None:
            require_ros()

    class _Client:
        def wait_for_service(self, *args: Any, **kwargs: Any) -> bool:
            require_ros()
            return False

        def call_async(self, *args: Any, **kwargs: Any) -> None:
            require_ros()

    class _ExecutorBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._nodes: list[Any] = []

        def add_node(self, node: Any) -> None:
            require_ros()

        def spin(self) -> None:
            require_ros()

    class Node:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_ros()

        def declare_parameter(self, *args: Any, **kwargs: Any) -> None:
            require_ros()

        def get_parameter(self, *args: Any, **kwargs: Any) -> _Parameter:
            require_ros()
            return _Parameter()

        def get_logger(self) -> _Logger:
            return _Logger()

        def create_subscription(self, *args: Any, **kwargs: Any) -> None:
            require_ros()
            return None

        def create_timer(self, *args: Any, **kwargs: Any) -> None:
            require_ros()
            return None

        def create_publisher(self, *args: Any, **kwargs: Any) -> _Publisher:
            require_ros()
            return _Publisher()

        def create_client(self, *args: Any, **kwargs: Any) -> _Client:
            require_ros()
            return _Client()

        def add_on_set_parameters_callback(self, *args: Any, **kwargs: Any) -> None:
            require_ros()

        def destroy_node(self) -> None:
            require_ros()

    class ReentrantCallbackGroup(_RosStub):
        pass

    class MutuallyExclusiveCallbackGroup(_RosStub):
        pass

    class MultiThreadedExecutor(_ExecutorBase):
        pass

    class QoSProfile(_RosStub):
        pass

    class ActionServer(_RosStub):
        pass

    class ReliabilityPolicy:
        RELIABLE = "RELIABLE"
        BEST_EFFORT = "BEST_EFFORT"

    class HistoryPolicy:
        KEEP_LAST = "KEEP_LAST"

    class DurabilityPolicy:
        TRANSIENT_LOCAL = "TRANSIENT_LOCAL"

    class CancelResponse:
        ACCEPT = "ACCEPT"
        REJECT = "REJECT"

    class GoalResponse:
        ACCEPT = "ACCEPT"
        REJECT = "REJECT"

    class SetParametersResult:
        def __init__(self, successful: bool = False) -> None:
            self.successful = successful

    class String:
        def __init__(self, data: str = "") -> None:
            self.data = data

    class _Vector3:
        def __init__(self) -> None:
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Twist:
        def __init__(self) -> None:
            self.linear = _Vector3()
            self.angular = _Vector3()


def require_ros() -> None:
    """Raise a clear runtime error if ROS2 packages are unavailable."""
    if ROS_AVAILABLE:
        return
    detail = f": {ROS_IMPORT_ERROR}" if ROS_IMPORT_ERROR else "."
    raise RuntimeError(
        "ROS2 dependencies are not available in this Python environment"
        f"{detail} Install/source ROS2 Humble and the workspace before running ROS nodes."
    )


__all__ = [
    "ROS_AVAILABLE",
    "ROS_IMPORT_ERROR",
    "ActionServer",
    "CancelResponse",
    "DurabilityPolicy",
    "GoalResponse",
    "HistoryPolicy",
    "MultiThreadedExecutor",
    "MutuallyExclusiveCallbackGroup",
    "Node",
    "QoSProfile",
    "ReentrantCallbackGroup",
    "ReliabilityPolicy",
    "SetParametersResult",
    "String",
    "Twist",
    "rclpy",
    "require_ros",
]
