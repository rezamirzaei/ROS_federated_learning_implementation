"""Test-only helpers for driving ROS node code paths without a ROS install.

Use via the :func:`fake_ros` pytest fixture or the :class:`FakeROSEnvironment`
context manager. The fakes are functional — they store publishers,
subscribers, timers, and services so tests can inspect them and drive
callbacks directly.

Example
-------
::

    from fl_robots.testing.fake_ros import FakeROSEnvironment

    with FakeROSEnvironment() as env:
        from fl_robots.aggregator import AggregatorNode
        node = AggregatorNode()
        node.on_configure(None)
        env.publish("/fl/robot_A/model_weights", {"round": 1, ...})
        assert env.publishers["/fl/global_model"].messages
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class _Parameter:
    def __init__(self, value: Any = None) -> None:
        self.value = value


class _Logger:
    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.records: list[tuple[str, str]] = []

    def debug(self, *args: Any, **_: Any) -> None:
        self.records.append(("debug", " ".join(str(a) for a in args)))

    def info(self, *args: Any, **_: Any) -> None:
        self.records.append(("info", " ".join(str(a) for a in args)))

    def warning(self, *args: Any, **_: Any) -> None:
        self.records.append(("warning", " ".join(str(a) for a in args)))

    warn = warning

    def error(self, *args: Any, **_: Any) -> None:
        self.records.append(("error", " ".join(str(a) for a in args)))


class FakePublisher:
    """Records every published message."""

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.messages: list[Any] = []

    def publish(self, msg: Any) -> None:
        self.messages.append(msg)


class FakeSubscription:
    def __init__(self, topic: str, callback: Callable[[Any], None]) -> None:
        self.topic = topic
        self.callback = callback


class FakeTimer:
    """Records creation; tests can call ``.fire()`` to invoke the callback."""

    def __init__(self, period: float, callback: Callable[[], None]) -> None:
        self.period = period
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.callback()


class FakeClient:
    def __init__(self, service_name: str) -> None:
        self.service_name = service_name
        self.requests: list[Any] = []

    def wait_for_service(self, timeout_sec: float = 0.0) -> bool:
        return True

    def call_async(self, request: Any) -> FakeFuture:
        self.requests.append(request)
        fut = FakeFuture()
        fut.set_result(None)
        return fut


class FakeFuture:
    def __init__(self) -> None:
        self._done = False
        self._result: Any = None
        self._callbacks: list[Callable[[FakeFuture], None]] = []

    def set_result(self, value: Any) -> None:
        self._result = value
        self._done = True
        for cb in self._callbacks:
            cb(self)

    def done(self) -> bool:
        return self._done

    def result(self) -> Any:
        return self._result

    def add_done_callback(self, cb: Callable[[FakeFuture], None]) -> None:
        self._callbacks.append(cb)
        if self._done:
            cb(self)


class FakeService:
    def __init__(self, service_name: str, callback: Callable[..., Any]) -> None:
        self.service_name = service_name
        self.callback = callback


class FakeActionServer:
    def __init__(
        self,
        node: Any,
        action_type: Any,
        action_name: str,
        execute_callback: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.node = node
        self.action_name = action_name
        self.execute_callback = execute_callback
        self.kwargs = kwargs


class FakeNode:
    """Minimal functional stand-in for ``rclpy.node.Node``."""

    def __init__(
        self, name: str = "fake", *, environment: FakeROSEnvironment | None = None, **_: Any
    ) -> None:
        self._name = name
        # If no explicit env provided, bind to the currently-active one so
        # that ``BaseNode`` references captured at module import time still
        # route publications/subscriptions to the test's env.
        if environment is None:
            environment = _CURRENT_ENV
        self._env = environment
        self._parameters: dict[str, _Parameter] = {}
        self._parameter_callbacks: list[Callable[..., Any]] = []
        self.publishers: dict[str, FakePublisher] = {}
        self.subscriptions: dict[str, FakeSubscription] = {}
        self.services: dict[str, FakeService] = {}
        self.clients: dict[str, FakeClient] = {}
        self.timers: list[FakeTimer] = []
        self._logger = _Logger(name)
        self._destroyed = False
        if environment is not None:
            environment.nodes.append(self)

    # Parameters
    def declare_parameter(self, name: str, default: Any = None) -> _Parameter:
        # Allow the env to override defaults so tests can steer paths
        # (e.g. write to a tmp dir instead of ``/ros2_ws/results``).
        if self._env is not None and name in self._env.parameter_overrides:
            default = self._env.parameter_overrides[name]
        param = _Parameter(default)
        self._parameters[name] = param
        return param

    def declare_parameters(
        self, namespace: str, parameters: list[tuple[str, Any]]
    ) -> list[_Parameter]:
        return [self.declare_parameter(name, default) for name, default in parameters]

    def get_parameter(self, name: str) -> _Parameter:
        return self._parameters.get(name, _Parameter(None))

    def set_parameters(
        self, params: list[Any]
    ) -> list[Any]:  # pragma: no cover - rarely used in tests
        for p in params:
            name = getattr(p, "name", None)
            if name:
                self._parameters[name] = _Parameter(getattr(p, "value", None))

        class _Result:
            successful = True

        return [_Result() for _ in params]

    def add_on_set_parameters_callback(self, cb: Callable[..., Any]) -> None:
        self._parameter_callbacks.append(cb)

    # Logging
    def get_logger(self) -> _Logger:
        return self._logger

    # Pub/sub
    def create_publisher(self, msg_type: Any, topic: str, qos: Any, **_: Any) -> FakePublisher:
        pub = FakePublisher(topic)
        self.publishers[topic] = pub
        if self._env is not None:
            self._env.publishers[topic] = pub
        return pub

    def create_subscription(
        self, msg_type: Any, topic: str, callback: Callable[[Any], None], qos: Any, **_: Any
    ) -> FakeSubscription:
        sub = FakeSubscription(topic, callback)
        self.subscriptions[topic] = sub
        if self._env is not None:
            self._env.subscriptions.setdefault(topic, []).append(sub)
        return sub

    # Services / clients
    def create_service(
        self, srv_type: Any, name: str, callback: Callable[..., Any], **_: Any
    ) -> FakeService:
        svc = FakeService(name, callback)
        self.services[name] = svc
        if self._env is not None:
            self._env.services[name] = svc
        return svc

    def create_client(self, srv_type: Any, name: str, **_: Any) -> FakeClient:
        client = FakeClient(name)
        self.clients[name] = client
        return client

    # Timers
    def create_timer(self, period: float, callback: Callable[[], None], **_: Any) -> FakeTimer:
        timer = FakeTimer(period, callback)
        self.timers.append(timer)
        if self._env is not None:
            self._env.timers.append(timer)
        return timer

    # Lifecycle
    def destroy_node(self) -> None:
        self._destroyed = True


class FakeExecutor:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._nodes: list[Any] = []

    def add_node(self, node: Any) -> None:
        self._nodes.append(node)

    def spin(self) -> None:  # pragma: no cover - tests drive callbacks manually
        pass

    def shutdown(self) -> None:  # pragma: no cover
        pass


class FakeROSEnvironment:
    """Context manager + pytest fixture target that installs functional fakes.

    Patches :mod:`fl_robots.ros_compat` module-level attributes in place, so
    any module that already imported ``Node`` / ``rclpy`` / etc. at import
    time sees the fakes for the duration of the test.
    """

    def __init__(self) -> None:
        self.nodes: list[FakeNode] = []
        self.publishers: dict[str, FakePublisher] = {}
        self.subscriptions: dict[str, list[FakeSubscription]] = {}
        self.services: dict[str, FakeService] = {}
        self.timers: list[FakeTimer] = []
        self.parameter_overrides: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._patches: dict[str, Any] = {}
        self._prev_env: FakeROSEnvironment | None = None

    def publish(self, topic: str, message: Any) -> int:
        """Invoke every subscriber for *topic*. Returns the count delivered."""
        subs = self.subscriptions.get(topic, [])
        for sub in subs:
            sub.callback(message)
        return len(subs)

    def fire_timers(self) -> None:
        for t in list(self.timers):
            t.fire()

    def __enter__(self) -> FakeROSEnvironment:
        from fl_robots import ros_compat

        global _CURRENT_ENV
        self._prev_env = _CURRENT_ENV
        _CURRENT_ENV = self

        targets = {
            "ROS_AVAILABLE": True,
            "Node": FakeNode,
            "MultiThreadedExecutor": FakeExecutor,
            "ActionServer": FakeActionServer,
        }

        for key, value in targets.items():
            self._patches[key] = getattr(ros_compat, key, None)
            setattr(ros_compat, key, value)

        class _RclpyOK:
            ok_flag = True

            def init(self, *a: Any, **kw: Any) -> None: ...
            def shutdown(self, *a: Any, **kw: Any) -> None: ...
            def spin(self, *a: Any, **kw: Any) -> None: ...  # pragma: no cover
            def ok(self) -> bool:
                return self.ok_flag

        self._patches["rclpy"] = ros_compat.rclpy
        ros_compat.rclpy = _RclpyOK()
        return self

    def __exit__(self, *exc: Any) -> None:
        from fl_robots import ros_compat

        for key, value in self._patches.items():
            setattr(ros_compat, key, value)
        self._patches.clear()

        global _CURRENT_ENV
        _CURRENT_ENV = self._prev_env

    def _make_node_class(self) -> type:  # kept for backwards compat
        return FakeNode


_CURRENT_ENV: FakeROSEnvironment | None = None
