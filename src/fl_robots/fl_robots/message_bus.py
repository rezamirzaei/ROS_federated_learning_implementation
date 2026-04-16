"""
In-process message bus that mirrors ROS2 topic semantics.

Used by the standalone simulation engine so the full system can run
without a ROS2 installation.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

from .sim_models import BusEvent

logger = logging.getLogger(__name__)

Subscriber = Callable[[BusEvent], None]

__all__ = ["MessageBus", "Subscriber", "BusEvent"]


class MessageBus:
    """Thread-safe in-process pub/sub bus with ROS2 topic semantics."""

    def __init__(self, max_events: int = 250) -> None:
        self._lock = threading.RLock()
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._events: deque[BusEvent] = deque(maxlen=max_events)

    def subscribe(self, topic: str, handler: Subscriber) -> None:
        """Register *handler* for *topic*.  Use ``"*"`` for wildcard."""
        with self._lock:
            self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Subscriber) -> bool:
        """Remove *handler* from *topic*.  Returns ``True`` if found."""
        with self._lock:
            try:
                self._subscribers[topic].remove(handler)
                return True
            except ValueError:
                return False

    def publish(self, topic: str, source: str, payload: dict[str, object]) -> BusEvent:
        event = BusEvent(timestamp=time.time(), topic=topic, source=source, payload=dict(payload))
        with self._lock:
            self._events.append(event)
            subscribers = list(self._subscribers.get(topic, ())) + list(self._subscribers.get("*", ()))

        for handler in subscribers:
            try:
                handler(event)
            except Exception:
                logger.exception("Handler %r raised on topic %s", handler, topic)

        return event

    def recent_events(self, limit: int = 50) -> list[BusEvent]:
        with self._lock:
            return list(self._events)[-limit:]

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._subscribers.values())
