from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Callable

from .models import BusEvent


Subscriber = Callable[[BusEvent], None]


class MessageBus:
    """Small in-process message bus that mirrors ROS topic semantics."""

    def __init__(self, max_events: int = 250) -> None:
        self._lock = threading.RLock()
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._events: deque[BusEvent] = deque(maxlen=max_events)

    def subscribe(self, topic: str, handler: Subscriber) -> None:
        with self._lock:
            self._subscribers[topic].append(handler)

    def publish(self, topic: str, source: str, payload: dict[str, object]) -> BusEvent:
        event = BusEvent(timestamp=time.time(), topic=topic, source=source, payload=dict(payload))
        with self._lock:
            self._events.append(event)
            subscribers = list(self._subscribers.get(topic, ())) + list(self._subscribers.get("*", ()))

        for handler in subscribers:
            handler(event)

        return event

    def recent_events(self, limit: int = 50) -> list[BusEvent]:
        with self._lock:
            return list(self._events)[-limit:]
