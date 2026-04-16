"""
In-process message bus that mirrors ROS2 topic semantics.

Provides publish/subscribe communication between simulation components
without requiring a ROS2 installation.  Supports wildcard (``"*"``)
subscriptions and thread-safe event history.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Callable

from .models import BusEvent

logger = logging.getLogger(__name__)

Subscriber = Callable[[BusEvent], None]


class MessageBus:
    """Thread-safe in-process pub/sub bus with ROS2 topic semantics.

    Features
    --------
    * Per-topic and wildcard (``"*"``) subscriptions.
    * Bounded event history for recent-event queries.
    * ``unsubscribe`` to cleanly remove handlers.
    * Handler exceptions are logged but never crash the bus.
    """

    def __init__(self, max_events: int = 250) -> None:
        self._lock = threading.RLock()
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._events: deque[BusEvent] = deque(maxlen=max_events)

    # ── Subscribe / Unsubscribe ─────────────────────────────────────

    def subscribe(self, topic: str, handler: Subscriber) -> None:
        """Register *handler* to be called whenever *topic* is published.

        Use ``topic="*"`` to receive every event regardless of topic.
        """
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

    # ── Publish ─────────────────────────────────────────────────────

    def publish(self, topic: str, source: str, payload: dict[str, object]) -> BusEvent:
        """Create a :class:`BusEvent` and deliver it to all matching subscribers.

        Parameters
        ----------
        topic:   ROS-style topic name, e.g. ``"/fl/robot_status"``.
        source:  Identifier of the publishing component.
        payload: Arbitrary data dictionary attached to the event.
        """
        event = BusEvent(
            timestamp=time.time(), topic=topic, source=source, payload=dict(payload),
        )
        with self._lock:
            self._events.append(event)
            subscribers = (
                list(self._subscribers.get(topic, ()))
                + list(self._subscribers.get("*", ()))
            )

        for handler in subscribers:
            try:
                handler(event)
            except Exception:
                logger.exception("Handler %r raised on topic %s", handler, topic)

        return event

    # ── Query ───────────────────────────────────────────────────────

    def recent_events(self, limit: int = 50) -> list[BusEvent]:
        """Return up to *limit* most recent events (oldest first)."""
        with self._lock:
            return list(self._events)[-limit:]

    @property
    def subscriber_count(self) -> int:
        """Total number of registered subscriptions (all topics)."""
        with self._lock:
            return sum(len(v) for v in self._subscribers.values())

