"""
In-process message bus that mirrors ROS2 topic semantics.

Supports publish / subscribe with string-keyed topics and JSON payloads,
exactly like the ``std_msgs/String`` protocol used by the ROS2 nodes.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List


class MessageBus:
    """Thread-safe publish / subscribe message bus."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: Dict[str, List[Callable[[dict], None]]] = defaultdict(list)
        self._latest: Dict[str, dict] = {}

    # ── Core API ────────────────────────────────────────────────────

    def publish(self, topic: str, data: dict) -> None:
        """Publish *data* on *topic*.  All registered callbacks are invoked."""
        with self._lock:
            self._latest[topic] = data
            callbacks = list(self._subscribers[topic])
        for cb in callbacks:
            try:
                cb(data)
            except Exception:
                pass  # Mirrors ROS best-effort semantics

    def subscribe(self, topic: str, callback: Callable[[dict], None]) -> None:
        """Register *callback* to be called on every publish to *topic*."""
        with self._lock:
            self._subscribers[topic].append(callback)

    def latest(self, topic: str) -> dict | None:
        """Return the most recent message on *topic*, or ``None``."""
        with self._lock:
            return self._latest.get(topic)

    # ── Convenience ─────────────────────────────────────────────────

    def topics(self) -> List[str]:
        with self._lock:
            return list(self._latest.keys())

