"""SQLite-backed persistence for aggregation history and robot metrics.

The simulation engine and the ROS aggregator both produce streams of
:class:`AggregationRecord` objects. This module gives them a uniform place
to land — so operators can query historical runs, build dashboards, and
compare experiments without scraping JSON files.

Design notes
------------
* WAL mode by default for concurrent read/write.
* Schema managed by a single in-file ``SCHEMA_SQL`` string; idempotent create.
* Only stdlib — no SQLAlchemy dependency.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["MetricsStore"]


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS aggregation_rounds (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id         INTEGER NOT NULL,
    participants     INTEGER NOT NULL,
    mean_loss        REAL    NOT NULL,
    mean_accuracy    REAL    NOT NULL,
    mean_divergence  REAL    NOT NULL,
    formation_error  REAL    NOT NULL,
    recorded_at      REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_round_id ON aggregation_rounds(round_id);
CREATE INDEX IF NOT EXISTS ix_recorded_at ON aggregation_rounds(recorded_at);

CREATE TABLE IF NOT EXISTS robot_metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    robot_id         TEXT    NOT NULL,
    round_id         INTEGER NOT NULL,
    loss             REAL,
    accuracy         REAL,
    tracking_error   REAL,
    recorded_at      REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_robot_round ON robot_metrics(robot_id, round_id);

CREATE TABLE IF NOT EXISTS events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    topic            TEXT    NOT NULL,
    source           TEXT    NOT NULL,
    payload_json     TEXT    NOT NULL,
    recorded_at      REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_events_topic ON events(topic, recorded_at);
"""


class MetricsStore:
    """Thread-safe SQLite writer/reader for FL run metrics.

    Example::

        store = MetricsStore(Path("results/run.sqlite"))
        store.record_round({"round": 1, "participants": 3, ...})
        rows = store.fetch_rounds(limit=10)
    """

    def __init__(self, path: str | Path, *, wal: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._wal = wal
        self._conn = self._open()
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if self._wal:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    # ── Writers ──────────────────────────────────────────────────────

    def record_round(self, record: dict[str, Any]) -> None:
        """Persist a round record.

        Accepts either the engine's :meth:`AggregationRecord.as_dict` shape
        (``round``/``mean_loss``/etc.) or any dict containing those fields.
        """
        row = (
            int(record.get("round", record.get("round_id", 0))),
            int(record.get("participants", 0)),
            float(record.get("mean_loss", 0.0)),
            float(record.get("mean_accuracy", 0.0)),
            float(record.get("mean_divergence", 0.0)),
            float(record.get("formation_error", 0.0)),
            float(record.get("recorded_at", time.time())),
        )
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO aggregation_rounds
                    (round_id, participants, mean_loss, mean_accuracy,
                     mean_divergence, formation_error, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )

    def record_robot_metric(
        self,
        robot_id: str,
        round_id: int,
        *,
        loss: float | None = None,
        accuracy: float | None = None,
        tracking_error: float | None = None,
    ) -> None:
        """Persist a per-robot metric row (loss, accuracy, tracking error)."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO robot_metrics
                    (robot_id, round_id, loss, accuracy, tracking_error, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (robot_id, round_id, loss, accuracy, tracking_error, time.time()),
            )

    def record_event(self, topic: str, source: str, payload: dict[str, Any]) -> None:
        """Persist a bus event (topic, source, JSON payload)."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO events (topic, source, payload_json, recorded_at)
                VALUES (?, ?, ?, ?)
                """,
                (topic, source, json.dumps(payload, default=str), time.time()),
            )

    # ── Readers ──────────────────────────────────────────────────────

    def fetch_rounds(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent aggregation rounds (newest first)."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT round_id, participants, mean_loss, mean_accuracy,
                       mean_divergence, formation_error, recorded_at
                FROM aggregation_rounds
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def fetch_robot_metrics(
        self,
        robot_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return the most recent metrics for a given robot (newest first)."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT robot_id, round_id, loss, accuracy, tracking_error, recorded_at
                FROM robot_metrics WHERE robot_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (robot_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:  # pragma: no cover
                pass

    def __enter__(self) -> MetricsStore:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
