"""Structured logging configuration.

Uses ``structlog`` when available to emit JSON-formatted log records suitable
for ingestion by Loki / Cloud Logging / Datadog. Falls back gracefully to a
compact stdlib format if ``structlog`` isn't installed.

Configure once at process start by calling :func:`configure_logging`.
"""

from __future__ import annotations

import logging
import os
import sys

__all__ = ["configure_logging", "get_logger"]


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "on"}


def configure_logging(level: int | str | None = None, *, json_logs: bool | None = None) -> None:
    """Configure root logging for the process.

    Parameters
    ----------
    level
        Log level; defaults to ``$FL_ROBOTS_LOG_LEVEL`` or ``INFO``.
    json_logs
        When *True*, emit structured JSON. Defaults to ``$FL_ROBOTS_JSON_LOGS``.
    """
    resolved_level = (
        level if level is not None else os.environ.get("FL_ROBOTS_LOG_LEVEL", "INFO").upper()
    )
    json_mode = json_logs if json_logs is not None else _env_bool("FL_ROBOTS_JSON_LOGS", False)

    try:
        import structlog  # type: ignore
    except ImportError:  # pragma: no cover - optional dep
        structlog = None  # type: ignore[assignment]

    handler = logging.StreamHandler(stream=sys.stdout)
    root = logging.getLogger()
    # Reset handlers so repeated configure calls don't duplicate.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved_level)

    if structlog is not None and json_mode:
        timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
        shared_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        handler.setFormatter(formatter)
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )


def get_logger(name: str) -> logging.Logger:
    """Prefer ``structlog.get_logger`` when available; else stdlib."""
    try:
        import structlog  # type: ignore

        return structlog.get_logger(name)  # type: ignore[return-value]
    except ImportError:  # pragma: no cover
        return logging.getLogger(name)
