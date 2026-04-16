"""Exponential-backoff retry decorator.

Used by ROS service callers (``web_dashboard._call_*``) and anywhere else
we wrap a potentially-flaky side effect. Stdlib only — no tenacity.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, overload

__all__ = ["RetryConfig", "retry"]

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Retry parameters. ``jitter`` is uniform additive noise, not a factor."""

    attempts: int = 3
    base_delay: float = 0.2
    max_delay: float = 2.0
    jitter: float = 0.1
    backoff_factor: float = 2.0
    retry_on: tuple[type[BaseException], ...] = (Exception,)


@overload
def retry(fn: Callable[..., T]) -> Callable[..., T]: ...
@overload
def retry(*, config: RetryConfig) -> Callable[[Callable[..., T]], Callable[..., T]]: ...


def retry(
    fn: Callable[..., T] | None = None,
    *,
    config: RetryConfig | None = None,
) -> Any:
    """Decorator: retry *fn* with exponential backoff.

    Usage::

        @retry
        def flaky(): ...

        @retry(config=RetryConfig(attempts=5, base_delay=0.5))
        def picky(): ...
    """
    cfg = config or RetryConfig()

    def decorate(inner: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(inner)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = cfg.base_delay
            last_exc: BaseException | None = None
            for attempt in range(1, cfg.attempts + 1):
                try:
                    return inner(*args, **kwargs)
                except cfg.retry_on as exc:
                    last_exc = exc
                    if attempt == cfg.attempts:
                        break
                    sleep_for = min(delay, cfg.max_delay) + random.uniform(0.0, cfg.jitter)
                    logger.warning(
                        "retry: %s attempt %d/%d failed (%s); sleeping %.2fs",
                        inner.__qualname__,
                        attempt,
                        cfg.attempts,
                        exc,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                    delay *= cfg.backoff_factor
            assert last_exc is not None  # for type-checkers
            raise last_exc

        return wrapper

    if fn is not None and callable(fn):
        return decorate(fn)
    return decorate
