"""Package for small, dependency-free utility helpers."""

from __future__ import annotations

from .retry import RetryConfig, retry

__all__ = ["RetryConfig", "retry"]
