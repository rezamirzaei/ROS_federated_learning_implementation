"""Optional OpenTelemetry tracing integration.

Activated only when ``FL_ROBOTS_OTEL=1`` is set in the environment **and**
the ``otel`` extra is installed. Keeping it opt-in avoids dragging the
full OpenTelemetry SDK into the core dependency footprint.

Typical usage:

    from fl_robots.observability.tracing import maybe_setup_tracing, span

    maybe_setup_tracing(service_name="fl-robots-standalone")

    with span("aggregate_round", round=5):
        run_aggregation()

Call ``maybe_setup_tracing`` exactly once, at application startup
(e.g. inside ``create_app``). Subsequent calls are no-ops.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

__all__ = ["maybe_setup_tracing", "span", "tracing_enabled"]

_SETUP_DONE = False
_TRACER: Any = None


def tracing_enabled() -> bool:
    """Return True iff tracing was successfully configured."""
    return _TRACER is not None


def maybe_setup_tracing(service_name: str = "fl-robots") -> bool:
    """Configure OTLP tracing once if requested and available.

    Returns True when tracing is active after the call, False otherwise.
    Never raises: missing optional deps simply yield a no-op tracer.
    """
    global _SETUP_DONE, _TRACER
    if _SETUP_DONE:
        return _TRACER is not None
    _SETUP_DONE = True

    if os.environ.get("FL_ROBOTS_OTEL", "0").lower() not in ("1", "true", "yes"):
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return False

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": os.environ.get("FL_ROBOTS_VERSION", "unknown"),
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("fl_robots")
    return True


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Context manager wrapping :meth:`Tracer.start_as_current_span`.

    No-op when tracing hasn't been configured — safe to sprinkle
    throughout hot paths.
    """
    if _TRACER is None:
        with nullcontext() as ctx:
            yield ctx
        return
    with _TRACER.start_as_current_span(name) as s:
        for key, value in attributes.items():
            try:
                s.set_attribute(key, value)
            except Exception:  # pragma: no cover - defensive
                pass
        yield s
