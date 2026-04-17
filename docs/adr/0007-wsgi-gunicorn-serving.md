# ADR-0007: WSGI Serving via Gunicorn

**Status:** Accepted
**Date:** 2026-04-17

## Context

The standalone container previously ran the Flask development server directly.
The dev server is single-threaded, unencrypted, and not designed for production
traffic. It also lacks process management, graceful shutdown, and worker
recycling.

## Decision

Replace the container `CMD` with `gunicorn` pointing at the WSGI entrypoint
(`fl_robots.wsgi:app`). Configuration:

- Workers: 1 (the simulation engine is stateful and in-process).
- Bind: `0.0.0.0:5000`.
- Timeout: 30 s (aligned with the k8s liveness probe).
- Access log: stdout (structured JSON when `FL_ROBOTS_JSON_LOGS=1`).

A `HEALTHCHECK` instruction is added to the Dockerfile so `docker ps` shows
container health without requiring orchestrator probes.

## Consequences

- The container is now suitable for direct deployment behind a TLS reverse proxy.
- Single-worker is intentional: the `SimulationEngine` holds mutable state.
  Scaling requires externalising state (Redis / Postgres) first.
- The `wsgi.py` module provides a stable WSGI callable for any WSGI server.
