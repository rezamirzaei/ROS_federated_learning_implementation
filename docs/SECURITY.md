# Security Policy

## Reporting a vulnerability

Please **do not** open public issues for security problems. Instead email
`security@fl-robots.local` with a description, reproducer, and any logs. We
aim to acknowledge within 2 business days.

## Threat model

This project is a **research / teaching demo**. It has never been audited for
production deployment. The default configuration trades safety for
convenience — in particular the dashboards run over plain HTTP on localhost
with no authentication. Before putting anything from this repo on a network
you do not control, you should at minimum:

1. **Set a bearer token.** Export `FL_ROBOTS_API_TOKEN=<random string>`
   before launching the standalone dashboard. All mutating endpoints will
   then require `Authorization: Bearer <token>`.
2. **Terminate TLS** in front of the Flask app (nginx, Caddy, Traefik, etc.).
   The built-in Flask dev server is unencrypted and single-threaded.
3. **Do not expose `/api/command`** to untrusted networks. It can reset
   simulation state, inject disturbances, and start/stop training.
4. **Review the ROS QoS settings** in `aggregator.py` and
   `robot_agent.py`. Some topics use `BEST_EFFORT` for telemetry; weight
   exchange uses `RELIABLE`+`TRANSIENT_LOCAL`. DDS security is not enabled
   by default; see the official ROS2 `sros2` tooling for production
   deployments.

## Hardening checklist

* [ ] `FL_ROBOTS_API_TOKEN` set in every launched environment.
* [ ] TLS reverse proxy in front of the Flask dashboard.
* [ ] `FL_ROBOTS_JSON_LOGS=1` for ingestion-friendly audit logs.
* [ ] Run the container as a non-root user (override the Dockerfile `USER`).
* [ ] Use a persistent volume for the SQLite `MetricsStore` and back it up.
* [ ] Keep dependencies patched — CI uses locked `uv sync` installs and runs
      the benchmark smoke gate on PRs; security-sensitive updates should land
      promptly.

## Secrets hygiene

`gitleaks` is enabled via pre-commit (see `.pre-commit-config.yaml`) to
prevent committed secrets. Please install pre-commit locally:

```bash
uv run pre-commit install
```
