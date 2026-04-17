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
2. **Terminate TLS** in front of the dashboard (nginx, Caddy, Traefik, etc.).
   The shipped standalone container now uses `gunicorn`, but it is still plain
   HTTP by default and should sit behind a TLS reverse proxy.
3. **Do not expose `/api/command`** to untrusted networks. It can reset
   simulation state, inject disturbances, and start/stop training.
4. **Review the ROS QoS settings** in `aggregator.py` and
   `robot_agent.py`. Some topics use `BEST_EFFORT` for telemetry; weight
   exchange uses `RELIABLE`+`TRANSIENT_LOCAL`. DDS security is not enabled
   by default; see the official ROS2 `sros2` tooling for production
   deployments.
5. **Deploy immutable container references.** The Dockerfile pins base images
   by digest and uses locked `uv` installs. The k8s manifest and Helm chart
   both reference the application image by digest. Always update the digest
   after each release.

## Current security posture

| Layer | Status |
|---|---|
| Container bases | Digest-pinned (`python:3.11-slim`, `ros:humble-*`) |
| Python deps | Locked via `uv.lock`, installed with `--locked` |
| Application image | Digest-pinned in k8s manifest and Helm chart |
| CSP | Fully self-hosted (no third-party CDN origins) |
| CSRF | Double-submit cookie (standalone + ROS dashboard) |
| Auth | Optional bearer token (`FL_ROBOTS_API_TOKEN`) |
| Scanning | Trivy (standalone + ROS images), CodeQL, Bandit, dependency-review |
| Secrets | gitleaks pre-commit hook |
| Network | k8s NetworkPolicy restricts ingress/egress |

## Hardening checklist

* [ ] `FL_ROBOTS_API_TOKEN` set in every launched environment.
* [ ] TLS reverse proxy in front of the Flask dashboard.
* [ ] `FL_ROBOTS_JSON_LOGS=1` for ingestion-friendly audit logs.
* [ ] Run the container as a non-root user (override the Dockerfile `USER`).
* [ ] Deploy image digests instead of mutable tags in Kubernetes / Compose.
* [ ] Use a persistent volume for the SQLite `MetricsStore` and back it up.
* [ ] Keep dependencies patched — CI uses locked `uv sync` installs, pinned
      container base digests, image scans, and the benchmark smoke gate on PRs;
      security-sensitive updates should land promptly.

## Secrets hygiene

`gitleaks` is enabled via pre-commit (see `.pre-commit-config.yaml`) to
prevent committed secrets. Please install pre-commit locally:

```bash
uv run pre-commit install
```
