# Kubernetes Deployment

Production-style manifests for the fl-robots standalone dashboard.

## Apply

```bash
kubectl apply -f deploy/k8s/standalone.yaml
```

Before applying in any real environment, replace the mutable example image tag
in `standalone.yaml` with a released digest, for example:

```bash
kubectl set image deployment/fl-robots fl-robots=ghcr.io/<org>/fl-robots-standalone@sha256:<digest> -n fl-robots
```

This provisions:

- `Namespace` — `fl-robots`
- `ConfigMap` — runtime knobs (`FL_ROBOTS_RATE_*`, `FL_ROBOTS_READY_STALE_S`)
- `Deployment` — single-replica, non-root (uid 1000), read-only rootfs,
  dropped capabilities, seccomp `RuntimeDefault`, resource requests/limits,
  `/api/ready` as readiness probe, `/api/health` as liveness + startup probes
- `Service` — ClusterIP on port 80 → containerPort 5000
- `ServiceMonitor` — scrapes `/metrics` every 15 s (Prometheus Operator)
- `PrometheusRule` — mirrors `docs/prometheus-rules.yml` in-cluster
- `PodDisruptionBudget` — permissive; adjust for multi-replica

## Security baseline

- `runAsNonRoot: true`, `runAsUser: 1000`
- `readOnlyRootFilesystem: true` with `emptyDir` mounts for `/tmp` and `/app/results`
- All Linux capabilities dropped, no privilege escalation
- Seccomp profile `RuntimeDefault`
- No ServiceAccount token auto-mount
- Docker bases are pinned by digest and Python deps are installed from the
  checked-in `uv.lock`; keep the deployment image itself digest-pinned too.
- Security headers layered on top (see `standalone_web.py::_security_headers`):
  CSP, X-Frame-Options: DENY, X-Content-Type-Options: nosniff,
  Referrer-Policy: no-referrer, Permissions-Policy tight-locked

## SLO wiring

Probes match the SLOs in [`docs/SLO.md`](../../docs/SLO.md):

| SLO | Probe | Threshold |
|---|---|---|
| 1 – Readiness availability | `/api/ready` (readinessProbe) | 3 × 5 s failures → NotReady |
| 2 – Round latency p95 | Prometheus scrape | Histogram via `/metrics` |
| 3 – MPC solve time p99 | Prometheus scrape | Histogram via `/metrics` |

## Scaling notes

The simulation engine is stateful (in-process `SimulationEngine` instance),
so horizontal scaling is not meaningful — **keep `replicas: 1`**. For a real
multi-tenant deployment, swap the in-memory state for an external store
(Redis/Postgres) first.

