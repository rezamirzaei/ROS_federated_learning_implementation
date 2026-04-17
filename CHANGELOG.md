# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Runtime FedProx** wired end-to-end — the aggregator broadcasts
  ``algorithm`` and ``proximal_mu`` inside the ``/fl/global_model``
  payload's ``config`` block; ``robot_agent.py`` latches the snapshot and
  injects the ``½·μ·‖w − w_global‖²`` proximal term into every local
  training loop (both the action-server and topic-command paths).
  No longer benchmark-only. 8 dedicated tests in
  ``tests/test_fedprox_runtime.py`` covering:
  opt-in default (FedAvg), snapshot capture under FedProx, zero-distance
  penalty at the snapshot point, analytical penalty growth with drift,
  and gradient-descent direction toward the snapshot.
- **Kubernetes manifests** under ``deploy/k8s/`` — non-root,
  read-only-rootfs, dropped caps, seccomp RuntimeDefault, probes wired to
  ``/api/ready``/``/api/health``, ``ServiceMonitor`` +
  ``PrometheusRule``, ``PodDisruptionBudget``.
- **Release workflow** (``.github/workflows/release.yml``) — tag-driven
  PyPI Trusted Publishing, multi-arch (amd64+arm64) Docker builds to
  GHCR, SBOM (`anchore/sbom-action`), provenance attestations
  (`actions/attest-build-provenance`), automated GitHub Release notes.
- **Docs site workflow** (``.github/workflows/docs.yml``) + ``mkdocs.yml``
  with ``mkdocs-material`` + ``mkdocstrings`` — API reference deployed to
  GitHub Pages on every merge.
- **Playwright frontend smoke suite** under ``web/`` — exercises
  index render, `/api/health`, security headers, command lifecycle, and
  unknown-command 400.
- **End-to-end command-flow test** (``tests/test_e2e_command_flow.py``):
  full start→step×5→/metrics→stop lifecycle plus rate-limit overflow and
  Bearer-token auth tests.
- **`/metrics` on ROS ``web_dashboard.py``** sharing the same registry as
  the standalone app; security headers added there too.
- **Nightly `ros-launch-tests` CI job** running ``launch_testing``
  scenarios inside the ROS 2 runtime image (``src/fl_robots/test/``).
- **Benchmark regression gate** — ``scripts/compare.py`` with
  ``--fail-on-regression --max-accuracy-drop``; ``results/baseline_ci.json``
  checked in; wired into the CI smoke benchmark job.
- **Communication-cost accounting** in the benchmark output
  (``param_bytes_per_client``, ``bytes_per_round``, ``total_bytes``).
- **Optional OpenTelemetry tracing** (``otel`` extra) — new
  ``observability/tracing.py`` helper with ``maybe_setup_tracing`` +
  ``span`` context manager, opt-in via ``FL_ROBOTS_OTEL=1``; wrapped
  ``/api/command`` dispatch in a span.
- **Non-root Docker runtime users** (UID 1000) for both
  ``standalone-runtime`` and ``ros-runtime`` stages.
- **Hardened ``.dockerignore``** — excludes `.hypothesis/`, caches,
  MNIST raw bytes, `data/MNIST/raw/`, `.DS_Store`, `.vscode/`.

### Changed

- **Coverage gate 65 → 70 %**. Expanded standalone coverage and test breadth;
  ``web_dashboard.py`` and ``scripts/benchmark.py`` remain omitted from the
  core coverage gate because they require optional/runtime-specific
  dependencies.
- ``pyproject.toml``: new ``otel`` optional-dep group.
- ``pytest.ini_options.norecursedirs`` now excludes
  ``src/fl_robots/test`` so standalone pytest runs don't try to collect
  the ROS launch tests.

### Added
- Root `LICENSE` file (MIT) — previously only declared in `pyproject.toml`.
- `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1).
- `docs/DEVELOPMENT.md` — local setup notes covering uv, torch CPU wheel,
  optional-deps matrix, macOS port conflicts, osqp build issues.
- `docs/SLO.md` — service-level objectives for the standalone dashboard
  (round-latency p95, readiness availability, MPC solve time budget).
- `docs/prometheus-rules.yml` — shipped alert rules (round stall, divergence
  spike, tracking-error spike, controller stuck, memory ceiling).
- Security headers (`Content-Security-Policy`, `X-Frame-Options`,
  `Referrer-Policy`, `X-Content-Type-Options`, `Permissions-Policy`) on all
  Flask responses via an `after_request` hook in `standalone_web.py`.
- `gitleaks` pre-commit hook + `pre-commit` CI job.
- Python 3.13 in the CI test matrix and `pyproject.toml` classifiers.
- `pyright` lint-only CI job (allowed to fail during ramp-up).
- Codecov coverage upload in CI.
- `pytest-xdist` in dev extras (`-n auto` opt-in).

### Changed
- Coverage gate ratcheted from 60 % → 65 % (`fail_under` in `pyproject.toml`).
- ROS runtime Docker stage now installs Python deps via
  `pip install '.[ml,ros]'` instead of duplicating pins — removes drift
  risk between `pyproject.toml` and the Dockerfile.

### Fixed
- README license badge no longer 404s.

## [1.0.0] — 2025-10-01

Initial public release.

### Added
- Dual-mode (standalone + ROS 2 Humble) federated learning demo.
- FedAvg reference benchmark + FedProx (benchmark-only) with
  Dirichlet non-IID splits and multi-seed reporting.
- OSQP-backed distributed MPC with linearised collision keep-out
  constraints, slew limits, terminal cost, and warm-starting.
- Flask dashboard with Prometheus `/metrics`, OpenAPI 3.1 schema,
  Bearer-token auth, sliding-window rate limiting, liveness/readiness
  probes.
- ROS 2 workspace with custom `fl_robots_interfaces` (msg/srv/action),
  lifecycle aggregator, multi-threaded executors, reentrant callback
  groups.
- Multi-stage Dockerfile (`standalone-runtime`, `standalone-test`,
  `ros-builder`, `ros-runtime`) and `compose.yaml` with healthchecks.
- GitHub Actions CI: lint, matrix tests (3.10/3.11/3.12), Docker build,
  ROS 2 workspace build, MNIST smoke benchmark.
