 # fl-robots — Engineering Audit & Gap-Closing Plan

*Audit date: 2026-04-17 · Scope: `/Users/rezami/PycharmProjects/ROS` · Method: static review of `pyproject.toml`, `.github/workflows/*.yml`, `src/fl_robots/fl_robots/**`, `tests/**`, `docker/Dockerfile`, `compose.yaml`, `deploy/k8s/standalone.yaml`, `docs/**`, `scripts/**`, `mkdocs.yml`, `CHANGELOG.md`, `CONTRIBUTING.md`, `.pre-commit-config.yaml`.*

## TL;DR

This is a **well-above-average research/teaching repo** with a coherent dual-mode design, ADR-backed architecture notes, real property-based tests, a regression-gated benchmark, locked CI installs, stronger operator docs, release provenance, SBOM, hardened k8s manifests, and an opt-in OTEL path. It is **not** production-grade: the FL math still has a real correctness smell (BatchNorm running stats are weighted by sample count in `federated_averaging`), the OSQP planner still rebuilds `osqp.OSQP()` every tick defeating much of the warm-start, mypy remains in a **relaxed, opt-in-strictness mode**, and important security/ops gaps remain (**no container CVE scans / SAST / CodeQL / mutation testing / NetworkPolicy / Helm chart**, CSP still allows `'unsafe-inline'`, no CSRF).

### Aggregate score (unweighted mean across 14 sections)

`(9.5 + 7 + 7 + 6.5 + 8 + 7 + 8 + 7.5 + 6.5 + 8.5 + 7.5 + 9.2 + 8.5 + 6.5) / 14 ≈ **7.7 / 10**`

**Overall characterisation.** A polished demo-grade system with genuine engineering effort (property tests, regression gates, lifecycle nodes, ADRs, runbook/observability docs, SBOM, k8s hardening). Blocked from production-grade primarily by a **correctness bug in FedAvg BN handling**, **an OSQP anti-pattern** that negates most of the warm-start claim, **remaining security hygiene gaps** (image scan, CodeQL, CSRF, CSP `unsafe-inline`), and **typing/CI rigor** that is still intentionally softer than a production service.

---

## 1. Section-by-section scorecard

### 1.1 Architecture & design — **9.5 / 10**

**Evidence.** `docs/ARCHITECTURE.md` now carries Mermaid context/container diagrams alongside the dual-mode explanation; `docs/adr/0001-dual-mode-ros-shim.md` through `0004-lifecycle-vs-node.md` record the main design choices; `fl_robots/ros_compat.py` is still the single seam between ROS and stub modes; `message_bus.MessageBus` mirrors ROS topic semantics in-process; planners remain swappable behind a common solve shape; persistence is still a thin stdlib SQLite wrapper.

**Why not 10.** The key design decisions are now documented, but the `coordinator`/`digital_twin`/`monitor` modules are still typed-out of mypy (`pyproject.toml` `ignore_errors = true`), which signals unresolved design debt in first-party code.

**To reach 10**
- Fold `coordinator.py` / `digital_twin.py` / `monitor.py` out of `tool.mypy.overrides.ignore_errors = true` — they are in-house code, not third-party.

---

### 1.2 Code quality & typing — **7 / 10**

**Evidence.** `pyproject.toml` runs a sensible ruff ruleset (E/W/F/I/B/UP/C4/SIM/RUF). Pre-commit is wired. Dead `[tool.black]` config is gone and Dependabot no longer tracks Black while ruff-format owns formatting. But mypy is still deliberately lax: `disallow_untyped_defs = false`, `disallow_incomplete_defs = false`, `warn_unused_ignores = false`; several first-party modules still carry `ignore_errors = true`; pyright remains "advisory" in CI. Inline `# type: ignore[misc,valid-type]` on `class AggregatorNode(BaseNode):` is still appropriate.

**To reach 10**
- Flip `disallow_untyped_defs = true` and `disallow_incomplete_defs = true` in `pyproject.toml` (L213-214); fix resulting errors module-by-module.
- Remove `ignore_errors = true` from `fl_robots.coordinator`, `fl_robots.digital_twin`, `fl_robots.monitor`, `fl_robots.observability.logging`, `fl_robots.web_dashboard` (`pyproject.toml:241-252`) — add real type annotations.
- Drop `continue-on-error: true` from the pyright job (`.github/workflows/ci.yml:48`) once it's green.
- Add `ANN`, `TCH`, `PL`, `N`, `A`, `ARG`, `PTH` to the ruff select list.
- Eliminate the per-file `F401` waivers on `aggregator.py`, `robot_agent.py`, `web_dashboard.py`, `mpc_qp.py` (`pyproject.toml:167–172`) by moving the optional-dep soft-imports inside `TYPE_CHECKING` blocks or using `importlib`.

---

### 1.3 Testing — **7 / 10**

**Evidence.** 18 test files, ~1 k+ LOC. Real Hypothesis property tests (`tests/test_properties.py:58-95` covers FedAvg permutation invariance / equal-samples-is-mean / single-client identity; L133–178 covers MPC invariants). Fake-ROS harness (`tests/test_ros_fake_env.py`, `conftest.py`). FedProx runtime (`tests/test_fedprox_runtime.py:35-52`) and benchmark (`tests/test_fedprox_benchmark.py`). E2E command-flow lifecycle (`tests/test_e2e_command_flow.py`). Security-headers test (`tests/test_security_headers.py`). Playwright dashboard smoke (`web/tests/dashboard.spec.ts`). CI has a 4-version matrix (3.10/3.11/3.12/3.13) at `ci.yml:60-87` and a benchmark regression gate (`ci.yml:172-193`). Coverage gate: `fail_under = 70` (`pyproject.toml:124`).

**Gaps.** No mutation testing (`mutmut` / `cosmic-ray`). `launch_testing` coverage is one file (`src/fl_robots/test/test_aggregator_launch.py`) — nightly job even swallows failures with `|| true` (`ci.yml:169`). No flakiness budget / rerun policy; CI matrix doesn't `pytest --randomly` (no `pytest-randomly` in `dev` extras). No contract tests for the ROS custom interfaces msg/srv/action shapes. Playwright spec is a single file. Coverage gate 70 % is middling. No load/stress test of `/api/command` beyond rate-limit smoke. No Schemathesis / OpenAPI fuzzing against `/api/openapi.json` (which **is** published at `standalone_web.py:200-308`). Property tests exist but not for `DistributedMPCPlanner` collision constraints (loose `assume(… >= 0.0)` at `test_properties.py:178`).

**To reach 10**
- Add `pytest-randomly` + `pytest-xdist` to run order-independent parallel tests in CI; add a `--flake-finder` job.
- Add `mutmut` config + a nightly mutation-testing job targeting `models/simple_nn.py`, `mpc_qp.py`, `aggregator.py`; enforce ≥ 70 % mutation score on FedAvg math.
- Remove `|| true` from `ci.yml:169` and actually block on `launch_testing` failures. Expand `src/fl_robots/test/` with scenarios for robot disconnect, stale-round drop (`aggregator.py:461-466`), lifecycle transitions.
- Add `schemathesis run /api/openapi.json` as a CI job to fuzz all `/api/*` endpoints.
- Raise `fail_under` to 85 in `pyproject.toml:124`; remove `web_dashboard.py` and `scripts/benchmark.py` from the `omit` list (`pyproject.toml:103–112`) — the CHANGELOG already claims this is done, but the code still has them omitted.
- Add MPC collision-distance lower-bound property (real `assert` instead of `assume` in `tests/test_properties.py:176-178`).
- Extend Playwright spec to cover chart rendering, Socket.IO events, and history endpoints.

---

### 1.4 ML correctness (FedAvg / FedProx / non-IID / reproducibility) — **6.5 / 10**

**Evidence of correctness.** Weighted FedAvg with sample counts is correct in the benchmark path. FedProx proximal penalty is applied only to `named_parameters()` (not buffers), and the runtime client path now fails loudly on missing or shape-mismatched FedProx snapshots instead of silently degrading. `SyntheticDataGenerator` no longer relies on randomized `hash(robot_id)` seeding; deterministic seeding is threaded from an explicit seed parameter via `fl_robots/utils/determinism.py`, and `scripts/benchmark.py` uses the same helper. Multi-seed sweep is real. Dirichlet non-IID shards remain implemented and seeded.

**Real defects.**
1. **FedAvg over full `state_dict` includes BatchNorm running stats weighted by sample counts** (`models/simple_nn.py:255-266` + `aggregator.py:636` `{name: arr.tolist() for name, arr in self.global_weights.items()}`). `SimpleNavigationNet` has three `BatchNorm1d` layers (`simple_nn.py:43,47,51`) whose `running_mean` / `running_var` / `num_batches_tracked` flow through `federated_averaging` via `get_weights()` (`simple_nn.py:107-117`). Weighted-averaging BN buffers by sample count produces biased statistics; canonical fixes are per-client BN (FedBN) or averaging parameters only. This is a **genuine ML bug**, not a style point.
2. **BatchNorm buffer handling still makes reproducibility/correctness worse than it should be.** `num_batches_tracked` is an `int64` buffer — the dtype promotion in `federated_averaging` keeps `np.float32` promoted, but then writes back to an int64 tensor via `set_weights(...to(dtype=...))` → silent truncation of the weighted fractional count.
3. **Determinism is improved but not yet documented as a strong contract.** `seed_everything` now sets Python/NumPy/Torch determinism knobs from the main entry points, but the repo still lacks a crisp reproducibility contract/checksum in `docs/BENCHMARKS.md`.

**Further.** `compute_gradient_divergence` is an L2 distance, not a gradient at all (`simple_nn.py:269-292`); the naming misleads readers. No FedAdam / FedAvgM / scaffold implementations. No client selection (`min_robots`, `participation_threshold`) beyond trivial thresholds. No differential-privacy noising.

**To reach 10**
- Exclude BN buffers from weighted averaging in `models/simple_nn.py:federated_averaging` — either parameter-only averaging or FedBN (keep BN local). Update `aggregator.py:143-144`/`634-656` accordingly.
- Rename `compute_gradient_divergence` → `compute_weight_l2_drift` (or reimplement it against actual gradients).
- Add FedProx correctness tests for BN-exclusion invariance.
- Add a `FedAdam` / server-side momentum variant to `models/simple_nn.py`; wire via the `algorithm` parameter at `aggregator.py:276`.
- Document reproducibility guarantees in `docs/BENCHMARKS.md` with a reproducible run checksum.

---

### 1.5 ROS2 integration — **8 / 10**

**Evidence.** Lifecycle Node used for aggregator with real transitions (`aggregator.py:228-261`). Custom interfaces migration is **partial but working**: `from fl_robots_interfaces.msg import AggregationResult` / `.srv import …` etc. with graceful String-fallback (`aggregator.py:56-62`, `robot_agent.py:58-65`). `CUSTOM_INTERFACES` flag gates typed service registration (`aggregator.py:185-206`). QoS is deliberate: `RELIABLE + TRANSIENT_LOCAL + KEEP_LAST depth=10` for weights/global model (`aggregator.py:152-157`, `:431-436`). Callback groups separated (reentrant for subs/services, mutually-exclusive for timers) — `aggregator.py:107-110`. MultiThreadedExecutor with 4 threads (`aggregator.py:700`). ActionServer for `TrainRound` with goal/cancel response (`robot_agent.py:41-43`). Compose has per-service health checks (`compose.yaml:64-68`). Nightly `launch_testing` job (`ci.yml:145-170`).

**Gaps.** Only one `launch_testing` file (`src/fl_robots/test/test_aggregator_launch.py`) and nightly job swallows its result (`ci.yml:169` `|| true`). No `sros2` (DDS-Security / ROS2 access control) or note of DDS partitioning. No composable components / container nodes. Parameter descriptors lack `ParameterDescriptor` (ranges, description) — e.g. `aggregator._declare_parameters` at `:267-277` uses bare `declare_parameter`. No BT.cpp / behavior trees. `_perform_aggregation` runs inside a service callback without a dedicated callback group isolating long work. `weight_subscribers` is created dynamically per-robot but never torn down on disconnect. No QoS overriding via YAML. `config/params.yaml` exists but not sourced into the launch.

**To reach 10**
- Expand `src/fl_robots/test/` with 5+ `launch_testing` scenarios; drop `|| true` in `.github/workflows/ci.yml:169`.
- Add `ParameterDescriptor(description=…, floating_point_range=…)` to every `declare_parameter` call in `aggregator.py:267-277` and `robot_agent.py:_declare_parameters`.
- Add `sros2` keystore generation to `docker/Dockerfile` (ros-runtime stage) and a commented launch-file stanza in `src/fl_robots/launch/fl_system.launch.py`.
- Introduce a `DestructionDetector` to clean `self.weight_subscribers[robot_id]` on health-check inactivation (`aggregator.py:672-687`).
- Convert the aggregator + robot_agent to composable `rclcpp_components`-style container loading via a launch `ComposableNodeContainer`.
- Add a `config/qos_overrides.yaml` and wire via `--ros-args --params-file`.

---

### 1.6 MPC / control — **7 / 10**

**Evidence.** Clean formulation in docstring (`mpc_qp.py:238-264`): stacked velocities, cumsum matrix `S`, box + slew + linearised neighbour half-planes. Terminal weight and shift-and-pad warm start (`:423-430`). OSQP tolerances and `max_iter=400` (`:382-386`). Diagnostics exposed per-robot + system (`:201-234`). Grid fallback in `mpc.py`. Property tests (`test_properties.py:133-178`). Solver status handled with a safe fallback velocity (`:407-420`).

**Real issues.**
1. **`osqp.OSQP()` + `prob.setup(...)` is called every tick per robot** (`mpc_qp.py:375-386`). OSQP's warm-start benefit comes from **reusing the factorisation**; the correct pattern is one `OSQP()` per robot cached on the planner, then `prob.update(Px=…, q=…, l=…, u=…)` + `prob.warm_start(x, y)`. Current code defeats most of the speedup it claims (L140-141 comment notwithstanding).
2. **Dual warm-start is disabled** (comment at L391-392). With a cached solver, `y` can be reused when the neighbour count is stable — the current reset on any neighbour-count change is unnecessarily conservative.
3. **Neighbour keep-out uses the *current* separation direction `n_hat` for all horizon steps** (`:344-365`). This linearises around `k=0` only. For long horizons with fast motion this under-constrains — using the *predicted* neighbour trajectory direction at each `k` would be tighter.
4. **No recursive feasibility or stability guarantee.** Terminal set / terminal constraint missing; only a weight-bump (`terminal_weight`, `:84-91`). Fine for a demo, but "stability" claim in the threat model would need a control-Lyapunov certificate.

**To reach 10**
- Refactor `QPMPCPlanner._plan_robot` to cache one `osqp.OSQP` instance per robot in `self._solver_cache` and use `prob.update(...)` + `prob.warm_start(x, y)` on subsequent calls (`mpc_qp.py:375-399`). Benchmark: expect 3-5× speedup.
- Restore dual warm-start when neighbour count is unchanged.
- Linearise each keep-out half-plane against the predicted neighbour position at that step k (use `pred[k]` already in hand at L352-355).
- Add a terminal set constraint or an explicit Lyapunov-decrease certificate; document in `standalone_web.py:MPC_EXPLAINER`.
- Add a solve-time p99 regression test against the `docs/SLO.md` 25 ms budget (fixture: 6 robots, horizon 8).

---

### 1.7 Observability — **8 / 10**

**Evidence.** Dedicated `CollectorRegistry` avoids global bleed. `update_from_snapshot` now exports the documented standalone FL/MPC contract, including `fl_training_active`, `fl_aggregation_divergence`, `fl_tracking_error`, `fl_mpc_solve_time_ms`, and round-latency observations derived from real aggregation history with scrape-safe deduplication. OTEL remains opt-in. Structlog logging module exists. `docs/OBSERVABILITY.md`, `docs/RUNBOOK.md`, `docs/prometheus-rules.yml`, and `deploy/k8s/standalone.yaml` now point at the same live `fl_*` metric family. `/api/ready` still uses tick staleness.

**Gaps.** No histogram for `/api/*` endpoint latency; no Counter for HTTP requests/status codes. No exemplars linking traces ↔ metrics. No log correlation ID injected into spans. OTEL instrumentation is still thin (`/api/command` only) — aggregation, QP solve, and local training are not instrumented. `grafana-dashboard.json` still does not cover RED / USE style API views.

**To reach 10**
- Add `fl_http_request_duration_seconds{path,method,status}` Histogram and `fl_http_requests_total` Counter; record in an `after_request` hook in `standalone_web.py`.
- Add a dedicated `fl_fedavg_aggregation_duration_seconds` Histogram on the real aggregation path, not just the standalone snapshot bridge.
- Wrap `_perform_aggregation` (`aggregator.py:531`) and `_plan_robot` (`mpc_qp.py:238`) with `span(...)` context.
- Add exemplars (`prometheus_client` exemplar support) linking histogram samples to trace IDs.
- Add a `request_id` → `correlation_id` structlog bind in `standalone_web.py`.

---

### 1.8 Security — **7.5 / 10**

**Evidence.** Non-root container users UID 1000. k8s hardening is strong: `runAsNonRoot`, `readOnlyRootFilesystem`, `seccompProfile: RuntimeDefault`, `capabilities.drop: ALL`, `automountServiceAccountToken: false`. Security headers are applied to every response. Bearer-token auth now uses `hmac.compare_digest`, which closes the earlier timing-attack footgun. Sliding-window rate limiter per-IP remains. `gitleaks` pre-commit + Dependabot for pip/actions/docker remain in place. Release workflow still emits SBOM + provenance attestations and uses PyPI Trusted Publishing.

**Gaps.**
1. **CSP still includes `'unsafe-inline'` for scripts and styles** (`standalone_web.py:381-382`) — weakest CSP rung; `deploy/k8s/standalone.yaml:26-28` even documents that the bundled JS has inline handlers.
2. **No CSRF** token on `POST /api/command` (only Bearer or nothing if unset). If Bearer is *not* set (default dev mode), any page can issue cross-origin POSTs; the CSP `form-action 'self'` does nothing for `fetch()`.
3. **No container image scanning** (`trivy` / `grype`) in CI or release workflows.
4. **No CodeQL / `bandit` SAST** job.
5. **No NetworkPolicy** in `deploy/k8s/` — pod can egress anywhere.
6. **No `--mount=type=cache` / pinned `apt` versions** in `docker/Dockerfile` (reproducibility + CVE drift).
7. **DDS security (`sros2`) not configured** (acknowledged in `docs/SECURITY.md` but no migration plan).

**To reach 10**
- Inline-free JS/CSS refactor of `src/fl_robots/fl_robots/web/templates/standalone.html` + tighten CSP to `script-src 'self'; style-src 'self'` in `standalone_web.py:381-382`.
- Add CSRF protection via a double-submit cookie on `POST /api/command` and `/api/history/**` mutating calls.
- Add Trivy image scan and `actions/dependency-review-action` to `.github/workflows/ci.yml`; add `github/codeql-action` workflow.
- Add a `bandit` pre-commit hook targeting `src/`.
- Add a `NetworkPolicy` in `deploy/k8s/` restricting egress to Prometheus + DNS + nothing else.
- Pin apt packages in `docker/Dockerfile:47-56` (`pkg=version`) or switch to `chainguard/python` / `gcr.io/distroless/python3` base; drop to distroless for `standalone-runtime`.
- Stage an `sros2`-enabled launch profile documented in `docs/SECURITY.md`.

---

### 1.9 Performance & scalability — **6.5 / 10**

**Evidence.** Benchmark harness with communication-cost accounting (`scripts/benchmark.py:287-294`), multi-seed (`:311-323`), 4 checked-in baselines (`results/*.json`). Histogram of round latency (`metrics.py:80`). SLO targets `docs/SLO.md:11-13`. OSQP is sparse. MultiThreadedExecutor.

**Gaps.**
1. OSQP rebuild-per-tick (see §1.6) leaves real throughput on the floor.
2. `DistributedMPCPlanner` and `QPMPCPlanner` are **single-threaded over robots** (`mpc_qp.py:183-199`) — an obvious `ThreadPoolExecutor.map` opportunity, the per-robot QPs are independent.
3. `standalone_web` uses the Flask dev server (`docker/Dockerfile:30` `CMD ["python", "main.py", "run", ...]`) — single-threaded, no `gunicorn`/`waitress`. SLO-5 (0.5 % error rate) is optimistic under contention.
4. No async I/O — `eventlet` is only pulled in the `ros` extra. `/api/command` is sync under the dev server → head-of-line blocking.
5. `aggregator._perform_aggregation` clones every weight to Python lists via JSON (`aggregator.py:636` `arr.tolist()`) — huge amplification for larger models. No binary payload path (e.g., `ModelWeights` msg with `float32[]`).
6. No profiling artefacts (no `py-spy`, no `scalene` snapshots in `docs/`).
7. Benchmark is only MNIST + ≤ 8 clients; no scaling curve to 32 / 64 clients.

**To reach 10**
- Parallelise per-robot QP solves in `mpc_qp.py:solve_with_refs` (thread-pool of size `min(n_robots, os.cpu_count())`).
- Persist `osqp.OSQP` instances in `self._solver_cache` (see §1.6 gap #1).
- Replace Flask dev server in `docker/Dockerfile:30` with `gunicorn --worker-class gthread --workers 2 --threads 8`; update `deploy/k8s/standalone.yaml:102-108` CPU requests accordingly.
- Move JSON-over-`std_msgs/String` weight transport to the typed `ModelWeights` msg with `float32[]` arrays where `CUSTOM_INTERFACES` is available (`aggregator.py:636`).
- Add a 32-/64-client scaling benchmark under `scripts/benchmark.py --clients 32`, checkpoint in `results/`, and a `scripts/scaling_plot.py`.
- Add `docs/PROFILING.md` with `py-spy record` / `scalene` commands and a flame-graph asset.

---

### 1.10 DevOps / CI-CD — **8.5 / 10**

**Evidence.** Concurrency + cancel-in-progress. pre-commit job. Lint + mypy + pyright-advisory. 4-version matrix with GHA cache via `astral-sh/setup-uv@v3`. Docker build + in-container pytest. ROS workspace build + import smoke. Nightly `launch_testing`. Benchmark regression gate with checked-in baseline. Release workflow with SBOM + provenance + GHCR multi-arch + PyPI Trusted Publishing. Dependabot for pip/actions/docker. CI now uses `uv sync --locked`, so `uv.lock` drift is enforced in the main workflows.

**Gaps.** No image-scan job (Trivy/Grype). No CodeQL. No `actions/cache` beyond `setup-uv` caching. No matrix on OS (only ubuntu-latest) — macOS ARM64 path (where `osqp` builds are fragile, per `docs/DEVELOPMENT.md`) is untested. `ci.yml:169` still uses `|| true` for nightly tests. No automatic version bump / release notes generator (`release-please` / `release-drafter`). No `pre-commit.ci`. No stale-bot. No CLA. No branch-protection-as-code (settings-as-code via `probot/settings`).

**To reach 10**
- Add `aquasecurity/trivy-action` and `github/codeql-action` jobs to `.github/workflows/ci.yml`.
- Add `macos-14` (ARM64) to the `standalone-tests` matrix (`ci.yml:62-63`) to catch `osqp` wheel drift.
- Remove `|| true` at `ci.yml:169`.
- Add `release-drafter.yml` for auto-generated release notes (currently `release.yml:130` relies on `generate_release_notes: true`, which is fine but has no changelog sections).
- Add `settings-as-code` (`.github/settings.yml`) with required checks.
- Add `pre-commit.ci` config for auto-update PRs.

---

### 1.11 Deployment — **7.5 / 10**

**Evidence.** Multi-stage Dockerfile with distinct `standalone-runtime`, `standalone-test`, `ros-builder`, `ros-runtime` targets. k8s manifests have Namespace, ConfigMap, Deployment (rolling update, 0 unavailable), Service, ServiceMonitor, PrometheusRule, PodDisruptionBudget. Probes: startup + readiness + liveness with sensible thresholds. Compose has health checks per service. The embedded `PrometheusRule` now mirrors the richer operator docs/rule set instead of drifting behind them.

**Gaps.**
1. **No Helm chart / Kustomize overlays** — only a single concatenated YAML, hostile to env-specific tuning.
2. **No HPA** (`HorizontalPodAutoscaler`) — single replica hard-coded (`standalone.yaml:39`).
3. **No Dockerfile `HEALTHCHECK`** (only compose-level).
4. **`python:3.11-slim` base** for standalone-runtime — consider `gcr.io/distroless/python3-debian12` for smaller attack surface (acknowledged gap). Pinning is `python:3.11-slim` (unpinned minor sha).
5. `ros-runtime` **pip install as root** happens before `USER appuser` (`Dockerfile:97-100`) — fine, but `/app` is not part of appuser's home after the `chown`, so any future writeable path must be explicit.
6. **PodDisruptionBudget `minAvailable: 0`** (`standalone.yaml:195`) is effectively no PDB.
7. No `topologySpreadConstraints` / `podAntiAffinity`.
8. No Ingress manifest, no TLS terminator example, no Gateway API resource.
9. No separate `deploy/k8s/ros.yaml` — only standalone.

**To reach 10**
- Add a Helm chart under `deploy/helm/fl-robots/` (Chart.yaml, values.yaml, templates/ from `standalone.yaml`); publish to GHCR OCI in the release workflow.
- Add an HPA manifest targeting CPU 70 % and a custom `fl_http_request_duration_seconds` p95 metric.
- Add `HEALTHCHECK` directive to `docker/Dockerfile` after L30.
- Pin base images by digest (`python:3.11-slim@sha256:…`) in `docker/Dockerfile:3, 43, 70`.
- Swap `standalone-runtime` base to `gcr.io/distroless/python3-debian12:nonroot`.
- Make `PodDisruptionBudget.minAvailable` conditional on replica count ≥ 2 (`standalone.yaml:195`).
- Add `deploy/k8s/ros.yaml` with aggregator Deployment, StatefulSet per robot group, Service, Role+RoleBinding for DDS multicast.
- Add a sample Gateway-API / Ingress manifest with `cert-manager` annotations.

---

### 1.12 Documentation — **9.2 / 10**

**Evidence.** `docs/ARCHITECTURE.md` now includes Mermaid diagrams; `docs/adr/0001`–`0004` record the major design choices; `docs/RUNBOOK.md` and `docs/OBSERVABILITY.md` give operators concrete playbooks and metric→alert mappings; `docs/BENCHMARKS.md`, `docs/SECURITY.md`, `docs/SLO.md`, and `docs/DEVELOPMENT.md` remain substantive; `docs/prometheus-rules.yml` and the Grafana dashboard are wired into the docs set; OpenAPI 3.1 is still served at runtime.

**Gaps.** No migration guide for `fl_robots_interfaces` consumers. No dedicated profiling guide/flamegraph doc. `docs/api/` is still more reference-heavy than operator-heavy. The changelog still relies on curated prose rather than a structured release-note taxonomy.

**To reach 10**
- Add a migration guide for `fl_robots_interfaces` consumers.
- Add `docs/PROFILING.md` with py-spy/scalene recipes and a flamegraph asset.
- Add explicit reproducibility/checksum notes to `docs/BENCHMARKS.md`.

---

### 1.13 Developer experience — **8.5 / 10**

**Evidence.** `uv` is still the single tool. `pre-commit` config has the core Python/security hooks. `run.sh` remains the compose wrapper. Comprehensive optional-dep groups exist. `mkdocs serve` is straightforward. A root `Makefile` now gives contributors discoverable `install`/`lint`/`fmt`/`typecheck`/`test`/`bench`/`docs` shortcuts, and `CONTRIBUTING.md` documents them.

**Gaps.**
1. **No devcontainer** (`.devcontainer/devcontainer.json`) despite a Docker-first deployment story.
2. `.pre-commit-config.yaml` is still missing `mypy`, `actionlint`, `shellcheck`, `hadolint`, and `check-jsonschema`.
3. `run.sh` is still a large shell surface with no `shellcheck` in CI.
4. No `uv run dev` script wrapper; users still need to remember extras/entrypoints.
5. CODEOWNERS exists but its content/coverage was not re-audited here.
6. PR template exists but issue templates content remains unverified.

**To reach 10**
- Add `.devcontainer/devcontainer.json` building from `docker/Dockerfile` target `standalone-test`.
- Extend `.pre-commit-config.yaml` with `hadolint`, `actionlint`, `shellcheck`, `pyupgrade`, `yamllint`, `check-jsonschema` (for the OpenAPI schema in `standalone_web.py`).
- Add a `shellcheck` CI job over `run.sh` and `docker/ros_entrypoint.sh`.

---

### 1.14 Reproducibility & data management — **6.5 / 10**

**Evidence.** Seeded MNIST Dirichlet partition remains in place. Multi-seed sweep remains. Checked-in baselines under `results/`. Benchmark regression gate is still real. SBOM + provenance on release. `uv.lock` is committed and CI now installs with `uv sync --locked`. `fl_robots/utils/determinism.py` centralizes Python/NumPy/Torch seeding, and both the benchmark path and `robot_agent` use it.

**Gaps.**
1. **No model registry** — trained models under `models/` are untracked; no MLflow / DVC / W&B integration; no model card.
2. **No dataset versioning** — `data/MNIST/raw/` is bound by `torchvision.datasets.MNIST` defaults; no content hash, no DVC remote, `.dockerignore` even removes the raw bytes (per CHANGELOG:54) making reproducible container builds rely on network fetch.
3. No **`env-info.json`** artefact uploaded with each benchmark (wheel versions, CPU model, BLAS backend). Current `BenchmarkConfig` stores the config but not the environment.
4. No run ledger — baselines are still plain files in `results/` without richer provenance metadata.
5. Reproducibility is improved in code but still under-documented as an operator/research contract.

**To reach 10**
- Add DVC or `datasets` lockfile under `data/` with remote URL + content hash; wire into `docker/Dockerfile` as a build-time `--mount=type=secret` fetch or bake into a base image.
- Add a `models/registry.json` with `{name, version, sha256, training_config, metrics}` written by `scripts/benchmark.py`.
- Emit an `env_info.json` per benchmark run capturing `platform.platform()`, `torch.__version__`, `numpy.__version__`, CPU info, BLAS.
- Add a `docs/MODEL_CARD.md` template.
- Add a reproducibility/checksum section to `docs/BENCHMARKS.md`.
- Consider MLflow integration for per-run tracking with a local file backend.

---

## 2. Prioritised roadmap (Top 15)

| # | Priority | Item | Section | Est. effort |
|---|:-:|---|:-:|:-:|
| 1 | **P0** | Fix FedAvg BN-buffer weighted-average bug; parameter-only averaging + tests | 1.4 | 0.5 d |
| 2 | **P0** | Cache `osqp.OSQP()` per robot in `mpc_qp.py`; use `.update()` + `warm_start` instead of rebuild each tick | 1.6, 1.9 | 0.75 d |
| 3 | **P0** | Tighten CSP by removing `'unsafe-inline'` and add CSRF protection for `/api/command` | 1.8 | 0.75 d |
| 4 | **P0** | Add Trivy image scan + CodeQL + Bandit to `.github/workflows/ci.yml` | 1.8, 1.10 | 0.5 d |
| 5 | **P0** | Remove `|| true` from nightly ROS launch tests and expand launch scenarios | 1.3, 1.5, 1.10 | 0.75 d |
| 6 | **P1** | Flip mypy to `disallow_untyped_defs = true`; remove `ignore_errors` from first-party modules | 1.2 | 1.0 d |
| 7 | **P1** | Replace Flask dev server with `gunicorn --threads` in Docker CMD; update k8s resource requests | 1.9, 1.11 | 0.5 d |
| 8 | **P1** | Add `fl_http_request_duration_seconds` histogram, request counters, correlation IDs, and richer tracing spans | 1.7 | 0.5 d |
| 9 | **P1** | Parallelise per-robot QP solves via `ThreadPoolExecutor` in `QPMPCPlanner.solve_with_refs` | 1.9 | 0.25 d |
| 10 | **P1** | Add Helm chart + HPA + NetworkPolicy under `deploy/helm/fl-robots/` | 1.8, 1.11 | 1.0 d |
| 11 | **P1** | Add `mutmut` nightly job targeting `models/simple_nn.py`, `mpc_qp.py`, `aggregator.py`; require ≥ 70 % score | 1.3 | 0.5 d |
| 12 | **P1** | Add `pytest-randomly`, `schemathesis` fuzz job against `/api/openapi.json` | 1.3 | 0.5 d |
| 13 | **P2** | Add a migration guide for `fl_robots_interfaces` and a reproducibility/checksum appendix in `docs/BENCHMARKS.md` | 1.12, 1.14 | 0.5 d |
| 14 | **P2** | Add `.devcontainer/devcontainer.json` plus shellcheck/hadolint/actionlint/check-jsonschema hooks | 1.13 | 0.5 d |
| 15 | **P2** | Add `env_info.json` / model-registry artefacts for benchmark provenance | 1.14 | 0.5 d |

---

## 3. Quick-reference file:line index

Useful anchors cited in this audit:

- `pyproject.toml:103-112` — coverage omit list
- `pyproject.toml:124` — coverage `fail_under = 70`
- `pyproject.toml:131-164` — ruff config
- `pyproject.toml:167-172` — per-file F401 waivers
- `pyproject.toml:211-216` — mypy laxness
- `pyproject.toml:241-252` — mypy `ignore_errors` blocklist
- `.github/workflows/ci.yml:13-15` — concurrency
- `.github/workflows/ci.yml:45-55` — pyright advisory
- `.github/workflows/ci.yml:60-87` — test matrix
- `.github/workflows/ci.yml:145-170` — nightly launch_testing
- `.github/workflows/ci.yml:169` — `|| true` swallow
- `.github/workflows/ci.yml:172-193` — benchmark gate
- `src/fl_robots/fl_robots/aggregator.py:56-62` — custom-interface soft-import
- `src/fl_robots/fl_robots/aggregator.py:107-110` — callback-group split
- `src/fl_robots/fl_robots/aggregator.py:152-157` — QoS for weights
- `src/fl_robots/fl_robots/aggregator.py:185-206` — typed service registration
- `src/fl_robots/fl_robots/aggregator.py:228-261` — Lifecycle transitions
- `src/fl_robots/fl_robots/aggregator.py:267-277` — parameter declarations
- `src/fl_robots/fl_robots/aggregator.py:531` — `_perform_aggregation`
- `src/fl_robots/fl_robots/aggregator.py:636` — JSON weight encoding
- `src/fl_robots/fl_robots/aggregator.py:638-649` — FedProx wiring
- `src/fl_robots/fl_robots/aggregator.py:672-687` — health-check inactivation
- `src/fl_robots/fl_robots/aggregator.py:700` — MultiThreadedExecutor(4)
- `src/fl_robots/fl_robots/robot_agent.py` — deterministic seed parameter + FedProx client penalty path
- `src/fl_robots/fl_robots/robot_agent.py:623-633` — FedProx client penalty
- `src/fl_robots/fl_robots/models/simple_nn.py:43,47,51` — BN layers
- `src/fl_robots/fl_robots/models/simple_nn.py:107-117` — `get_weights`
- `src/fl_robots/fl_robots/models/simple_nn.py:129-131` — dtype cast
- `src/fl_robots/fl_robots/models/simple_nn.py:255-266` — `federated_averaging`
- `src/fl_robots/fl_robots/models/simple_nn.py:269-292` — mis-named divergence
- `src/fl_robots/fl_robots/mpc_qp.py:84-91` — terminal weight
- `src/fl_robots/fl_robots/mpc_qp.py:183-199` — sequential robot loop
- `src/fl_robots/fl_robots/mpc_qp.py:238-264` — QP formulation docstring
- `src/fl_robots/fl_robots/mpc_qp.py:344-365` — keep-out half-plane
- `src/fl_robots/fl_robots/mpc_qp.py:375-399` — OSQP setup-per-tick anti-pattern
- `src/fl_robots/fl_robots/mpc_qp.py:391-392` — disabled dual warm-start
- `src/fl_robots/fl_robots/mpc_qp.py:407-420` — solver fallback
- `src/fl_robots/fl_robots/mpc_qp.py:423-430` — shift-and-pad warm start
- `src/fl_robots/fl_robots/standalone_web.py:200-308` — OpenAPI schema
- `src/fl_robots/fl_robots/standalone_web.py:319-355` — auth + rate limit
- `src/fl_robots/fl_robots/standalone_web.py:326` — bearer auth comparison + rate limit
- `src/fl_robots/fl_robots/standalone_web.py:379-403` — security headers
- `src/fl_robots/fl_robots/standalone_web.py:381-382` — CSP `'unsafe-inline'`
- `src/fl_robots/fl_robots/observability/metrics.py:31` — dedicated registry
- `src/fl_robots/fl_robots/observability/metrics.py:80-85` — round latency histogram
- `src/fl_robots/fl_robots/observability/metrics.py:110-136` — snapshot bridge
- `src/fl_robots/fl_robots/utils/determinism.py` — shared deterministic seeding helper
- `docs/adr/0001-dual-mode-ros-shim.md` … `0004-lifecycle-vs-node.md` — architecture decisions
- `docs/RUNBOOK.md` / `docs/OBSERVABILITY.md` — operator docs
- `scripts/benchmark.py:30-47` — BenchmarkConfig
- `scripts/benchmark.py:174-178` — FedProx params-only penalty
- `scripts/benchmark.py:200-211` — weighted FedAvg
- `scripts/benchmark.py` — benchmark seeding + environment config
- `scripts/benchmark.py:221-228` — Dirichlet seeding
- `scripts/benchmark.py:287-294` — comms accounting
- `scripts/benchmark.py:311-323` — seed sweep
- `docker/Dockerfile:3, 43, 70` — unpinned base images
- `docker/Dockerfile:23-26, 114-119` — UID 1000 non-root
- `docker/Dockerfile:30` — Flask dev-server CMD
- `docker/Dockerfile:47-56, 78-88` — unpinned apt
- `deploy/k8s/standalone.yaml:26-28` — inline-JS comment
- `deploy/k8s/standalone.yaml:39` — replicas: 1
- `deploy/k8s/standalone.yaml:60-66, 109-113` — pod/container security context
- `deploy/k8s/standalone.yaml:79-101` — probes
- `deploy/k8s/standalone.yaml:102-108` — resource requests
- `deploy/k8s/standalone.yaml:144-187` — ServiceMonitor + PrometheusRule
- `deploy/k8s/standalone.yaml:178-187` — minimal alert
- `deploy/k8s/standalone.yaml:195` — PDB minAvailable:0
- `docs/SECURITY.md:26-28` — sros2 gap noted
- `docs/SLO.md:11-13` — latency targets
- `docs/SLO.md:30-32` — determinism caveat
- `docs/DEVELOPMENT.md:7-21` — uv workflow
- `docs/DEVELOPMENT.md:59-66` — Apple Silicon osqp caveat
- `tests/test_properties.py:58-95` — FedAvg properties
- `tests/test_properties.py:133-178` — MPC properties
- `tests/test_properties.py:176-178` — loose `assume`
- `tests/test_fedprox_runtime.py:35-52` — FedProx runtime
- `src/fl_robots/test/test_aggregator_launch.py` — only launch_testing file

---

*End of audit.*
