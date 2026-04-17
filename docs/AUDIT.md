 # fl-robots — Engineering Audit & Gap-Closing Plan

*Audit date: 2026-04-17 · Scope: `/Users/rezami/PycharmProjects/ROS` · Method: static review of `pyproject.toml`, `.github/workflows/*.yml`, `src/fl_robots/fl_robots/**`, `tests/**`, `docker/Dockerfile`, `compose.yaml`, `deploy/k8s/standalone.yaml`, `docs/**`, `scripts/**`, `mkdocs.yml`, `CHANGELOG.md`, `CONTRIBUTING.md`, `.pre-commit-config.yaml`.*

## TL;DR

This is a **well-above-average research/teaching repo** with a coherent dual-mode design, real property-based tests, a regression-gated benchmark, release provenance, SBOM, hardened k8s manifests, and an opt-in OTEL path. It is **not** production-grade: the FL math has a real correctness smell (BatchNorm running stats are weighted by sample count in `federated_averaging`), reproducibility has holes (no global torch/cudnn determinism; `SyntheticDataGenerator` uses `hash(robot_id)` which is randomized), the OSQP planner rebuilds `osqp.OSQP()` every tick defeating much of the warm-start, there are **no container CVE scans / SAST / CodeQL / mutation testing / NetworkPolicy / Helm chart / ADRs**, and mypy runs in a **relaxed, opt-in-strictness mode**.

### Aggregate score (unweighted mean across 14 sections)

`(9 + 6.5 + 7 + 6 + 8 + 7 + 7 + 7 + 6.5 + 8 + 7.5 + 8.5 + 8 + 5.5) / 14 ≈ **7.3 / 10**`

**Overall characterisation.** A polished demo-grade system with genuine engineering effort (property tests, regression gates, lifecycle nodes, SBOM, k8s hardening). Blocked from production-grade primarily by a **correctness bug in FedAvg BN handling**, **reproducibility gaps** (hash-seed, no torch determinism), **an OSQP anti-pattern** that negates most of the warm-start claim, and **missing security hygiene** (image scan, CodeQL, CSRF, CSP `unsafe-inline`, non-constant-time auth). The Top-5 P0 items below are each < 1 day and together would lift the scorecard meaningfully.

---

## 1. Section-by-section scorecard

### 1.1 Architecture & design — **9 / 10**

**Evidence.** `docs/ARCHITECTURE.md` L1–71 cleanly documents the dual-mode pattern; `fl_robots/ros_compat.py` (imported at `aggregator.py:35–46`, `robot_agent.py:40–55`) is the single seam between ROS and stub modes; `message_bus.MessageBus` mirrors ROS topic semantics in-process; planners are swappable behind a `solve(robots, leader_position)` shape (`CONTRIBUTING.md:49–54`); persistence is a thin stdlib SQLite wrapper (`persistence.py:1–60`).

**Why not 10.** No Architecture Decision Records (ADRs); the "standalone vs ROS" split, the JSON-over-`std_msgs/String` fallback protocol, and the optional custom-interfaces migration are undocumented design choices that will bite a new maintainer. The `coordinator`/`digital_twin`/`monitor` modules are typed-out of mypy (`pyproject.toml:241-252` `ignore_errors = true`), suggesting design debt.

**To reach 10**
- Create `docs/adr/0001-dual-mode-ros-shim.md`, `0002-json-string-fallback-protocol.md`, `0003-planner-pluggability.md`, `0004-lifecycle-vs-node.md` (each ≤ 1 page, MADR template).
- Add a C4 context + container diagram to `docs/ARCHITECTURE.md` (Mermaid is already enabled via pymdownx.superfences).
- Fold `coordinator.py` / `digital_twin.py` / `monitor.py` out of `tool.mypy.overrides.ignore_errors = true` — they are in-house code, not third-party.

---

### 1.2 Code quality & typing — **6.5 / 10**

**Evidence.** `pyproject.toml:131–164` runs a sensible ruff ruleset (E/W/F/I/B/UP/C4/SIM/RUF). Pre-commit is wired (`.pre-commit-config.yaml`). But mypy is deliberately lax: `disallow_untyped_defs = false`, `disallow_incomplete_defs = false`, `warn_unused_ignores = false` (`pyproject.toml:211–216`); six first-party modules carry `ignore_errors = true` (L241–252); pyright is "advisory" (`ci.yml:45–55` `continue-on-error: true`). Per-file F401 waivers on five first-party modules (`pyproject.toml:168–172`). `[tool.black]` is declared (`:126–129`) but unused — ruff-format owns formatting; this is dead config. Inline `# type: ignore[misc,valid-type]` on `class AggregatorNode(BaseNode):` (`aggregator.py:84`) is appropriate.

**To reach 10**
- Flip `disallow_untyped_defs = true` and `disallow_incomplete_defs = true` in `pyproject.toml` (L213-214); fix resulting errors module-by-module.
- Remove `ignore_errors = true` from `fl_robots.coordinator`, `fl_robots.digital_twin`, `fl_robots.monitor`, `fl_robots.observability.logging`, `fl_robots.web_dashboard` (`pyproject.toml:241-252`) — add real type annotations.
- Drop `[tool.black]` block (`pyproject.toml:126–129`) and the `black` line from `.github/dependabot.yml:36`.
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

### 1.4 ML correctness (FedAvg / FedProx / non-IID / reproducibility) — **6 / 10**

**Evidence of correctness.** Weighted FedAvg with sample counts is correct in the benchmark path (`scripts/benchmark.py:200-211` uses `torch.stack` + per-client weights). FedProx proximal penalty is applied only to `named_parameters()` (not buffers) — `scripts/benchmark.py:174-178`, correct. Runtime FedProx wiring in `aggregator.py:638-649` and `robot_agent.py:623-633` looks consistent. Multi-seed sweep is real (`benchmark.py:311-323`). Dirichlet non-IID shards implemented (`data/` module, seeded).

**Real defects.**
1. **FedAvg over full `state_dict` includes BatchNorm running stats weighted by sample counts** (`models/simple_nn.py:255-266` + `aggregator.py:636` `{name: arr.tolist() for name, arr in self.global_weights.items()}`). `SimpleNavigationNet` has three `BatchNorm1d` layers (`simple_nn.py:43,47,51`) whose `running_mean` / `running_var` / `num_batches_tracked` flow through `federated_averaging` via `get_weights()` (`simple_nn.py:107-117`). Weighted-averaging BN buffers by sample count produces biased statistics; canonical fixes are per-client BN (FedBN) or averaging parameters only. This is a **genuine ML bug**, not a style point.
2. **`SyntheticDataGenerator` seed uses `hash(robot_id) % 10000`** (`robot_agent.py:77`). Python's `hash()` is randomised unless `PYTHONHASHSEED` is pinned → different "seeds" every process start ⇒ non-reproducible non-IID splits across runs in ROS mode.
3. **No global determinism knobs.** `grep torch.manual_seed` only hits `scripts/benchmark.py:216`. Missing: `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic = True`, `numpy.random.default_rng(seed)` plumbing into the simulation. `docs/SLO.md:30-32` acknowledges "wheel-level non-determinism" but does not address `cudnn.benchmark` or `DataLoader` worker seeds (though DataLoader is not used in the benchmark path).
4. **`num_batches_tracked` is an `int64` buffer** — the dtype promotion in `federated_averaging` (`simple_nn.py:256`) does keep `np.float32` promoted, but then written back to an int64 tensor via `set_weights`' `.to(dtype=state_dict[name].dtype)` (`simple_nn.py:129-131`) → silent truncation of the weighted fractional count.

**Further.** `compute_gradient_divergence` is an L2 distance, not a gradient at all (`simple_nn.py:269-292`); the naming misleads readers. No FedAdam / FedAvgM / scaffold implementations. No client selection (`min_robots`, `participation_threshold`) beyond trivial thresholds. No differential-privacy noising.

**To reach 10**
- Exclude BN buffers from weighted averaging in `models/simple_nn.py:federated_averaging` — either parameter-only averaging or FedBN (keep BN local). Update `aggregator.py:143-144`/`634-656` accordingly.
- Rename `compute_gradient_divergence` → `compute_weight_l2_drift` (or reimplement it against actual gradients).
- Make `SyntheticDataGenerator.__init__` require an explicit seed and thread it from the ROS param `seed` (add a `seed` parameter in `robot_agent._declare_parameters`). Fix `robot_agent.py:77` to not rely on `hash()`.
- Add `fl_robots/utils/determinism.py` with `seed_everything(seed)` that sets `random`, `numpy`, `torch`, `torch.cuda`, `torch.backends.cudnn.deterministic/benchmark`, `PYTHONHASHSEED`, and call it from `scripts/benchmark.py:run_benchmark` (before L216) and from `robot_agent.__init__`.
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

### 1.7 Observability — **7 / 10**

**Evidence.** Dedicated `CollectorRegistry` avoids global bleed (`observability/metrics.py:31`). 8 metrics incl. `fl_round_latency_seconds` histogram (`:80-85`). `update_from_snapshot` is idempotent (`:110-136`). OTEL opt-in helper (`observability/tracing.py`). Structlog logging module. Prometheus rules + Grafana dashboard in `docs/`. `ServiceMonitor` + `PrometheusRule` in `deploy/k8s/standalone.yaml:144-187`. `/api/ready` readiness uses tick staleness (`standalone_web.py:_READY_STALE_S`, probe wired at `standalone.yaml:79-86`).

**Gaps.** No histogram for `/api/*` endpoint latency; no Counter for HTTP requests/status codes; no `fl_mpc_solve_time_seconds` histogram (only a gauge-via-snapshot). No exemplars linking traces ↔ metrics. No log correlation ID injected into spans. Alert rules are minimal (`standalone.yaml:178-187` is just `up == 0` for 2 m). Only one OTEL span wired (`/api/command` per CHANGELOG) — aggregation, QP solve, local training are not instrumented. No RED / USE dashboards; `grafana-dashboard.json` exists but unvetted.

**To reach 10**
- Add `fl_http_request_duration_seconds{path,method,status}` Histogram and `fl_http_requests_total` Counter; record in an `after_request` hook in `standalone_web.py:392`.
- Add `fl_mpc_solve_time_seconds{robot_id}` Histogram (per-robot) and `fl_fedavg_aggregation_duration_seconds` Histogram; update in `mpc_qp.py:194` and `aggregator.py:574-576`.
- Wrap `_perform_aggregation` (`aggregator.py:531`) and `_plan_robot` (`mpc_qp.py:238`) with `span(...)` context.
- Add exemplars (`prometheus_client` exemplar support) linking histogram samples to trace IDs.
- Expand `docs/prometheus-rules.yml` with: FL round stall, divergence spike, p95 latency SLO burn-rate multi-window alerts (2 %/1 h, 5 %/6 h).
- Add a `request_id` → `correlation_id` structlog bind in `standalone_web.py`.

---

### 1.8 Security — **7 / 10**

**Evidence.** Non-root container users UID 1000 (`docker/Dockerfile:23-26, 114-119`). k8s hardening is strong: `runAsNonRoot`, `readOnlyRootFilesystem`, `seccompProfile: RuntimeDefault`, `capabilities.drop: ALL`, `automountServiceAccountToken: false` (`standalone.yaml:60-66, 109-113`). Security headers on every response (`standalone_web.py:379-403`: CSP, X-Frame, X-Content-Type, Referrer, Permissions-Policy). Bearer-token auth with constant-string compare (`standalone_web.py:319-328`) — note this is **not constant-time** (`token.strip() != expected`). Sliding-window rate limiter per-IP (`:331-355`). `gitleaks` pre-commit + dependabot for pip/actions/docker. SBOM + provenance attestations in release workflow (`release.yml:30-39, 107-113`). PyPI Trusted Publishing (`release.yml:45-63`).

**Gaps.**
1. **CSP still includes `'unsafe-inline'` for scripts and styles** (`standalone_web.py:381-382`) — weakest CSP rung; `deploy/k8s/standalone.yaml:26-28` even documents that the bundled JS has inline handlers.
2. **Bearer compare is not `hmac.compare_digest`** (`standalone_web.py:326`) — timing-attack viable on small tokens.
3. **No CSRF** token on `POST /api/command` (only Bearer or nothing if unset). If Bearer is *not* set (default dev mode), any page can issue cross-origin POSTs; the CSP `form-action 'self'` does nothing for `fetch()`.
4. **No container image scanning** (`trivy` / `grype`) in CI or release workflows.
5. **No CodeQL / `bandit` SAST** job.
6. **No NetworkPolicy** in `deploy/k8s/` — pod can egress anywhere.
7. **No `--mount=type=cache` / pinned `apt` versions** in `docker/Dockerfile:47-56, 78-88` (reproducibility + CVE drift).
8. **DDS security (`sros2`) not configured** (acknowledged in `docs/SECURITY.md:26-28` but no migration plan).
9. `uv.lock` is checked in but CI uses `uv sync --extra …` without `--locked` — lock drift is not enforced.

**To reach 10**
- Inline-free JS/CSS refactor of `src/fl_robots/fl_robots/web/templates/standalone.html` + tighten CSP to `script-src 'self'; style-src 'self'` in `standalone_web.py:381-382`.
- Swap `token.strip() != expected` → `hmac.compare_digest(token.strip().encode(), expected.encode())` (`standalone_web.py:326`).
- Add CSRF protection via a double-submit cookie on `POST /api/command` and `/api/history/**` mutating calls.
- Add Trivy image scan and `actions/dependency-review-action` to `.github/workflows/ci.yml`; add `github/codeql-action` workflow.
- Add a `bandit` pre-commit hook targeting `src/`.
- Add a `NetworkPolicy` in `deploy/k8s/` restricting egress to Prometheus + DNS + nothing else.
- Pin apt packages in `docker/Dockerfile:47-56` (`pkg=version`) or switch to `chainguard/python` / `gcr.io/distroless/python3` base; drop to distroless for `standalone-runtime`.
- Add `--locked` to every `uv sync` in `.github/workflows/*.yml`.
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

### 1.10 DevOps / CI-CD — **8 / 10**

**Evidence.** Concurrency + cancel-in-progress (`ci.yml:13-15`). pre-commit job (`:18-26`). Lint + mypy + pyright-advisory (`:28-55`). 4-version matrix with GHA cache via `astral-sh/setup-uv@v3` (`:60-86`). Docker build + in-container pytest (`:89-118`). ROS workspace build + import smoke (`:120-143`). Nightly `launch_testing` (`:145-170`). Benchmark regression gate with checked-in baseline (`:172-193`). Release workflow with SBOM + provenance + GHCR multi-arch + PyPI Trusted Publishing (`release.yml`). Dependabot for pip/actions/docker (`.github/dependabot.yml`).

**Gaps.** No image-scan job (Trivy/Grype). No CodeQL. No `actions/cache` beyond `setup-uv` caching. No matrix on OS (only ubuntu-latest) — macOS ARM64 path (where `osqp` builds are fragile, per `docs/DEVELOPMENT.md:59-66`) is untested. `ci.yml:169` uses `|| true` for nightly tests. No automatic version bump / release notes generator (`release-please` / `release-drafter`). No `pre-commit.ci`. No stale-bot. No CLA. No branch-protection-as-code (settings-as-code via `probot/settings`). `uv.lock` is not enforced (no `--locked` flag).

**To reach 10**
- Add `aquasecurity/trivy-action` and `github/codeql-action` jobs to `.github/workflows/ci.yml`.
- Add `macos-14` (ARM64) to the `standalone-tests` matrix (`ci.yml:62-63`) to catch `osqp` wheel drift.
- Remove `|| true` at `ci.yml:169`.
- Add `release-drafter.yml` for auto-generated release notes (currently `release.yml:130` relies on `generate_release_notes: true`, which is fine but has no changelog sections).
- Add `--locked` to every `uv sync` invocation.
- Add `settings-as-code` (`.github/settings.yml`) with required checks.
- Add `pre-commit.ci` config for auto-update PRs.

---

### 1.11 Deployment — **7.5 / 10**

**Evidence.** Multi-stage Dockerfile with distinct `standalone-runtime`, `standalone-test`, `ros-builder`, `ros-runtime` targets (`docker/Dockerfile`). k8s manifests have Namespace, ConfigMap, Deployment (rolling update, 0 unavailable), Service, ServiceMonitor, PrometheusRule, PodDisruptionBudget. Probes: startup + readiness + liveness with sensible thresholds (`standalone.yaml:79-101`). Compose has health checks per service.

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

### 1.12 Documentation — **8.5 / 10**

**Evidence.** `docs/ARCHITECTURE.md` (273 lines, topics/services/actions table), `docs/BENCHMARKS.md` with reproducible command + headline table, `docs/SECURITY.md`, `docs/SLO.md`, `docs/DEVELOPMENT.md` with real troubleshooting (macOS port 5000, Apple Silicon osqp), `docs/prometheus-rules.yml`, Grafana dashboard JSON, MkDocs-Material + mkdocstrings API reference auto-deploying to GitHub Pages (`docs.yml`). OpenAPI 3.1 schema served at runtime (`standalone_web.py:OPENAPI_SCHEMA`).

**Gaps.** No ADRs (see §1.1). No runbook for incident response (e.g., "aggregation stalled at round N, do X"). No data-flow diagram as Mermaid (only ASCII art `ARCHITECTURE.md:7-23`). No migration guide for `fl_robots_interfaces` consumers. `docs/api/` references 6 pages but no observability runbook. No CHANGELOG entry for the reported `v1.0.0` release's **breaking** items (there are none listed). Typo/inconsistency: CHANGELOG says `scripts/benchmark.py` was removed from coverage omit but `pyproject.toml:103-112` still omits it.

**To reach 10**
- Add `docs/adr/` (4-6 ADRs).
- Add `docs/RUNBOOK.md` with playbooks for: aggregation stalled, readiness flapping, memory creep, OSQP infeasibility spike.
- Convert the ASCII diagram in `ARCHITECTURE.md:7-23` to Mermaid; add C4 container diagram.
- Fix the `pyproject.toml:103-112` / CHANGELOG inconsistency.
- Add `docs/OBSERVABILITY.md` mapping each metric → alert → runbook.

---

### 1.13 Developer experience — **8 / 10**

**Evidence.** `uv` as the single tool (`pyproject.toml`, `docs/DEVELOPMENT.md:7-21`). `pre-commit` config with ruff + gitleaks + standard hooks. `run.sh` shell wrapper for compose ops. Comprehensive optional-dep groups (`pyproject.toml:41-85`). `pytest-xdist` in `dev` per CHANGELOG. `mkdocs serve`-able. CONTRIBUTING.md is short and actionable.

**Gaps.**
1. **No `Makefile` / `justfile` / `tox.ini`** — every onboarding dev retypes `uv run ruff check . && uv run ruff format --check . && uv run mypy … && uv run pytest`.
2. **No devcontainer** (`.devcontainer/devcontainer.json`) despite a Docker-first deployment story.
3. `.pre-commit-config.yaml` missing: `mypy`, `pyupgrade`, `actionlint`, `shellcheck` (for `run.sh`/`docker/ros_entrypoint.sh`), `hadolint` (Dockerfile), `check-jsonschema`.
4. `run.sh` is 282 lines of shell with no `shellcheck` in CI.
5. No `uv run dev` script wrapper; users must memorise extras.
6. CODEOWNERS exists (not read but listed) — content unverified.
7. PR template exists but issue templates content not verified.

**To reach 10**
- Add `Makefile` (or `justfile`) with `install`, `lint`, `fmt`, `typecheck`, `test`, `bench`, `docs`, `image`, `k8s` targets.
- Add `.devcontainer/devcontainer.json` building from `docker/Dockerfile` target `standalone-test`.
- Extend `.pre-commit-config.yaml` with `hadolint`, `actionlint`, `shellcheck`, `pyupgrade`, `yamllint`, `check-jsonschema` (for the OpenAPI schema in `standalone_web.py`).
- Add a `shellcheck` CI job over `run.sh` and `docker/ros_entrypoint.sh`.
- Document `make <target>` shortcuts in `CONTRIBUTING.md:20-25`.

---

### 1.14 Reproducibility & data management — **5.5 / 10**

**Evidence.** Seeded MNIST Dirichlet partition (`scripts/benchmark.py:221-228`). Multi-seed sweep (`:311-323`). Checked-in baselines under `results/`. Benchmark regression gate (`ci.yml:184-188`). SBOM + provenance on release. `uv.lock` committed.

**Gaps.**
1. **No model registry** — trained models under `models/` are untracked; no MLflow / DVC / W&B integration; no model card.
2. **No dataset versioning** — `data/MNIST/raw/` is bound by `torchvision.datasets.MNIST` defaults; no content hash, no DVC remote, `.dockerignore` even removes the raw bytes (per CHANGELOG:54) making reproducible container builds rely on network fetch.
3. Python hash seed non-determinism (§1.4 #2).
4. `torch` deterministic flags never set (§1.4 #3).
5. No **`env-info.json`** artefact uploaded with each benchmark (wheel versions, CPU model, BLAS backend). Current `BenchmarkConfig` (`scripts/benchmark.py:30-47`) stores the config but not the environment.
6. No run ledger — baselines are just files in `results/` without provenance metadata.
7. `uv sync` in CI doesn't use `--locked` (uv.lock drift not enforced — §1.10).

**To reach 10**
- Add DVC or `datasets` lockfile under `data/` with remote URL + content hash; wire into `docker/Dockerfile` as a build-time `--mount=type=secret` fetch or bake into a base image.
- Add a `models/registry.json` with `{name, version, sha256, training_config, metrics}` written by `scripts/benchmark.py`.
- Emit an `env_info.json` per benchmark run capturing `platform.platform()`, `torch.__version__`, `numpy.__version__`, CPU info, BLAS.
- Add `fl_robots/utils/determinism.py:seed_everything` and call it from every entry point (see §1.4).
- Add a `docs/MODEL_CARD.md` template.
- Add `--locked` to `uv sync` invocations in CI/release.
- Consider MLflow integration for per-run tracking with a local file backend.

---

## 2. Prioritised roadmap (Top 15)

| # | Priority | Item | Section | Est. effort |
|---|:-:|---|:-:|:-:|
| 1 | **P0** | Fix FedAvg BN-buffer weighted-average bug; parameter-only averaging + tests | 1.4 | 0.5 d |
| 2 | **P0** | Add `seed_everything` (torch/cudnn/numpy/python/`PYTHONHASHSEED`); replace `hash(robot_id)` seed in `robot_agent.py:77` | 1.4, 1.14 | 0.5 d |
| 3 | **P0** | Cache `osqp.OSQP()` per robot in `mpc_qp.py`; use `.update()` + `warm_start` instead of rebuild each tick | 1.6, 1.9 | 0.75 d |
| 4 | **P0** | Use `hmac.compare_digest` for bearer-token check in `standalone_web.py:326`; tighten CSP by removing `'unsafe-inline'` | 1.8 | 0.5 d |
| 5 | **P0** | Add Trivy image scan + CodeQL + Bandit to `.github/workflows/ci.yml` | 1.8, 1.10 | 0.5 d |
| 6 | **P1** | Enforce `uv sync --locked` in all CI/release workflows | 1.10, 1.14 | 0.1 d |
| 7 | **P1** | Replace Flask dev server with `gunicorn --threads` in Docker CMD; update k8s resource requests | 1.9, 1.11 | 0.5 d |
| 8 | **P1** | Flip mypy to `disallow_untyped_defs = true`; remove `ignore_errors` from 5 first-party modules | 1.2 | 1.0 d |
| 9 | **P1** | Parallelise per-robot QP solves via `ThreadPoolExecutor` in `QPMPCPlanner.solve_with_refs` | 1.9 | 0.25 d |
| 10 | **P1** | Add `fl_http_request_duration_seconds` histogram, `fl_mpc_solve_time_seconds` histogram, exemplar linking | 1.7 | 0.5 d |
| 11 | **P1** | Add Helm chart + HPA + NetworkPolicy under `deploy/helm/fl-robots/` | 1.8, 1.11 | 1.0 d |
| 12 | **P1** | Add `mutmut` nightly job targeting `models/simple_nn.py`, `mpc_qp.py`, `aggregator.py`; require ≥ 70 % score | 1.3 | 0.5 d |
| 13 | **P1** | Add `pytest-randomly`, `schemathesis` fuzz job against `/api/openapi.json` | 1.3 | 0.5 d |
| 14 | **P2** | Introduce ADRs (`docs/adr/0001…0005`) + Mermaid C4 diagrams | 1.1, 1.12 | 0.75 d |
| 15 | **P2** | Add `Makefile` + devcontainer + shellcheck/hadolint/actionlint pre-commit hooks | 1.13 | 0.5 d |

---

## 3. Quick-reference file:line index

Useful anchors cited in this audit:

- `pyproject.toml:103-112` — coverage omit list
- `pyproject.toml:124` — coverage `fail_under = 70`
- `pyproject.toml:126-129` — dead `[tool.black]`
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
- `src/fl_robots/fl_robots/robot_agent.py:77` — `hash(robot_id) % 10000` bug
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
- `src/fl_robots/fl_robots/standalone_web.py:326` — non-constant-time compare
- `src/fl_robots/fl_robots/standalone_web.py:379-403` — security headers
- `src/fl_robots/fl_robots/standalone_web.py:381-382` — CSP `'unsafe-inline'`
- `src/fl_robots/fl_robots/observability/metrics.py:31` — dedicated registry
- `src/fl_robots/fl_robots/observability/metrics.py:80-85` — round latency histogram
- `src/fl_robots/fl_robots/observability/metrics.py:110-136` — snapshot bridge
- `scripts/benchmark.py:30-47` — BenchmarkConfig
- `scripts/benchmark.py:174-178` — FedProx params-only penalty
- `scripts/benchmark.py:200-211` — weighted FedAvg
- `scripts/benchmark.py:216` — `torch.manual_seed`
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

