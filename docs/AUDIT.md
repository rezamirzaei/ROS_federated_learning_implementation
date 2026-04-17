# fl-robots — Engineering Audit & Gap-Closing Plan

*Audit date: 2026-04-17 (refreshed after remediation) · Scope: `/Users/rezami/PycharmProjects/ROS` · Method: live review of current source, tests, Docker/Kubernetes assets, CI workflows, and docs.*

## TL;DR

This repository is now a **strong demo / research system with several production-grade building blocks**. The largest correctness and ops gaps from the prior audit have been materially reduced.

### Closed since the previous audit

The following findings are now considered **fixed** or **fixed enough to be removed from the active gap list**:

- **FedAvg BatchNorm bug fixed**: `src/fl_robots/fl_robots/models/simple_nn.py` now excludes BN running buffers from aggregation, and `robot_agent.py` publishes trainable weights only.
- **OSQP warm-start is real now**: `src/fl_robots/fl_robots/mpc_qp.py` caches solver instances and updates workspaces instead of rebuilding the solver every tick.
- **Standalone CSRF added**: `src/fl_robots/fl_robots/standalone_web.py` now protects mutating standalone endpoints with a double-submit token when bearer auth is not configured.
- **Standalone CSP tightened**: `standalone_web.py`, `web/templates/standalone.html`, `web/static/js/standalone.js`, and `web/static/css/standalone.css` no longer rely on `unsafe-inline`.
- **HTTP observability added**: `observability/metrics.py` now exports `fl_http_requests_total`, `fl_http_request_duration_seconds`, and `fl_fedavg_aggregation_duration_seconds`.
- **Production runtime fixed**: `docker/Dockerfile` now runs the standalone app under `gunicorn` and includes a `HEALTHCHECK`; `src/fl_robots/fl_robots/wsgi.py` provides the WSGI entrypoint.
- **Security scanning added**: `.github/workflows/{codeql,trivy,dependency-review}.yml`, `ci.yml`, `.pre-commit-config.yaml`, and `bandit.yaml` add CodeQL, Trivy, dependency review, and Bandit.
- **NetworkPolicy added**: `deploy/k8s/networkpolicy.yaml` now restricts k8s traffic by default.
- **Standalone reproducibility improved**: `SimulationEngine` now accepts a deterministic seed and `create_app()` honors `FL_ROBOTS_SEED`.

### Overall score

- **Unweighted mean across 14 sections:** **8.3 / 10**
- **Weighted production-readiness score:** **8.0 / 10**

### Current characterisation

The project is now **clearly above “polished demo” quality** and has crossed into **credible pre-production engineering** for the standalone mode. The biggest remaining gaps are no longer the old P0 defects; they are now mostly:

1. **Security parity between the standalone UI and the ROS dashboard** (`web_dashboard.py` / `dashboard.html` / `dashboard.js` still lag behind the hardened standalone app).
2. **Supply-chain hardening depth** (base-image digest pinning, apt version pinning, dependency-policy maturity).
3. **Typing / CI strictness** (`mypy` remains intentionally relaxed, `pyright` is still advisory).
4. **Deployment maturity** (no Helm chart / HPA, still a single main standalone manifest).
5. **Scalability work left on the table** (per-robot QPs are still solved sequentially even though solver reuse is now fixed).

---

## 1. Section-by-section scorecard

### 1.1 Architecture & design — **9.5 / 10**

**What is strong now**
- `docs/ARCHITECTURE.md` and the ADR set in `docs/adr/` document the main design choices.
- `src/fl_robots/fl_robots/ros_compat.py` remains the central seam between ROS and standalone execution.
- Planner swapping remains clean (`mpc.py` vs `mpc_qp.py`).
- The standalone / ROS split is still coherent rather than duplicated.

**Why not 10**
- `pyproject.toml` still exempts several first-party modules from strict type coverage.
- Some runtime/security design decisions are implemented but not fully captured as ADRs yet.

**To reach 10**
- Remove the remaining first-party `mypy` `ignore_errors` overrides.
- Add ADRs for the newer security/runtime decisions (standalone CSRF, solver caching, WSGI serving).

### 1.2 Code quality & typing — **7.0 / 10**

**Improved**
- The codebase is cleaner and better guarded by tests than before.
- The recent changes landed without widening lint debt.

**Why not 10**
- `pyproject.toml` still keeps `disallow_untyped_defs = false` and `disallow_incomplete_defs = false`.
- `pyright` remains advisory in `.github/workflows/ci.yml`.
- Several first-party modules are still outside real static guarantees.

**To reach 10**
- Turn on stricter mypy flags.
- Remove the remaining `ignore_errors` blocks.
- Make `pyright` required instead of advisory.

### 1.3 Testing — **7.5 / 10**

**Improved**
- The fixes in this pass came with regression coverage for:
  - BN-safe FedAvg
  - CSRF behavior
  - HTTP metrics
  - solver-cache reuse
  - deterministic seeding
- Nightly ROS launch tests in `.github/workflows/ci.yml` no longer swallow failures with `|| true`.

**Why not 10**
- Coverage gate remains `70` in `pyproject.toml`.
- No mutation testing.
- No Schemathesis / OpenAPI fuzzing.
- ROS `launch_testing` breadth is still thin.

**To reach 10**
- Raise the coverage bar.
- Add mutation testing and API fuzzing.
- Expand `src/fl_robots/test/` beyond the current limited launch coverage.

### 1.4 ML correctness (FedAvg / FedProx / reproducibility) — **8.5 / 10**

**Improved**
- The most serious ML correctness issue is gone: BN buffers are no longer aggregated.
- `robot_agent.py` and `simple_nn.py` now align on parameter-only FL exchange.
- Standalone seeding is now configurable and deterministic through `SimulationEngine(seed=...)` and `FL_ROBOTS_SEED`.

**Why not 10**
- The helper name `compute_gradient_divergence` remains legacy-compatible even though it is really weight drift.
- Reproducibility is improved in standalone mode, but not every runtime path in the ROS side has a documented seed contract.
- More advanced FL optimizers (e.g. FedAdam / server momentum) are still absent.

**To reach 10**
- Finish the naming cleanup at all call-sites.
- Document a repository-wide reproducibility contract.
- Add at least one server-side optimizer beyond FedAvg/FedProx.

### 1.5 ROS2 integration — **8.0 / 10**

**Still strong**
- Lifecycle integration, QoS choices, callback-group separation, and custom-interface fallback are all solid.
- The ROS fake environment and ROS-focused tests continue to provide good confidence.

**Why not 10**
- Parameter descriptors are still not richly declared.
- `sros2` / DDS security remains unimplemented.
- Launch-test breadth is still limited.
- Some ROS-specific cleanup and subscriber lifecycle management remain rough.

**To reach 10**
- Add `ParameterDescriptor`s.
- Add more launch scenarios.
- Introduce an `sros2` path and document it.

### 1.6 MPC / control — **8.5 / 10**

**Improved**
- The prior OSQP anti-pattern is fixed: `mpc_qp.py` now reuses solver state and does real warm-start updates.
- Keep-out linearisation is improved with nominal-trajectory-based normals rather than a single static normal.

**Why not 10**
- Per-robot QP solves are still sequential.
- There is still no formal terminal set / stability guarantee.
- The planner is now materially better, but not yet fully optimized for scale.

**To reach 10**
- Parallelize per-robot solves.
- Add a terminal constraint / documented stability rationale.
- Add explicit solve-time regression gates aligned with `docs/SLO.md`.

### 1.7 Observability — **8.8 / 10**

**Improved**
- `observability/metrics.py` now exports HTTP RED metrics and a dedicated aggregation-duration histogram.
- Existing MPC tracking and solve-time histograms remain in place.
- Standalone metrics now match the operator story much better.

**Why not 10**
- OTEL coverage is still thin outside a few paths.
- No exemplars or deeper trace-to-metric linking yet.
- ROS dashboard mode still lags standalone mode in observability polish.

**To reach 10**
- Instrument aggregation / planner / training spans more deeply.
- Add trace correlation and exemplars.
- Bring the ROS dashboard path to the same observability standard.

### 1.8 Security — **8.6 / 10**

**Improved**
- Bearer auth uses constant-time comparison.
- Standalone mutating endpoints now have CSRF protection in open-auth mode.
- Standalone CSP is materially stricter.
- CI now includes Bandit, CodeQL, Trivy, and dependency review.
- k8s now includes a `NetworkPolicy`.

**Why not 10**
- `src/fl_robots/fl_robots/web_dashboard.py` still uses a weaker model:
  - permissive `CORS(app)`
  - `cors_allowed_origins="*"`
  - inline handlers / inline styles in `dashboard.html`
  - no CSRF parity on ROS-dashboard mutation routes
- Container images are still not digest-pinned.
- apt packages are still not version-pinned.

**To reach 10**
- Bring `web_dashboard.py`, `dashboard.html`, and `dashboard.js` to parity with the hardened standalone app.
- Pin container base images by digest.
- Pin apt package versions or move to a more locked-down base strategy.

### 1.9 Performance & scalability — **7.8 / 10**

**Improved**
- Gunicorn replaces the Flask dev server.
- OSQP reuse significantly improves real control-loop performance.
- FL payloads are smaller because robots publish trainable parameters only.

**Why not 10**
- QPs are still solved sequentially per robot.
- JSON remains the transport shape for model exchange in the fallback path.
- There is still no larger-scale benchmark curve beyond the existing benchmark scope.

**To reach 10**
- Parallelize per-robot solves.
- Prefer typed/binary weight transport when custom interfaces are available.
- Add larger-client scaling benchmarks.

### 1.10 DevOps / CI-CD — **8.7 / 10**

**Improved**
- CI now includes Bandit.
- Dedicated `CodeQL`, `Trivy`, and `dependency-review` workflows are present.
- The nightly ROS launch test now fails correctly when broken.

**Why not 10**
- `pyright` is still advisory.
- The matrix is still Linux-only for the main standalone test job.
- There is still room for more release / settings automation.

**To reach 10**
- Make `pyright` required.
- Add macOS ARM coverage.
- Add repository settings-as-code / release automation polish.

### 1.11 Deployment — **8.2 / 10**

**Improved**
- `docker/Dockerfile` now has a production WSGI path and health check.
- `deploy/k8s/networkpolicy.yaml` closes a major deployment-security gap.

**Why not 10**
- Still no Helm chart or overlays.
- No HPA.
- Base images are still tag-pinned, not digest-pinned.
- `deploy/k8s/standalone.yaml` is still the main deployment story.

**To reach 10**
- Add Helm / Kustomize.
- Add HPA.
- Pin image digests.

### 1.12 Documentation — **9.1 / 10**

**Improved**
- The codebase now matches more of the operator/security story than before.
- Previous docs additions (ADRs, RUNBOOK, OBSERVABILITY) remain valuable.

**Why not 10**
- The audit itself had drifted and needed this refresh.
- Some new runtime/security choices are implemented faster than they are documented in the supporting docs.

**To reach 10**
- Keep `docs/AUDIT.md`, `docs/SECURITY.md`, and deployment docs in lockstep with the code.
- Add a short “current security posture” delta summary to `docs/SECURITY.md`.

### 1.13 Developer experience — **8.4 / 10**

**Improved**
- Pre-commit now includes Bandit.
- Security tooling is more visible in local and CI flows.

**Why not 10**
- Still no devcontainer.
- Shell / YAML / Docker linting coverage is still incomplete.
- Some workflows remain more mature than the local DX story.

**To reach 10**
- Add devcontainer support.
- Add shellcheck / hadolint / yamllint / actionlint to local hooks and CI.

### 1.14 Reproducibility & data management — **7.8 / 10**

**Improved**
- Standalone mode now supports explicit seeded runs end-to-end.
- Benchmark reproducibility was already stronger than the interactive runtime and is now better aligned.

**Why not 10**
- There is still no run ledger / environment manifest per benchmark artifact.
- No model registry.
- No dataset versioning system.

**To reach 10**
- Emit environment metadata for benchmark outputs.
- Add a simple model registry / manifest.
- Add dataset locking/versioning.

---

## 2. Updated roadmap (remaining work only)

## P0

1. **Harden the ROS dashboard path to match standalone security**
   - Files: `src/fl_robots/fl_robots/web_dashboard.py`, `web/templates/dashboard.html`, `web/static/js/dashboard.js`
   - Remove inline handlers/styles, tighten CSP, add CSRF/auth parity, restrict CORS/Socket.IO origins.

2. **Pin container/runtime supply chain inputs**
   - Files: `docker/Dockerfile`
   - Pin base images by digest and pin apt package versions where feasible.

## P1

3. **Make type-checking materially stricter**
   - Files: `pyproject.toml`, `.github/workflows/ci.yml`, first-party modules currently ignored by mypy
   - Reduce or eliminate `ignore_errors`, promote `pyright` from advisory.

4. **Parallelize per-robot QP solves**
   - Files: `src/fl_robots/fl_robots/mpc_qp.py`
   - The solver reuse fix is already in place; parallelism is now the next obvious performance step.

5. **Expand deployment packaging**
   - Files: `deploy/helm/**` or `deploy/k8s/**`
   - Add Helm / HPA / environment overlays.

6. **Expand advanced testing**
   - Add mutation testing and OpenAPI fuzzing.
   - Expand ROS `launch_testing` scenarios.

## P2

7. **Add reproducibility artefacts**
   - benchmark env manifests, model registry, dataset locking

8. **Further observability / tracing depth**
   - trace exemplars, broader OTEL span coverage

---

## 3. Change log vs the previous audit

### Findings removed from the active gap list

These are no longer open audit items:

- FedAvg BN-buffer averaging defect
- OSQP rebuild-per-tick defect
- standalone missing CSRF
- standalone CSP `unsafe-inline`
- missing HTTP request latency / request count metrics
- missing aggregation-duration histogram
- Flask dev server in container CMD
- missing Bandit / CodeQL / Trivy / dependency review
- missing NetworkPolicy
- nightly ROS launch tests swallowing failures

### Findings still open

The highest-value remaining issues are now:

- ROS dashboard security parity
- strict typing / stronger CI enforcement
- image digest / apt pinning
- Helm / HPA / packaging maturity
- sequential multi-robot QP solves
- mutation/API fuzzing / broader ROS launch coverage

---

## 4. Bottom line

This repository is no longer blocked by the original headline defects. The audit should now focus on **parity, hardening, and scale**, not on the old correctness bugs.

If another pass is taken immediately, the best next target is **bringing the ROS dashboard security model up to the same standard as the hardened standalone UI**.
