# Development Guide

This document captures the local-dev trouble-shooting knowledge you'd
otherwise only get by pairing with a maintainer.

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.13 | 3.11 is the primary dev target |
| [uv](https://github.com/astral-sh/uv) | ≥ 0.4 | Replaces pip/venv/pip-tools |
| Docker | ≥ 24 | For ROS & multi-arch build validation |
| ROS 2 Humble | optional | Only required for the `ros` profile |

## Bootstrap

```bash
uv sync --extra dev --extra ml --extra viz --extra qp
uv run pre-commit install
uv run pytest
```

## Optional-dependency matrix

| Extra | Purpose | Notable wheels |
|---|---|---|
| `ml`   | Federated learning, MNIST loader, aggregation | `torch` (CPU), `torchvision`, `scikit-learn` |
| `qp`   | OSQP-backed MPC planner                       | `osqp`, `scipy` |
| `ros`  | Full ROS 2 Socket.IO dashboard                | `flask-socketio`, `eventlet`, `matplotlib` |
| `viz`  | Offline benchmark plotting                    | `matplotlib`, `pandas` |
| `dev`  | Lint + test + type tooling                    | `pytest`, `ruff`, `mypy`, `hypothesis`, `pytest-xdist` |

Install several extras at once: `uv sync --extra ml --extra qp --extra dev`.

## Common pitfalls

### macOS: port 5000 already in use

Apple's ControlCenter binds port 5000 on recent macOS releases. Either:

```bash
python main.py run --port 5050
# or
FL_ROBOTS_PORT=5050 python main.py run
```

or disable _Receiver_ in _System Settings → General → AirDrop & Handoff →
AirPlay Receiver_.

### `torch` wheel is wrong architecture / too large

Install the CPU-only wheel explicitly (the Dockerfiles do this already):

```bash
uv pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.0.0,<=2.2.2" "torchvision>=0.15.0,<=0.17.2"
```

### `osqp` build fails on Apple Silicon

Upgrade pip tooling and build tools, then force a source build:

```bash
xcode-select --install
uv pip install --no-binary osqp osqp
```

On Linux, install `libomp-dev` first.

### Ruff / mypy disagree with PyCharm

Both the IDE and CI run against `pyproject.toml`. If the IDE underlines
something the CI accepts, check `[tool.ruff.lint.ignore]` and
`[tool.mypy]` — they drive the source of truth.

### pre-commit hook too slow

```bash
SKIP=gitleaks uv run pre-commit run --all-files
```

Skips the gitleaks history scan; still runs ruff/format/yaml/toml/eof/etc.

## Running the test matrix

```bash
uv run pytest                          # full suite
uv run pytest tests/test_mpc_observability.py -k qp  # targeted
uv run pytest -n auto                  # parallel (requires pytest-xdist)
uv run pytest --hypothesis-show-statistics tests/test_properties.py
```

## Running the ROS 2 build locally

Use the Docker path — it guarantees a reproducible ROS 2 environment:

```bash
docker build -f docker/Dockerfile --target ros-runtime -t fl-robots-ros:dev .
docker run --rm -it fl-robots-ros:dev bash
# inside container:
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
ros2 launch fl_robots fl_system.launch.py
```

## Benchmarking

```bash
uv run python scripts/benchmark.py \
    --rounds 15 --clients 4 --samples-per-client 400 \
    --dirichlet-alpha 0.5 --num-seeds 3 \
    --output results/my_benchmark.json

uv run python scripts/visualize.py results/my_benchmark.json
```

## Release workflow

1. Bump version in `pyproject.toml` and `package.xml`.
2. Add a new `[X.Y.Z]` section in `CHANGELOG.md`.
3. `git tag vX.Y.Z && git push --tags`.
4. GitHub Actions publishes PyPI + Docker images (when the release workflow
   is enabled).
