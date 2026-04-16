# Contributing to fl-robots

Thanks for your interest! The project stays small on purpose; please keep PRs
focused and reversible.

## Development setup

```bash
uv sync --extra ml --extra dev --extra viz
uv run pre-commit install         # optional but recommended
uv run python -m pytest tests/ -v
uv run python scripts/benchmark.py --rounds 3
```

## Quality gates

All of the following must pass locally before opening a PR. The same checks
run in CI (`.github/workflows/ci.yml`).

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/fl_robots/fl_robots
uv run pytest tests/
```

If you add a new module, please:

* Add a module docstring explaining its single responsibility.
* Prefer Pydantic `BaseModel` for structured data crossing boundaries.
* Cover new public behaviour with at least one example-based test; add a
  Hypothesis property if you can express one.

## Adding a command

1. Append the new name to `COMMAND_NAMES` in
   `src/fl_robots/fl_robots/controller.py`.
2. Handle it inside `SimulationEngine._handle_command_event`.
3. If it needs special UI treatment, update
   `src/fl_robots/fl_robots/web/templates/standalone.html`.

## Adding a dataset

1. Add a module under `src/fl_robots/fl_robots/data/` that exposes:
   * `make_federated_shards(cfg) -> list[tuple[np.ndarray, np.ndarray]]`
   * optionally `make_federated_<name>(cfg) -> (shards, (X_test, y_test))`
2. Wire it into `scripts/benchmark.py` behind a `--dataset` flag.

## Adding a planner

Mirror the public shape of
`fl_robots.mpc.DistributedMPCPlanner.solve(robots, leader_position)`. Place
it in its own module and make the `SimulationEngine` constructor accept a
planner argument.

## Commit messages

Use present-tense imperative ("add benchmark", "fix stale-round guard"). The
first line should be <= 72 chars.

## Licensing

By contributing you agree that your contributions will be licensed under the
MIT License (see `LICENSE`).

