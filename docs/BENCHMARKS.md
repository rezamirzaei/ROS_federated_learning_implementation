# Benchmarks — Federated MNIST

All numbers below are produced by `scripts/benchmark.py` and are fully
reproducible:

```bash
uv run python scripts/benchmark.py --rounds 15 --clients 4 --alpha 0.5 \
    --samples-per-client 400 --seed 42 --output results/my_run.json
```

The benchmark trains a 3-layer MLP (784 → 128 → 64 → 10) with SGD+momentum on
MNIST shards split across clients via a Dirichlet(α) partition of the labels.
Small α ⇒ heavy non-IID (each client sees few classes); large α ⇒ near-IID.

## Headline results

Seed = 42, 400 samples per client, SGD(lr=0.05, momentum=0.9), batch=32,
1 local epoch per round. CPU.

| Experiment             | Clients | α (Dirichlet) | Rounds | Final acc | Best acc | Wall |
|------------------------|--------:|--------------:|-------:|----------:|---------:|-----:|
| Near-IID               |       4 |          10.0 |     15 | **91.18%** | 91.76%  | 0.85 s |
| Moderate non-IID       |       4 |           0.5 |     10 |   88.78%  | 88.78%  | 0.62 s |
| Heavy non-IID          |       4 |           0.1 |     15 |   83.46%  | 85.42%  | 1.00 s |
| Moderate non-IID (8)   |       8 |           0.5 |     20 | **92.97%** | 92.97%  | 2.60 s |

Observations

* Going from near-IID to heavy non-IID costs ~8 percentage points at 4
  clients — the canonical FedAvg failure mode.
* Doubling clients from 4 → 8 at α=0.5 recovers and exceeds the near-IID
  4-client number, matching prior work (FedAvg benefits from more diverse
  gradients so long as each client still has enough samples).
* All four runs complete in well under 3 seconds on a laptop CPU, so the
  benchmark is cheap enough to run in CI. See `.github/workflows/ci.yml`
  (`benchmark` job).

## Raw JSON

Full per-round data lands in `results/*.json`:

```json
{
  "config": {"rounds": 10, "clients": 4, "alpha": 0.5, "lr": 0.05, ...},
  "rounds": [
    {"round": 1, "train_loss": 1.65, "test_loss": 1.48, "test_accuracy": 66.17, ...},
    ...
  ],
  "summary": {
    "final_test_accuracy": 88.78,
    "final_test_loss": 0.366,
    "best_test_accuracy": 88.78,
    "total_wall_seconds": 0.62
  }
}
```

## Reproducing in CI

The `benchmark` job in `.github/workflows/ci.yml` runs a 3-round smoke with
4 clients and uploads `results/ci_benchmark.json` as an artifact, so
regressions in aggregation math surface as test-accuracy drops, not just as
test failures.

## Extending

* **Different model** — swap `_build_mlp` in
  `src/fl_robots/fl_robots/scripts/benchmark.py`.
* **Different dataset** — replace `make_federated_mnist` with another loader
  that returns `(list[(X, y)], (X_test, y_test))`; CIFAR-10 / FashionMNIST
  slot in with trivial changes to `src/fl_robots/fl_robots/data/`.
* **Different aggregator** — the benchmark uses a private `_fedavg_state_dicts`
  helper to keep the loop dependency-free; the production path is
  `fl_robots.models.simple_nn.federated_averaging`, which operates on numpy
  state dicts and is covered by the FedAvg properties in
  `tests/test_properties.py`.

