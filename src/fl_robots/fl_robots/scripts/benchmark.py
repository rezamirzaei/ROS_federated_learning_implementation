"""Reproducible FedAvg benchmark on MNIST with Dirichlet non-IID splits.

Usage::

    uv run python scripts/benchmark.py --rounds 20 --clients 4 --alpha 0.3
    uv run python scripts/benchmark.py --help

Outputs a JSON report to ``--output`` (default ``results/benchmark.json``) and
prints a human-readable summary table. The result file is what
``docs/BENCHMARKS.md`` and the CI job consume.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ── Lightweight argparse-only flow so `--help` works even if torch is missing.


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    rounds: int = 20
    clients: int = 4
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.05
    alpha: float = 0.5  # Dirichlet concentration, small=non-IID
    samples_per_client: int = 512
    seed: int = 42
    output: str = "results/benchmark.json"
    data_root: str = "./data"
    device: str = "cpu"
    #: Algorithm — "fedavg" is the McMahan 2017 baseline, "fedprox" adds a
    #: proximal term μ/2·‖w - w_global‖² to each client's local objective
    #: (Li et al., 2020). FedProx is the canonical non-IID fix.
    algorithm: str = "fedavg"
    #: μ coefficient for FedProx. Ignored for FedAvg.
    proximal_mu: float = 0.01
    #: Number of random seeds to sweep. When > 1, reports mean ± std across
    #: seeds in the output JSON under ``summary.seeds``.
    num_seeds: int = 1


@dataclass(frozen=True, slots=True)
class RoundResult:
    round: int
    train_loss: float
    test_loss: float
    test_accuracy: float
    elapsed_seconds: float


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    p = argparse.ArgumentParser(
        prog="fl-robots-benchmark",
        description="Reproducible MNIST FedAvg benchmark (standalone, no ROS).",
    )
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--clients", type=int, default=4)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration; small=non-IID, ≥10=near-IID",
    )
    p.add_argument("--samples-per-client", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="results/benchmark.json")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument(
        "--algorithm",
        type=str,
        default="fedavg",
        choices=("fedavg", "fedprox"),
        help="Federated optimisation algorithm. fedprox adds a proximal term.",
    )
    p.add_argument(
        "--proximal-mu",
        type=float,
        default=0.01,
        help="μ coefficient for FedProx (ignored for FedAvg).",
    )
    p.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="If > 1, sweep [seed, seed+1, …] and report mean ± std.",
    )
    args = p.parse_args(argv)
    return BenchmarkConfig(**vars(args))


def _load_torch():
    """Import torch lazily and return the module, or exit with a friendly msg."""
    try:
        import torch
        import torch.nn as nn  # noqa: F401
    except ImportError:  # pragma: no cover
        sys.stderr.write(
            "Error: torch is not installed. Install the ML extras with:\n"
            "  uv sync --extra ml\n"
            "or:  pip install 'fl-robots[ml]'\n"
        )
        sys.exit(2)
    return torch


def _build_mlp(torch, *, input_dim: int = 784, num_classes: int = 10):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, num_classes),
    )


def _local_train(
    torch,
    model,
    X,
    y,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    global_state: dict | None = None,
    proximal_mu: float = 0.0,
):
    """Local SGD. When ``proximal_mu > 0`` and ``global_state`` is supplied,
    appends the FedProx proximal penalty ``μ/2·‖w - w_global‖²`` to the loss
    so the optimizer pulls local weights back toward the round's global model.
    This is the canonical non-IID fix from Li et al., 2020.
    """
    import torch.nn.functional as F

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Snapshot global params for the proximal term. Detach so autograd does
    # not try to propagate gradients back into the (frozen) server model.
    global_params: list | None = None
    if proximal_mu > 0.0 and global_state is not None:
        global_params = [
            global_state[name].detach().to(device)
            for name, _ in model.named_parameters()
        ]

    X = X.to(device)
    y = y.to(device)
    n = X.shape[0]
    total_loss = 0.0
    total_batches = 0
    for _epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(X[idx])
            loss = F.cross_entropy(logits, y[idx])
            if global_params is not None:
                prox = 0.0
                for (_, p_local), p_global in zip(model.named_parameters(), global_params):
                    prox = prox + ((p_local - p_global) ** 2).sum()
                loss = loss + 0.5 * proximal_mu * prox
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1
    return total_loss / max(total_batches, 1)


def _evaluate(torch, model, X, y, *, device: str) -> tuple[float, float]:
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = float(F.cross_entropy(logits, y).item())
        pred = logits.argmax(dim=-1)
        acc = float((pred == y).float().mean().item() * 100.0)
    return loss, acc


def _fedavg_state_dicts(torch, state_dicts, weights):
    """Weighted average of a list of state dicts."""
    total = sum(weights)
    weights = [w / total for w in weights]
    out = {}
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        w_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).view(
            [-1] + [1] * (stacked.dim() - 1)
        )
        out[key] = (stacked * w_tensor).sum(dim=0).to(state_dicts[0][key].dtype)
    return out


def run_benchmark(cfg: BenchmarkConfig) -> dict[str, Any]:
    torch = _load_torch()
    torch.manual_seed(cfg.seed)

    # Deferred so `--help` works without torch.
    from fl_robots.data import FederatedMNISTConfig, make_federated_mnist

    mnist_cfg = FederatedMNISTConfig(
        num_clients=cfg.clients,
        alpha=cfg.alpha,
        samples_per_client=cfg.samples_per_client,
        root=cfg.data_root,
        seed=cfg.seed,
    )
    client_shards, (X_test, y_test) = make_federated_mnist(mnist_cfg)

    global_model = _build_mlp(torch).to(cfg.device)

    per_round: list[RoundResult] = []
    total_start = time.perf_counter()

    for round_idx in range(1, cfg.rounds + 1):
        round_start = time.perf_counter()

        # Each client starts from the current global weights.
        global_state = global_model.state_dict()
        client_states = []
        client_sizes = []
        round_train_loss = 0.0

        for X_i, y_i in client_shards:
            client_model = _build_mlp(torch).to(cfg.device)
            client_model.load_state_dict(global_state)
            loss_i = _local_train(
                torch,
                client_model,
                X_i,
                y_i,
                epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                device=cfg.device,
                global_state=global_state if cfg.algorithm == "fedprox" else None,
                proximal_mu=cfg.proximal_mu if cfg.algorithm == "fedprox" else 0.0,
            )
            round_train_loss += loss_i * X_i.shape[0]
            client_states.append(client_model.state_dict())
            client_sizes.append(X_i.shape[0])

        # Weighted aggregation (true FedAvg).
        avg_state = _fedavg_state_dicts(torch, client_states, client_sizes)
        global_model.load_state_dict(avg_state)

        total_samples = sum(client_sizes) or 1
        train_loss = round_train_loss / total_samples
        test_loss, test_acc = _evaluate(torch, global_model, X_test, y_test, device=cfg.device)
        elapsed = time.perf_counter() - round_start

        rr = RoundResult(
            round=round_idx,
            train_loss=round(train_loss, 6),
            test_loss=round(test_loss, 6),
            test_accuracy=round(test_acc, 3),
            elapsed_seconds=round(elapsed, 4),
        )
        per_round.append(rr)
        print(
            f"round={rr.round:>3}  train_loss={rr.train_loss:.4f}  "
            f"test_loss={rr.test_loss:.4f}  test_acc={rr.test_accuracy:6.2f}%  "
            f"elapsed={rr.elapsed_seconds:.2f}s",
            flush=True,
        )

    wall = time.perf_counter() - total_start
    return {
        "config": asdict(cfg),
        "rounds": [asdict(r) for r in per_round],
        "summary": {
            "final_test_accuracy": per_round[-1].test_accuracy if per_round else None,
            "final_test_loss": per_round[-1].test_loss if per_round else None,
            "best_test_accuracy": max((r.test_accuracy for r in per_round), default=None),
            "total_wall_seconds": round(wall, 2),
        },
    }


def run_multi_seed(cfg: BenchmarkConfig) -> dict[str, Any]:
    """Run :func:`run_benchmark` across ``cfg.num_seeds`` random seeds.

    Produces mean ± std summary statistics — the right way to report FL
    numbers given the high variance of non-IID training. Seeds are the
    deterministic sequence ``[cfg.seed, cfg.seed+1, …]``.
    """
    from dataclasses import replace
    from statistics import mean, stdev

    seed_runs = []
    for offset in range(cfg.num_seeds):
        sub_cfg = replace(cfg, seed=cfg.seed + offset, num_seeds=1)
        print(f"\n── Seed {sub_cfg.seed} ({offset + 1}/{cfg.num_seeds}) ──")
        seed_runs.append(run_benchmark(sub_cfg))

    finals = [r["summary"]["final_test_accuracy"] for r in seed_runs]
    bests = [r["summary"]["best_test_accuracy"] for r in seed_runs]
    walls = [r["summary"]["total_wall_seconds"] for r in seed_runs]

    def _stat(xs: list[float]) -> dict[str, float]:
        return {
            "mean": round(mean(xs), 3),
            "std": round(stdev(xs), 3) if len(xs) > 1 else 0.0,
            "min": round(min(xs), 3),
            "max": round(max(xs), 3),
            "values": [round(x, 3) for x in xs],
        }

    return {
        "config": asdict(cfg),
        "seeds": [cfg.seed + o for o in range(cfg.num_seeds)],
        "per_seed": seed_runs,
        "summary": {
            "final_test_accuracy": _stat(finals),
            "best_test_accuracy": _stat(bests),
            "total_wall_seconds": _stat(walls),
        },
    }


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    result = run_multi_seed(cfg) if cfg.num_seeds > 1 else run_benchmark(cfg)

    out_path = Path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
