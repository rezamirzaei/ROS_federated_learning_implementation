"""Tests for FedProx and the multi-seed benchmark orchestrator.

These exercise the benchmark script *without* actually downloading MNIST
by swapping in a tiny synthetic shard set via monkeypatching.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from fl_robots.scripts import benchmark as bm


def _fake_federated(cfg):
    """Replace the MNIST loader with 3 tiny random shards + test split."""
    import torch

    torch.manual_seed(0)
    client_shards = []
    for _ in range(cfg.num_clients):
        X = torch.rand(32, 784)
        y = torch.randint(0, 10, (32,))
        client_shards.append((X, y))
    X_test = torch.rand(64, 784)
    y_test = torch.randint(0, 10, (64,))
    return client_shards, (X_test, y_test)


@pytest.fixture(autouse=True)
def stub_mnist(monkeypatch):
    monkeypatch.setattr(
        "fl_robots.data.make_federated_mnist", _fake_federated, raising=True
    )


def _cfg(**over) -> bm.BenchmarkConfig:
    defaults = {
        "rounds": 2,
        "clients": 3,
        "local_epochs": 1,
        "batch_size": 16,
        "lr": 0.05,
        "alpha": 0.5,
        "samples_per_client": 32,
        "seed": 0,
        "output": "/dev/null",
        "data_root": "./data",
        "device": "cpu",
        "algorithm": "fedavg",
        "proximal_mu": 0.01,
        "num_seeds": 1,
    }
    defaults.update(over)
    return bm.BenchmarkConfig(**defaults)


def test_fedavg_benchmark_runs_and_produces_per_round_stats():
    result = bm.run_benchmark(_cfg())
    assert len(result["rounds"]) == 2
    for r in result["rounds"]:
        # Accuracy should be in [0, 100]; with random data near 10%.
        assert 0.0 <= r["test_accuracy"] <= 100.0
    assert result["summary"]["final_test_accuracy"] is not None


def test_fedprox_benchmark_runs_without_error():
    """FedProx path must train without NaNs and produce valid output."""
    result = bm.run_benchmark(_cfg(algorithm="fedprox", proximal_mu=0.05))
    final = result["summary"]["final_test_accuracy"]
    assert final is not None
    # Sanity: no NaN / inf leaked out.
    import math as _m

    for r in result["rounds"]:
        assert _m.isfinite(r["train_loss"])
        assert _m.isfinite(r["test_loss"])
        assert _m.isfinite(r["test_accuracy"])


def test_multi_seed_reports_mean_and_std(tmp_path):
    out = tmp_path / "multi.json"
    cfg = _cfg(num_seeds=3, output=str(out))
    result = bm.run_multi_seed(cfg)
    assert result["seeds"] == [0, 1, 2]
    assert len(result["per_seed"]) == 3
    summary = result["summary"]["final_test_accuracy"]
    assert "mean" in summary and "std" in summary
    assert len(summary["values"]) == 3
    # std should be >= 0 and reported to 3 decimals.
    assert summary["std"] >= 0.0


def test_cli_main_writes_json_file(tmp_path):
    out = tmp_path / "bench.json"
    argv = [
        "--rounds", "1",
        "--clients", "2",
        "--samples-per-client", "32",
        "--batch-size", "16",
        "--output", str(out),
    ]
    rc = bm.main(argv)
    assert rc == 0
    data = json.loads(Path(out).read_text())
    assert "summary" in data
    assert "rounds" in data

