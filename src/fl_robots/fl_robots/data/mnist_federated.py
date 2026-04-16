"""Federated MNIST data shards with IID and Dirichlet non-IID partitioning.

Usage
-----
>>> cfg = FederatedMNISTConfig(num_clients=5, alpha=0.3, root="./data")
>>> shards = make_federated_shards(cfg)   # [(X_i, y_i), ...]

``alpha=+inf`` → fully IID.  Small ``alpha`` (e.g. 0.1) → heavily skewed,
which is the standard stress-test for FedAvg.

The module degrades gracefully: if torch/torchvision aren't installed the
public helpers raise a clear :class:`ImportError` at call time rather than
at import time, so the benchmark CLI surfaces a one-line actionable error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - type-only
    import torch

__all__ = [
    "FederatedMNISTConfig",
    "make_federated_mnist",
    "make_federated_shards",
]


@dataclass(frozen=True, slots=True)
class FederatedMNISTConfig:
    """Reproducible FL-MNIST partitioning recipe."""

    num_clients: int = 4
    alpha: float = 0.5  # Dirichlet concentration; <1 = non-IID
    samples_per_client: int = 512
    root: str = "./data"
    seed: int = 42
    flatten: bool = True  # Return (N, 784) vectors vs (N, 28, 28)
    normalize: bool = True  # Scale to [0, 1] and mean/std normalize


def _load_torchvision_mnist(root: str) -> tuple[np.ndarray, np.ndarray]:
    """Download MNIST once; return ``(images, labels)`` as numpy arrays."""
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torchvision is required to load MNIST. Install with: "
            "`uv sync --extra ml` or `pip install 'fl-robots[ml]'`."
        ) from exc

    dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    # `data` is uint8 (N, 28, 28); `targets` is int64 (N,)
    images = dataset.data.numpy().astype(np.float32)
    labels = dataset.targets.numpy().astype(np.int64)
    return images, labels


def _dirichlet_partition(
    labels: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """Standard Dirichlet label-skewed split.

    Each class's sample indices are split proportionally to a Dirichlet(α)
    draw across clients. Very small α concentrates a class on few clients,
    which is the canonical FL non-IID stress test (Hsu et al., 2019).
    """
    classes = np.unique(labels)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        # Dirichlet proportions across clients for this class.
        proportions = rng.dirichlet([alpha] * num_clients)
        # Cumulative split points.
        splits = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        chunks = np.split(cls_idx, splits)
        for client_id, chunk in enumerate(chunks):
            client_indices[client_id].extend(chunk.tolist())

    # Shuffle each client's pool so batches aren't ordered by class.
    out: list[np.ndarray] = []
    for idx_list in client_indices:
        arr = np.array(idx_list, dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out


def make_federated_shards(
    cfg: FederatedMNISTConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of ``(X, y)`` per client as plain numpy arrays.

    The per-client size is capped at ``cfg.samples_per_client`` so short runs
    and CI jobs have predictable cost.
    """
    rng = np.random.default_rng(cfg.seed)
    images, labels = _load_torchvision_mnist(cfg.root)

    if cfg.normalize:
        images = images / 255.0
        mu, sigma = 0.1307, 0.3081  # standard MNIST stats
        images = (images - mu) / sigma

    if cfg.flatten:
        images = images.reshape(images.shape[0], -1)

    partitions = _dirichlet_partition(labels, cfg.num_clients, cfg.alpha, rng)

    shards: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in partitions:
        if cfg.samples_per_client and len(idx) > cfg.samples_per_client:
            idx = idx[: cfg.samples_per_client]
        shards.append((images[idx].astype(np.float32), labels[idx].astype(np.int64)))
    return shards


def make_federated_mnist(
    cfg: FederatedMNISTConfig,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]:
    """Torch-tensor variant: returns ``(client_shards, (X_test, y_test))``.

    The test set is the full MNIST test split, used for global evaluation of
    the aggregated model after each round.
    """
    try:
        import torch
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch + torchvision required. Install with `uv sync --extra ml`."
        ) from exc

    shards_np = make_federated_shards(cfg)
    shards = [(torch.from_numpy(X), torch.from_numpy(y)) for X, y in shards_np]

    test = datasets.MNIST(
        root=cfg.root,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    X_test = test.data.numpy().astype(np.float32)
    y_test = test.targets.numpy().astype(np.int64)
    if cfg.normalize:
        X_test = X_test / 255.0
        X_test = (X_test - 0.1307) / 0.3081
    if cfg.flatten:
        X_test = X_test.reshape(X_test.shape[0], -1)
    return shards, (torch.from_numpy(X_test), torch.from_numpy(y_test))
