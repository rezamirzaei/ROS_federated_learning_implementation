"""Federated-learning datasets.

Provides reproducible client shards for MNIST / FashionMNIST with both IID
and non-IID (Dirichlet) splits. Kept dependency-light — PyTorch + torchvision
are only imported when you actually construct a loader.
"""

from __future__ import annotations

from .mnist_federated import (
    FederatedMNISTConfig,
    make_federated_mnist,
    make_federated_shards,
)

__all__ = [
    "FederatedMNISTConfig",
    "make_federated_mnist",
    "make_federated_shards",
]
