"""Helpers for reproducible execution across standalone and ROS entrypoints."""

from __future__ import annotations

import hashlib
import os
import random

__all__ = ["derive_seed", "seed_everything"]


def derive_seed(label: str, base_seed: int) -> int:
    """Derive a stable integer seed from a string label and shared base seed."""
    digest = hashlib.blake2s(f"{base_seed}:{label}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def seed_everything(seed: int) -> None:
    """Best-effort deterministic seeding for stdlib, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - optional dependency
        pass

    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
