"""
Neural Network Models for Federated Learning.

This module contains the model architectures used by robot agents
for distributed training.  Demonstrates proper weight serialization
for federated averaging.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ObstacleAvoidanceNet",
    "SimpleNavigationNet",
    "compute_gradient_divergence",
    "federated_averaging",
]


class SimpleNavigationNet(nn.Module):
    """
    A simple MLP for robot navigation/obstacle avoidance.

    Input: Sensor readings (e.g., LIDAR distances, relative goal position)
    Output: Action probabilities (forward, left, right, stop)

    Architecture designed for clear demonstration of federated learning concepts.
    """

    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, output_dim: int = 4):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Three-layer MLP with dropout for regularization
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        self.fc_out = nn.Linear(hidden_dim // 2, output_dim)

        # Initialize weights using Xavier initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Action logits of shape (batch_size, output_dim)
        """
        # Handle single sample case for batch norm
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        if x.size(0) > 1:
            x = self.bn3(x)
        x = F.relu(x)

        x = self.fc_out(x)
        return x

    def predict(self, x: torch.Tensor) -> int:
        """Get the predicted action as an integer."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return int(torch.argmax(logits, dim=-1).item())

    def get_weights(self) -> dict[str, np.ndarray]:
        """
        Extract full model state (parameters + BN running stats) as numpy arrays.

        Returns:
            Dictionary mapping layer names to weight arrays
        """
        weights = {}
        for name, tensor in self.state_dict().items():
            weights[name] = tensor.detach().cpu().numpy()
        return weights

    def set_weights(self, weights: dict[str, np.ndarray]):
        """
        Set model weights from numpy arrays.

        Args:
            weights: Dictionary mapping layer names to weight arrays
        """
        state_dict = self.state_dict()
        for name, weight in weights.items():
            if name in state_dict:
                state_dict[name] = torch.from_numpy(np.asarray(weight)).to(
                    dtype=state_dict[name].dtype
                )
        self.load_state_dict(state_dict)

    def get_flat_weights(self) -> np.ndarray:
        """Get all trainable weights as a single flattened array."""
        weights = []
        for param in self.parameters():
            weights.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(weights)

    def set_flat_weights(self, flat_weights: np.ndarray):
        """Set trainable weights from a flattened array."""
        idx = 0
        for param in self.parameters():
            param_shape = param.shape
            param_size = int(np.prod(param_shape))
            param.data = torch.from_numpy(
                flat_weights[idx : idx + param_size].reshape(param_shape).copy()
            ).to(dtype=param.dtype)
            idx += param_size

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ObstacleAvoidanceNet(nn.Module):
    """
    Convolutional network for image-based obstacle avoidance.
    Takes depth images as input and outputs navigation commands.
    """

    def __init__(self, input_channels: int = 1, output_dim: int = 4):
        super().__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # Adaptive pooling so the network handles arbitrary spatial sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               Expected input size: (batch, 1, 64, 64)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_weights(self) -> dict[str, np.ndarray]:
        weights = {}
        for name, tensor in self.state_dict().items():
            weights[name] = tensor.detach().cpu().numpy()
        return weights

    def set_weights(self, weights: dict[str, np.ndarray]):
        state_dict = self.state_dict()
        for name, weight in weights.items():
            if name in state_dict:
                state_dict[name] = torch.from_numpy(np.asarray(weight)).to(
                    dtype=state_dict[name].dtype
                )
        self.load_state_dict(state_dict)


def federated_averaging(
    weights_list: list[dict[str, np.ndarray]], sample_counts: list[int] | None = None
) -> dict[str, np.ndarray]:
    """
    Perform Federated Averaging (FedAvg) on a list of model weights.

    FedAvg Algorithm:
    - Weighted average of model parameters
    - Weights are proportional to the number of local training samples

    Args:
        weights_list: List of weight dictionaries from each client
        sample_counts: Number of samples used for training by each client.
                      If None, equal weighting is used.

    Returns:
        Averaged weight dictionary
    """
    if not weights_list:
        raise ValueError("weights_list cannot be empty")

    n_clients = len(weights_list)

    # Use equal weights if sample_counts not provided
    if sample_counts is None:
        sample_counts = [1] * n_clients

    total_samples = sum(sample_counts)
    client_weights = [count / total_samples for count in sample_counts]

    # Initialize averaged weights with a floating dtype so integer buffers
    # (for example BatchNorm tracking counters) can participate safely.
    averaged_weights = {}
    for key in weights_list[0].keys():
        dtype = np.result_type(*(weights[key].dtype for weights in weights_list), np.float32)
        averaged_weights[key] = np.zeros(weights_list[0][key].shape, dtype=dtype)

    # Compute weighted average
    for client_idx, (weights, weight_factor) in enumerate(zip(weights_list, client_weights)):
        for key in averaged_weights:
            averaged_weights[key] += (
                np.asarray(weights[key], dtype=averaged_weights[key].dtype) * weight_factor
            )

    return averaged_weights


def compute_gradient_divergence(
    weights_list: list[dict[str, np.ndarray]], global_weights: dict[str, np.ndarray]
) -> list[float]:
    """
    Compute gradient divergence between local and global models.
    Useful for detecting non-IID data distribution.

    Args:
        weights_list: List of local model weights
        global_weights: Global model weights

    Returns:
        List of L2 distances from global model for each client
    """
    divergences = []

    for local_weights in weights_list:
        total_diff = 0.0
        for key in global_weights:
            diff = local_weights[key] - global_weights[key]
            total_diff += np.sum(diff**2)
        divergences.append(np.sqrt(total_diff))

    return divergences
