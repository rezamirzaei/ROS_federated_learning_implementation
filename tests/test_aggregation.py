#!/usr/bin/env python3
"""
Unit tests for the Federated Learning system.

Tests cover:
- Model weight serialization/deserialization
- Federated averaging algorithm
- Gradient divergence computation
- Data generation
"""

import numpy as np
import pytest
import torch
from fl_robots.models.simple_nn import (
    SimpleNavigationNet,
    compute_gradient_divergence,
    federated_averaging,
)


class TestSimpleNavigationNet:
    """Tests for SimpleNavigationNet model."""

    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)
        assert model is not None
        assert model.input_dim == 12
        assert model.hidden_dim == 64
        assert model.output_dim == 4

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)

        # Single sample
        x = torch.randn(1, 12)
        output = model(x)
        assert output.shape == (1, 4)

        # Batch
        x_batch = torch.randn(32, 12)
        output_batch = model(x_batch)
        assert output_batch.shape == (32, 4)

    def test_get_weights(self):
        """Test weight extraction."""
        model = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)
        weights = model.get_weights()

        assert isinstance(weights, dict)
        assert len(weights) > 0

        # Check all weights are numpy arrays
        for name, arr in weights.items():
            assert isinstance(arr, np.ndarray)

    def test_set_weights(self):
        """Test weight setting."""
        model1 = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)
        model2 = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)

        # Get weights from model1
        weights = model1.get_weights()

        # Set weights to model2
        model2.set_weights(weights)

        # Verify weights match
        weights2 = model2.get_weights()
        for name in weights.keys():
            np.testing.assert_array_almost_equal(weights[name], weights2[name])

    def test_flat_weights(self):
        """Test flattened weight operations."""
        model = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)

        flat = model.get_flat_weights()
        assert flat.ndim == 1
        assert len(flat) == model.count_parameters()

        # Modify and set back
        flat_modified = flat + 0.01
        model.set_flat_weights(flat_modified)

        flat_new = model.get_flat_weights()
        np.testing.assert_array_almost_equal(flat_modified, flat_new, decimal=5)

    def test_predict(self):
        """Test prediction method."""
        model = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)

        x = torch.randn(12)
        action = model.predict(x)

        assert isinstance(action, int)
        assert 0 <= action < 4

    def test_count_parameters(self):
        """Test parameter counting."""
        model = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)

        count = model.count_parameters()
        assert count > 0

        # Verify count is correct
        expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert count == expected


class TestFederatedAveraging:
    """Tests for FedAvg algorithm."""

    def test_equal_weights_averaging(self):
        """Test averaging with equal weights."""
        # Create simple weight dictionaries
        weights1 = {"layer1": np.array([1.0, 2.0, 3.0])}
        weights2 = {"layer1": np.array([3.0, 4.0, 5.0])}

        averaged = federated_averaging([weights1, weights2])

        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(averaged["layer1"], expected)

    def test_weighted_averaging(self):
        """Test averaging with different sample counts."""
        weights1 = {"layer1": np.array([1.0, 2.0])}
        weights2 = {"layer1": np.array([5.0, 6.0])}

        # weights1 has 3x more samples
        averaged = federated_averaging([weights1, weights2], sample_counts=[300, 100])

        # Expected: (1*0.75 + 5*0.25, 2*0.75 + 6*0.25) = (2.0, 3.0)
        expected = np.array([2.0, 3.0])
        np.testing.assert_array_almost_equal(averaged["layer1"], expected)

    def test_multi_layer_averaging(self):
        """Test averaging with multiple layers."""
        weights1 = {
            "fc1.weight": np.ones((4, 3)),
            "fc1.bias": np.zeros(4),
            "fc2.weight": np.ones((2, 4)) * 2,
        }
        weights2 = {
            "fc1.weight": np.ones((4, 3)) * 3,
            "fc1.bias": np.ones(4) * 2,
            "fc2.weight": np.ones((2, 4)) * 4,
        }

        averaged = federated_averaging([weights1, weights2])

        np.testing.assert_array_almost_equal(averaged["fc1.weight"], np.ones((4, 3)) * 2)
        np.testing.assert_array_almost_equal(averaged["fc1.bias"], np.ones(4))
        np.testing.assert_array_almost_equal(averaged["fc2.weight"], np.ones((2, 4)) * 3)

    def test_empty_list_raises_error(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError):
            federated_averaging([])

    def test_single_client(self):
        """Test averaging with single client returns same weights."""
        weights = {"layer1": np.array([1.0, 2.0, 3.0])}

        averaged = federated_averaging([weights])

        np.testing.assert_array_equal(averaged["layer1"], weights["layer1"])


class TestGradientDivergence:
    """Tests for gradient divergence computation."""

    def test_zero_divergence_same_weights(self):
        """Test divergence is zero when weights match global."""
        global_weights = {"layer1": np.array([1.0, 2.0, 3.0])}
        local_weights = [
            {"layer1": np.array([1.0, 2.0, 3.0])},
            {"layer1": np.array([1.0, 2.0, 3.0])},
        ]

        divergences = compute_gradient_divergence(local_weights, global_weights)

        assert len(divergences) == 2
        assert all(d == 0.0 for d in divergences)

    def test_positive_divergence(self):
        """Test divergence is positive when weights differ."""
        global_weights = {"layer1": np.array([0.0, 0.0, 0.0])}
        local_weights = [
            {"layer1": np.array([1.0, 0.0, 0.0])},  # L2 = 1.0
            {"layer1": np.array([0.0, 3.0, 4.0])},  # L2 = 5.0
        ]

        divergences = compute_gradient_divergence(local_weights, global_weights)

        assert len(divergences) == 2
        assert abs(divergences[0] - 1.0) < 1e-6
        assert abs(divergences[1] - 5.0) < 1e-6


class TestModelIntegration:
    """Integration tests for model + federated averaging."""

    def test_full_federated_round(self):
        """Test a complete federated learning round."""
        # Create 3 models
        models = [SimpleNavigationNet(input_dim=12, hidden_dim=32, output_dim=4) for _ in range(3)]

        # Simulate local training (just modify weights randomly)
        for model in models:
            for param in model.parameters():
                param.data += torch.randn_like(param.data) * 0.1

        # Extract weights
        weights_list = [m.get_weights() for m in models]
        sample_counts = [100, 150, 200]

        # Perform federated averaging
        averaged_weights = federated_averaging(weights_list, sample_counts)

        # Apply to a new model
        global_model = SimpleNavigationNet(input_dim=12, hidden_dim=32, output_dim=4)
        global_model.set_weights(averaged_weights)

        # Verify model works
        x = torch.randn(1, 12)
        output = global_model(x)
        assert output.shape == (1, 4)

    def test_convergence_simulation(self):
        """Simulate multiple rounds of federated learning."""
        # This is a simplified simulation
        num_rounds = 5
        num_clients = 3

        # Initialize global model
        global_model = SimpleNavigationNet(input_dim=12, hidden_dim=32, output_dim=4)

        divergence_history = []

        for round_num in range(num_rounds):
            # Each client starts from global model
            client_weights = []
            for _ in range(num_clients):
                client_model = SimpleNavigationNet(input_dim=12, hidden_dim=32, output_dim=4)
                client_model.set_weights(global_model.get_weights())

                # Simulate local training (random perturbation)
                for param in client_model.parameters():
                    param.data += torch.randn_like(param.data) * 0.05

                client_weights.append(client_model.get_weights())

            # Compute divergence
            divergences = compute_gradient_divergence(client_weights, global_model.get_weights())
            divergence_history.append(np.mean(divergences))

            # Aggregate
            averaged = federated_averaging(client_weights)
            global_model.set_weights(averaged)

        # Divergence should be roughly consistent (random training)
        assert len(divergence_history) == num_rounds
        assert all(d > 0 for d in divergence_history)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
