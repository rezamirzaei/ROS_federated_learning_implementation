#!/usr/bin/env python3
"""
Visualization script for training results.

Generates plots from the training results saved by the monitor node.
Run this after training completes to visualize:
- Loss convergence over rounds
- Accuracy progression
- Gradient divergence trends
- Participation statistics
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from fl_robots.results_artifacts import (
    AGGREGATION_HISTORY_JSON,
    ROBOT_METRICS_JSON,
    resolve_summary_path,
)

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Please install visualization dependencies:")
    print("  pip install matplotlib pandas numpy")
    sys.exit(1)


def load_results(results_dir: str | Path) -> dict[str, Any]:
    """Load all result files from the results directory."""
    results_path = Path(results_dir)

    data: dict[str, Any] = {}

    # Load aggregation history
    agg_file = results_path / AGGREGATION_HISTORY_JSON
    if agg_file.exists():
        with agg_file.open(encoding="utf-8") as f:
            data["aggregation"] = json.load(f)

    # Load robot metrics
    robot_file = results_path / ROBOT_METRICS_JSON
    if robot_file.exists():
        with robot_file.open(encoding="utf-8") as f:
            data["robots"] = json.load(f)

    # Load summary
    summary_file = resolve_summary_path(results_path)
    if summary_file is not None:
        with summary_file.open(encoding="utf-8") as f:
            data["summary"] = json.load(f)

    return data


def plot_convergence(data: dict, output_dir: str):
    """Plot training convergence metrics."""
    if "aggregation" not in data:
        print("No aggregation data found")
        return

    agg_data = data["aggregation"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Federated Learning Training Analysis", fontsize=14)

    # Plot 1: Gradient Divergence over rounds
    rounds = [d["round"] for d in agg_data]
    divergences = [d.get("mean_divergence", 0) for d in agg_data]

    axes[0, 0].plot(rounds, divergences, "b-o", linewidth=2, markersize=4)
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Mean Gradient Divergence")
    axes[0, 0].set_title("Gradient Divergence Over Rounds")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Participants per round
    participants = [d.get("num_participants", 0) for d in agg_data]

    axes[0, 1].bar(rounds, participants, color="green", alpha=0.7)
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Number of Participants")
    axes[0, 1].set_title("Robot Participation per Round")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Plot 3: Total samples over rounds
    samples = [d.get("total_samples", 0) for d in agg_data]
    cumulative_samples = np.cumsum(samples)

    axes[1, 0].fill_between(rounds, cumulative_samples, alpha=0.3, color="purple")
    axes[1, 0].plot(rounds, cumulative_samples, "purple", linewidth=2)
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Cumulative Training Samples")
    axes[1, 0].set_title("Training Data Volume")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Aggregation time
    agg_times = [d.get("aggregation_time", 0) * 1000 for d in agg_data]  # Convert to ms

    axes[1, 1].plot(rounds, agg_times, "r-s", linewidth=2, markersize=4)
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].set_ylabel("Aggregation Time (ms)")
    axes[1, 1].set_title("FedAvg Computation Time")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / "training_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    plt.show()


def plot_robot_metrics(data: dict, output_dir: str):
    """Plot per-robot training metrics."""
    if "robots" not in data:
        print("No robot metrics found")
        return

    robot_data = data["robots"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Per-Robot Training Metrics", fontsize=14)

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(robot_data)))

    # Plot losses
    for (robot_id, metrics), color in zip(robot_data.items(), colors):
        if not metrics:
            continue

        # Extract round-level metrics
        round_metrics = {}
        for m in metrics:
            r = m.get("round", 0)
            if r not in round_metrics and "loss" in m and m["loss"] is not None:
                round_metrics[r] = m["loss"]

        if round_metrics:
            rounds = sorted(round_metrics.keys())
            losses = [round_metrics[r] for r in rounds]
            axes[0].plot(
                rounds, losses, "-o", label=robot_id, color=color, linewidth=2, markersize=4
            )

    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Local Training Loss per Robot")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracies
    for (robot_id, metrics), color in zip(robot_data.items(), colors):
        if not metrics:
            continue

        round_metrics = {}
        for m in metrics:
            r = m.get("round", 0)
            if r not in round_metrics and "accuracy" in m and m["accuracy"] is not None:
                round_metrics[r] = m["accuracy"]

        if round_metrics:
            rounds = sorted(round_metrics.keys())
            accuracies = [round_metrics[r] for r in rounds]
            axes[1].plot(
                rounds, accuracies, "-o", label=robot_id, color=color, linewidth=2, markersize=4
            )

    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Local Training Accuracy per Robot")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / "robot_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    plt.show()


def print_summary(data: dict):
    """Print training summary statistics."""
    if "summary" not in data:
        print("No summary data found")
        return

    summary = data["summary"]

    print("\n" + "=" * 50)
    print("  TRAINING SUMMARY")
    print("=" * 50)
    print(f"  Total Rounds: {summary.get('total_rounds', 'N/A')}")
    print(f"  Total Aggregations: {summary.get('total_aggregations', 'N/A')}")
    print(f"  Training Duration: {summary.get('elapsed_time', 0) / 60:.1f} minutes")
    print(f"  Number of Robots: {len(summary.get('robots', []))}")
    print("-" * 50)

    participation = summary.get("robot_participation", {})
    if participation:
        print("  Robot Participation:")
        for robot, rounds in participation.items():
            print(f"    {robot}: {rounds} rounds")

    print("=" * 50 + "\n")


def main():
    """Main function."""
    # Default results directory
    results_dir = os.environ.get("RESULTS_DIR", "results")

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        print("Usage: python visualize.py [results_directory]")
        sys.exit(1)

    print(f"Loading results from: {results_dir}")
    data = load_results(results_dir)

    if not data:
        print("No result files found!")
        sys.exit(1)

    # Print summary
    print_summary(data)

    # Generate plots
    plot_convergence(data, results_dir)
    plot_robot_metrics(data, results_dir)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
