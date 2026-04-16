#!/usr/bin/env python3
"""
Monitor Node — Real-time Metrics Visualization.

This node collects and visualizes training metrics from all components
of the federated learning system.

ROS2 Concepts Demonstrated:
- Multiple subscriptions with different QoS profiles
- Data aggregation and processing
- File I/O for results persistence
- Timer-based periodic operations
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

from .ros_compat import (
    DurabilityPolicy,
    HistoryPolicy,
    MultiThreadedExecutor,
    Node,
    QoSProfile,
    ReentrantCallbackGroup,
    ReliabilityPolicy,
    String,
    rclpy,
    require_ros,
)


class MonitorNode(Node):
    """
    Monitoring and Visualization Node.

    Collects metrics from:
    - Robot agents (local training progress)
    - Aggregator (global model updates)
    - Coordinator (training orchestration)

    Outputs:
    - Real-time console dashboard
    - JSON results files
    - CSV training history
    """

    def __init__(self):
        super().__init__("monitor")

        # Callback group
        self.cb_group = ReentrantCallbackGroup()

        # Declare parameters
        self.declare_parameter("output_dir", "/ros2_ws/results")
        self.declare_parameter("save_interval", 30.0)

        self.output_dir = self.get_parameter("output_dir").value
        os.makedirs(self.output_dir, exist_ok=True)

        self.get_logger().info("Initializing Training Monitor")

        # Metrics storage (bounded to prevent unbounded memory growth)
        self._max_metrics_per_robot = 500
        self.robot_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.aggregation_metrics: list[dict[str, Any]] = []
        self.coordinator_status: dict[str, Any] = {}
        self.start_time = time.time()

        # Summary statistics
        self.total_rounds = 0
        self.total_aggregations = 0
        self.robot_participation: dict[str, int] = defaultdict(int)

        # QoS profiles
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        _qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10
        )

        # Subscribers
        self.robot_status_sub = self.create_subscription(
            String,
            "/fl/robot_status",
            self.robot_status_callback,
            qos_reliable,
            callback_group=self.cb_group,
        )

        self.aggregation_sub = self.create_subscription(
            String,
            "/fl/aggregation_metrics",
            self.aggregation_callback,
            qos_reliable,
            callback_group=self.cb_group,
        )

        self.coordinator_sub = self.create_subscription(
            String,
            "/fl/coordinator_status",
            self.coordinator_callback,
            qos_reliable,
            callback_group=self.cb_group,
        )

        # Subscribe to individual robot metrics
        self.robot_metric_subs = {}

        # Timers
        self.dashboard_timer = self.create_timer(
            10.0, self.print_dashboard, callback_group=self.cb_group
        )

        self.save_timer = self.create_timer(
            self.get_parameter("save_interval").value,
            self.save_results,
            callback_group=self.cb_group,
        )

        self.get_logger().info(f"Monitor initialized. Results will be saved to {self.output_dir}")

    def robot_status_callback(self, msg: String):
        """Handle robot status messages."""
        try:
            data = json.loads(msg.data)
            robot_id = data.get("robot_id")
            msg_type = data.get("type")

            if not robot_id:
                return

            if msg_type == "registration":
                self._handle_robot_registration(robot_id, data)
            elif msg_type == "status":
                self._handle_robot_status(robot_id, data)

        except Exception as e:
            self.get_logger().error(f"Error processing robot status: {e}")

    def _handle_robot_registration(self, robot_id: str, data: dict[str, Any]):
        """Handle new robot registration."""
        self.get_logger().info(f"📡 Robot {robot_id} registered")

        # Create subscriber for this robot's metrics
        if robot_id not in self.robot_metric_subs:
            topic = f"/fl/{robot_id}/metrics"

            def callback(msg, rid=robot_id):
                self._handle_robot_metrics(rid, msg)

            self.robot_metric_subs[robot_id] = self.create_subscription(
                String,
                topic,
                callback,
                QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=50,
                ),
                callback_group=self.cb_group,
            )

    def _handle_robot_status(self, robot_id: str, data: dict[str, Any]):
        """Handle robot status update."""
        if data.get("last_loss") is not None:
            self.robot_metrics[robot_id].append(
                {
                    "round": data.get("training_round", 0),
                    "loss": data.get("last_loss"),
                    "accuracy": data.get("last_accuracy"),
                    "timestamp": data.get("timestamp", time.time()),
                }
            )
            # Trim to bounded size
            if len(self.robot_metrics[robot_id]) > self._max_metrics_per_robot:
                self.robot_metrics[robot_id] = self.robot_metrics[robot_id][
                    -self._max_metrics_per_robot :
                ]
            self.robot_participation[robot_id] = data.get("training_round", 0)

    def _handle_robot_metrics(self, robot_id: str, msg: String):
        """Handle detailed robot metrics."""
        try:
            data = json.loads(msg.data)
            if data.get("type") == "training_progress":
                self.robot_metrics[robot_id].append(
                    {
                        "round": data.get("round", 0),
                        "epoch": data.get("epoch", 0),
                        "loss": data.get("loss"),
                        "timestamp": data.get("timestamp", time.time()),
                    }
                )
        except Exception as e:
            self.get_logger().debug(f"Error processing robot metrics: {e}")

    def aggregation_callback(self, msg: String):
        """Handle aggregation metrics."""
        try:
            data = json.loads(msg.data)

            self.aggregation_metrics.append(data)
            self.total_aggregations += 1
            self.total_rounds = data.get("round", self.total_rounds)

            self.get_logger().info(
                f"🔄 Aggregation round {data.get('round')}: "
                f"{data.get('num_participants')} participants, "
                f"divergence={data.get('mean_divergence', 0):.4f}"
            )

        except Exception as e:
            self.get_logger().error(f"Error processing aggregation: {e}")

    def coordinator_callback(self, msg: String):
        """Handle coordinator status."""
        try:
            self.coordinator_status = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Error processing coordinator status: {e}")

    def print_dashboard(self):
        """Print a real-time dashboard to console."""
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("  FEDERATED LEARNING MONITOR DASHBOARD")
        print("=" * 60)
        print(f"  Elapsed Time: {elapsed / 60:.1f} minutes")
        print(f"  Total Rounds: {self.total_rounds}")
        print(f"  Total Aggregations: {self.total_aggregations}")
        print("-" * 60)

        # Coordinator status
        if self.coordinator_status:
            print(f"  Coordinator State: {self.coordinator_status.get('state', 'UNKNOWN')}")
            print(
                f"  Current Round: {self.coordinator_status.get('current_round', 0)}"
                f"/{self.coordinator_status.get('total_rounds', '?')}"
            )
            print(f"  Active Robots: {self.coordinator_status.get('active_robots', 0)}")

        print("-" * 60)
        print("  ROBOT STATUS:")

        # Robot metrics
        for robot_id, metrics in self.robot_metrics.items():
            if metrics:
                latest = metrics[-1]
                loss = latest.get("loss", "N/A")
                acc = latest.get("accuracy", "N/A")
                if isinstance(loss, float):
                    loss = f"{loss:.4f}"
                if isinstance(acc, float):
                    acc = f"{acc:.1f}%"
                print(
                    f"    {robot_id}: Loss={loss}, Acc={acc}, "
                    f"Rounds={self.robot_participation[robot_id]}"
                )

        print("-" * 60)

        # Aggregation metrics
        if self.aggregation_metrics:
            latest_agg = self.aggregation_metrics[-1]
            print(f"  LATEST AGGREGATION (Round {latest_agg.get('round', '?')}):")
            print(f"    Participants: {latest_agg.get('num_participants', 0)}")
            print(f"    Total Samples: {latest_agg.get('total_samples', 0)}")
            print(f"    Mean Divergence: {latest_agg.get('mean_divergence', 0):.4f}")

        print("=" * 60 + "\n")

    def save_results(self):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save aggregation history
        agg_file = os.path.join(self.output_dir, "aggregation_history.json")
        with open(agg_file, "w") as f:
            json.dump(self.aggregation_metrics, f, indent=2)

        # Save robot metrics
        robot_file = os.path.join(self.output_dir, "robot_metrics.json")
        with open(robot_file, "w") as f:
            json.dump(dict(self.robot_metrics), f, indent=2)

        # Save summary
        summary = {
            "start_time": self.start_time,
            "elapsed_time": time.time() - self.start_time,
            "total_rounds": self.total_rounds,
            "total_aggregations": self.total_aggregations,
            "robots": list(self.robot_metrics.keys()),
            "robot_participation": dict(self.robot_participation),
            "final_coordinator_status": self.coordinator_status,
            "saved_at": timestamp,
        }

        summary_file = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Save CSV for easy analysis
        if self.aggregation_metrics:
            csv_file = os.path.join(self.output_dir, "aggregation_history.csv")
            headers = [
                "round",
                "num_participants",
                "total_samples",
                "aggregation_time",
                "mean_divergence",
                "timestamp",
            ]
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
                writer.writeheader()
                for m in self.aggregation_metrics:
                    writer.writerow({h: m.get(h, "") for h in headers})

        self.get_logger().info(f"Results saved to {self.output_dir}")

    def generate_final_report(self) -> str:
        """Generate a final training report."""
        report_lines = [
            "=" * 60,
            "  FEDERATED LEARNING - FINAL REPORT",
            "=" * 60,
            "",
            f"Training Duration: {(time.time() - self.start_time) / 60:.1f} minutes",
            f"Total Rounds Completed: {self.total_rounds}",
            f"Total Aggregations: {self.total_aggregations}",
            f"Number of Robots: {len(self.robot_metrics)}",
            "",
            "Robot Participation:",
        ]

        for robot_id, rounds in self.robot_participation.items():
            report_lines.append(f"  - {robot_id}: {rounds} rounds")

        if self.aggregation_metrics:
            report_lines.extend(
                [
                    "",
                    "Aggregation Statistics:",
                ]
            )

            divergences = [m.get("mean_divergence", 0) for m in self.aggregation_metrics]
            participants = [m.get("num_participants", 0) for m in self.aggregation_metrics]

            report_lines.extend(
                [
                    f"  - Avg Divergence: {sum(divergences) / len(divergences):.4f}",
                    f"  - Final Divergence: {divergences[-1]:.4f}",
                    f"  - Avg Participants: {sum(participants) / len(participants):.1f}",
                ]
            )

        report_lines.extend(
            [
                "",
                "=" * 60,
            ]
        )

        return "\n".join(report_lines)


def main(args=None):
    require_ros()
    rclpy.init(args=args)

    monitor = MonitorNode()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(monitor)

    try:
        executor.spin()
    except KeyboardInterrupt:
        # Generate final report
        report = monitor.generate_final_report()
        print(report)

        # Save final results
        monitor.save_results()
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
