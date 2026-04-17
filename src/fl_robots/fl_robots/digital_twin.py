#!/usr/bin/env python3
"""
Digital Twin Visualization Node.

This node provides a real-time 2D visualization of the federated learning system,
showing:
- Robot positions and states
- Training progress per robot
- Global model convergence
- Network topology

Uses matplotlib for visualization with ROS2 integration.
"""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

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
)

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend for Docker
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class RobotVisualState(BaseModel):
    """Visual state of a robot in the digital twin."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    robot_id: str
    position: tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0
    color: str = "blue"
    loss: float = 1.0
    accuracy: float = 0.0
    is_training: bool = False
    last_update: float = 0.0
    rounds_completed: int = 0


class SystemVisualState(BaseModel):
    """Overall system visual state."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    robots: dict[str, RobotVisualState] = Field(default_factory=dict)
    global_round: int = 0
    total_aggregations: int = 0
    mean_divergence: float = 1.0
    coordinator_state: str = "IDLE"
    aggregator_active: bool = True


class DigitalTwinNode(Node):
    """
    Digital Twin Visualization Node.

    Creates a real-time visualization of the federated learning system
    showing robot states, training progress, and system topology.
    """

    def __init__(self) -> None:
        super().__init__("digital_twin")

        self.cb_group = ReentrantCallbackGroup()

        # Parameters
        self.declare_parameter("output_dir", "/ros2_ws/results")
        self.declare_parameter("update_interval", 5.0)
        self.declare_parameter("image_width", 1200)
        self.declare_parameter("image_height", 800)

        self.output_dir = self.get_parameter("output_dir").value
        self.update_interval = self.get_parameter("update_interval").value

        self.get_logger().info("Initializing Digital Twin Visualization")

        # Visual state
        self.state = SystemVisualState()
        self.state_lock = threading.Lock()

        # Robot positions in a circle around aggregator
        self.robot_positions: dict[str, tuple[float, float]] = {}
        self.aggregator_position = (0.5, 0.5)  # Center

        # Color scheme
        self.colors = {
            "robot_idle": "#3498db",  # Blue
            "robot_training": "#e74c3c",  # Red
            "robot_complete": "#2ecc71",  # Green
            "aggregator": "#9b59b6",  # Purple
            "coordinator": "#f39c12",  # Orange
            "connection": "#95a5a6",  # Gray
            "active_connection": "#27ae60",  # Green
        }

        # QoS
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
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

        # Visualization timer
        if MATPLOTLIB_AVAILABLE:
            self.viz_timer = self.create_timer(
                self.update_interval, self.update_visualization, callback_group=self.cb_group
            )
        else:
            self.get_logger().warning("Matplotlib not available, visualization disabled")

        self.get_logger().info("Digital Twin initialized")

    def robot_status_callback(self, msg: String) -> None:
        """Handle robot status updates."""
        try:
            data = json.loads(msg.data)
            robot_id = data.get("robot_id")
            msg_type = data.get("type")

            if not robot_id:
                return

            with self.state_lock:
                if robot_id not in self.state.robots:
                    self.state.robots[robot_id] = RobotVisualState(
                        robot_id=robot_id,
                    )
                    # Recalculate all positions on a circle whenever a robot is added
                    self._recalculate_positions()
                    self.get_logger().info(f"Added robot {robot_id} to digital twin")

                robot = self.state.robots[robot_id]
                robot.last_update = time.time()

                if msg_type == "status":
                    robot.is_training = data.get("is_training", False)
                    robot.rounds_completed = data.get("training_round", 0)
                    if data.get("last_loss") is not None:
                        robot.loss = data["last_loss"]
                    if data.get("last_accuracy") is not None:
                        robot.accuracy = max(0.0, min(100.0, data["last_accuracy"]))

        except Exception as e:
            self.get_logger().error(f"Error in robot status callback: {e}")

    def aggregation_callback(self, msg: String) -> None:
        """Handle aggregation metrics."""
        try:
            data = json.loads(msg.data)

            with self.state_lock:
                self.state.global_round = data.get("round", 0)
                self.state.total_aggregations += 1
                self.state.mean_divergence = data.get("mean_divergence", 1.0)

        except Exception as e:
            self.get_logger().error(f"Error in aggregation callback: {e}")

    def coordinator_callback(self, msg: String) -> None:
        """Handle coordinator status."""
        try:
            data = json.loads(msg.data)

            with self.state_lock:
                self.state.coordinator_state = data.get("state", "UNKNOWN")
                self.state.global_round = data.get("current_round", 0)

        except Exception as e:
            self.get_logger().error(f"Error in coordinator callback: {e}")

    def _recalculate_positions(self) -> None:
        """Evenly distribute all robots on a circle (must hold state_lock)."""
        num = len(self.state.robots)
        if num == 0:
            return
        radius = 0.35
        for idx, robot in enumerate(self.state.robots.values()):
            angle = (idx * 2 * math.pi / num) + math.pi / 2
            robot.position = (0.5 + radius * math.cos(angle), 0.5 + radius * math.sin(angle))
            robot.angle = angle

    def update_visualization(self) -> None:
        """Generate and save visualization image."""
        if not MATPLOTLIB_AVAILABLE:
            return

        try:
            with self.state_lock:
                state_copy = SystemVisualState(
                    robots={k: v.model_copy() for k, v in self.state.robots.items()},
                    global_round=self.state.global_round,
                    total_aggregations=self.state.total_aggregations,
                    mean_divergence=self.state.mean_divergence,
                    coordinator_state=self.state.coordinator_state,
                )

            self._render_visualization(state_copy)

        except Exception as e:
            self.get_logger().error(f"Error updating visualization: {e}")

    def _render_visualization(self, state: SystemVisualState) -> None:
        """Render the digital twin visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle("Federated Learning Digital Twin", fontsize=16, fontweight="bold")

        # Left plot: Network topology
        ax1 = axes[0]
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect("equal")
        ax1.set_title("System Topology", fontsize=12)
        ax1.axis("off")

        # Draw aggregator (center)
        agg_circle = Circle(
            self.aggregator_position,
            0.08,
            facecolor=self.colors["aggregator"],
            edgecolor="black",
            linewidth=2,
            zorder=10,
        )
        ax1.add_patch(agg_circle)
        ax1.text(
            0.5,
            0.5,
            "AGG",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            zorder=11,
        )

        # Draw robots
        for robot_id, robot in state.robots.items():
            # Robot color based on state
            if robot.is_training:
                color = self.colors["robot_training"]
            elif robot.accuracy > 60:
                color = self.colors["robot_complete"]
            else:
                color = self.colors["robot_idle"]

            # Draw robot circle
            robot_circle = Circle(
                robot.position, 0.06, facecolor=color, edgecolor="black", linewidth=2, zorder=5
            )
            ax1.add_patch(robot_circle)

            # Robot label
            label = robot_id.replace("robot_", "R")
            ax1.text(
                robot.position[0],
                robot.position[1],
                label,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
                zorder=6,
            )

            # Connection line to aggregator
            conn_color = (
                self.colors["active_connection"] if robot.is_training else self.colors["connection"]
            )
            ax1.plot(
                [robot.position[0], 0.5],
                [robot.position[1], 0.5],
                color=conn_color,
                linewidth=2,
                alpha=0.5,
                zorder=1,
            )

            # Accuracy indicator (arc around robot)
            if robot.accuracy > 0:
                theta = np.linspace(0, 2 * np.pi * robot.accuracy / 100, 50)
                x = robot.position[0] + 0.075 * np.cos(theta)
                y = robot.position[1] + 0.075 * np.sin(theta)
                ax1.plot(x, y, color="#27ae60", linewidth=3, zorder=4)

        # Legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.colors["robot_idle"],
                markersize=10,
                label="Idle",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.colors["robot_training"],
                markersize=10,
                label="Training",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.colors["robot_complete"],
                markersize=10,
                label="Complete",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.colors["aggregator"],
                markersize=10,
                label="Aggregator",
            ),
        ]
        ax1.legend(handles=legend_elements, loc="lower left", fontsize=8)

        # Right plot: Metrics dashboard
        ax2 = axes[1]
        ax2.axis("off")
        ax2.set_title("Training Metrics", fontsize=12)

        # System status
        status_text = f"""
╔══════════════════════════════════════╗
║         SYSTEM STATUS                ║
╠══════════════════════════════════════╣
║  Coordinator: {state.coordinator_state:<20} ║
║  Global Round: {state.global_round:<19} ║
║  Aggregations: {state.total_aggregations:<19} ║
║  Mean Divergence: {state.mean_divergence:<16.4f} ║
╚══════════════════════════════════════╝
"""
        ax2.text(
            0.05,
            0.95,
            status_text,
            transform=ax2.transAxes,
            fontsize=10,
            fontfamily="monospace",
            verticalalignment="top",
        )

        # Robot metrics table
        robot_text = "╔═══════════╦═════════╦══════════╦════════╗\n"
        robot_text += "║   Robot   ║  Loss   ║ Accuracy ║ Rounds ║\n"
        robot_text += "╠═══════════╬═════════╬══════════╬════════╣\n"

        for robot_id, robot in state.robots.items():
            label = robot_id.replace("robot_", "R")
            robot_text += f"║ {label:^9} ║ {robot.loss:^7.3f} ║ {robot.accuracy:^8.1f}% ║ {robot.rounds_completed:^6} ║\n"

        robot_text += "╚═══════════╩═════════╩══════════╩════════╝"

        ax2.text(
            0.05,
            0.55,
            robot_text,
            transform=ax2.transAxes,
            fontsize=9,
            fontfamily="monospace",
            verticalalignment="top",
        )

        # Progress bar for average accuracy
        if state.robots:
            avg_acc = max(0.0, min(100.0, sum(r.accuracy for r in state.robots.values()) / len(state.robots)))
            ax2.text(
                0.05,
                0.2,
                f"Average Accuracy: {avg_acc:.1f}%",
                transform=ax2.transAxes,
                fontsize=11,
                fontweight="bold",
            )

            # Draw progress bar
            bar_width = 0.6
            bar_height = 0.03
            bar_x = 0.05
            bar_y = 0.12

            # Background
            ax2.add_patch(
                Rectangle(
                    (bar_x, bar_y),
                    bar_width,
                    bar_height,
                    transform=ax2.transAxes,
                    facecolor="#ecf0f1",
                    edgecolor="black",
                )
            )
            # Progress
            progress_width = bar_width * (avg_acc / 100)
            color = "#27ae60" if avg_acc > 60 else "#f39c12" if avg_acc > 40 else "#e74c3c"
            ax2.add_patch(
                Rectangle(
                    (bar_x, bar_y),
                    progress_width,
                    bar_height,
                    transform=ax2.transAxes,
                    facecolor=color,
                )
            )

        # Timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.99, 0.01, f"Updated: {timestamp}", ha="right", fontsize=8, color="gray")

        plt.tight_layout()

        # Save to file
        output_path = f"{self.output_dir}/digital_twin.png"
        plt.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        self.get_logger().debug(f"Saved digital twin visualization to {output_path}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)

    node = DigitalTwinNode()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
