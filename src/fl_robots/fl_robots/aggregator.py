#!/usr/bin/env python3
"""
Federated Learning Aggregator Node

This node implements the Federated Averaging (FedAvg) algorithm server.
It collects model weights from all robot agents and computes the weighted
average to create a global model.

ROS2 Concepts Demonstrated:
- Multiple subscriptions (collecting weights from all robots)
- Service servers (model registration, aggregation trigger)
- Quality of Service (QoS) profiles
- Parameter management
- Lifecycle awareness

Algorithm: FedAvg (McMahan et al., 2017)
- Collect local model weights from K clients
- Compute weighted average: w_global = Σ (n_k/n) * w_k
- Broadcast global model to all clients
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import String

import numpy as np
import json
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict
import threading
from dataclasses import dataclass, field

from fl_robots.models import SimpleNavigationNet, federated_averaging, compute_gradient_divergence


@dataclass
class RobotState:
    """Track state of each registered robot."""
    robot_id: str
    registered_at: float
    last_seen: float
    is_active: bool = True
    rounds_participated: int = 0
    current_weights: Optional[Dict[str, np.ndarray]] = None
    samples_trained: int = 0
    last_loss: Optional[float] = None
    last_accuracy: Optional[float] = None


class AggregatorNode(Node):
    """
    Federated Learning Aggregator Node.

    Central server that:
    1. Maintains registry of participating robots
    2. Collects local model updates
    3. Performs federated averaging
    4. Broadcasts global model
    5. Tracks training metrics
    """

    def __init__(self):
        super().__init__('aggregator')

        # Callback groups
        self.cb_group_subs = ReentrantCallbackGroup()
        self.cb_group_timers = MutuallyExclusiveCallbackGroup()

        # Declare parameters
        self._declare_parameters()

        self.min_robots = self.get_parameter('min_robots').value
        self.aggregation_timeout = self.get_parameter('aggregation_timeout').value
        self.auto_aggregate = self.get_parameter('auto_aggregate').value

        self.get_logger().info('Initializing Federated Learning Aggregator')

        # State tracking
        self.robots: Dict[str, RobotState] = {}
        self.current_round = 0
        self.pending_weights: Dict[str, Dict[str, np.ndarray]] = {}
        self.pending_samples: Dict[str, int] = {}
        self.state_lock = threading.Lock()

        # Global model (initialized when first robot registers)
        self.global_model = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)
        self.global_weights = self.global_model.get_weights()

        # Training history
        self.aggregation_history = []
        self.divergence_history = []

        # QoS profiles
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Publishers
        self.global_model_publisher = self.create_publisher(
            String,
            '/fl/global_model',
            qos_reliable
        )

        self.aggregation_metrics_publisher = self.create_publisher(
            String,
            '/fl/aggregation_metrics',
            qos_reliable
        )

        self.training_command_publisher = self.create_publisher(
            String,
            '/fl/training_command',
            qos_reliable
        )

        # Subscribers for robot status and weights
        self.status_subscriber = self.create_subscription(
            String,
            '/fl/robot_status',
            self.robot_status_callback,
            qos_reliable,
            callback_group=self.cb_group_subs
        )

        # Dynamic subscription to robot weight topics
        self.weight_subscribers = {}

        # Timers
        self.health_check_timer = self.create_timer(
            10.0,
            self.health_check_callback,
            callback_group=self.cb_group_timers
        )

        self.aggregation_timer = None
        if self.auto_aggregate:
            self.aggregation_timer = self.create_timer(
                self.aggregation_timeout,
                self.auto_aggregation_callback,
                callback_group=self.cb_group_timers
            )

        self.get_logger().info('Aggregator initialized, waiting for robots...')
        self.get_logger().info(f'Minimum robots required: {self.min_robots}')

    def _declare_parameters(self):
        """Declare aggregator parameters."""
        self.declare_parameter('min_robots', 2)
        self.declare_parameter('aggregation_timeout', 30.0)
        self.declare_parameter('auto_aggregate', True)
        self.declare_parameter('participation_threshold', 0.5)

    def robot_status_callback(self, msg: String):
        """Handle robot status and registration messages."""
        try:
            data = json.loads(msg.data)
            msg_type = data.get('type')
            robot_id = data.get('robot_id')

            if not robot_id:
                return

            if msg_type == 'registration':
                self._handle_registration(data)
            elif msg_type == 'status':
                self._handle_status_update(data)

        except Exception as e:
            self.get_logger().error(f'Error processing robot message: {e}')

    def _handle_registration(self, data: Dict[str, Any]):
        """Handle new robot registration."""
        robot_id = data['robot_id']

        with self.state_lock:
            if robot_id not in self.robots:
                self.robots[robot_id] = RobotState(
                    robot_id=robot_id,
                    registered_at=time.time(),
                    last_seen=time.time()
                )

                # Create subscriber for this robot's weights
                self._create_weight_subscriber(robot_id)

                self.get_logger().info(
                    f'Robot {robot_id} registered. '
                    f'Total robots: {len(self.robots)}'
                )

                # Send current global model to new robot
                self._publish_global_model()
            else:
                self.robots[robot_id].last_seen = time.time()
                self.robots[robot_id].is_active = True

    def _create_weight_subscriber(self, robot_id: str):
        """Create a subscriber for a robot's weight topic."""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        def callback(msg):
            self._handle_weight_update(robot_id, msg)

        topic = f'/fl/{robot_id}/model_weights'
        self.weight_subscribers[robot_id] = self.create_subscription(
            String,
            topic,
            callback,
            qos,
            callback_group=self.cb_group_subs
        )

        self.get_logger().debug(f'Created weight subscriber for {robot_id}')

    def _handle_weight_update(self, robot_id: str, msg: String):
        """Handle incoming weight update from a robot."""
        try:
            data = json.loads(msg.data)

            if data.get('type') != 'local_weights':
                return

            round_num = data.get('round', 0)

            # Deserialize weights
            weights = {}
            for name, values in data['weights'].items():
                weights[name] = np.array(values, dtype=np.float32)

            samples = data.get('samples_trained', 1)
            loss = data.get('loss')
            accuracy = data.get('accuracy')

            with self.state_lock:
                # Store weights for aggregation
                self.pending_weights[robot_id] = weights
                self.pending_samples[robot_id] = samples

                # Update robot state
                if robot_id in self.robots:
                    self.robots[robot_id].last_seen = time.time()
                    self.robots[robot_id].current_weights = weights
                    self.robots[robot_id].samples_trained = samples
                    self.robots[robot_id].last_loss = loss
                    self.robots[robot_id].last_accuracy = accuracy
                    self.robots[robot_id].rounds_participated += 1

            self.get_logger().info(
                f'Received weights from {robot_id} '
                f'(round {round_num}, loss: {loss if loss is None else f"{loss:.4f}"}, '
                f'acc: {accuracy if accuracy is None else f"{accuracy:.2f}"}%)'
            )

            # Check if we can aggregate
            self._check_aggregation_readiness()

        except Exception as e:
            self.get_logger().error(f'Error processing weights from {robot_id}: {e}')

    def _handle_status_update(self, data: Dict[str, Any]):
        """Handle robot status update."""
        robot_id = data['robot_id']

        with self.state_lock:
            if robot_id in self.robots:
                self.robots[robot_id].last_seen = time.time()
                self.robots[robot_id].is_active = True

    def _check_aggregation_readiness(self):
        """Check if we have enough weights to perform aggregation."""
        with self.state_lock:
            num_pending = len(self.pending_weights)
            num_active = sum(1 for r in self.robots.values() if r.is_active)

            participation = num_pending / max(num_active, 1)
            threshold = self.get_parameter('participation_threshold').value

            if num_pending >= self.min_robots and participation >= threshold:
                self.get_logger().info(
                    f'Aggregation ready: {num_pending}/{num_active} robots submitted'
                )
                if not self.auto_aggregate:
                    self._perform_aggregation()

    def auto_aggregation_callback(self):
        """Periodically check and perform aggregation."""
        with self.state_lock:
            if len(self.pending_weights) >= self.min_robots:
                self._perform_aggregation()

    def _perform_aggregation(self):
        """
        Perform Federated Averaging on collected weights.

        FedAvg Algorithm:
        1. Collect weights W_k and sample counts n_k from K clients
        2. Compute n = Σ n_k
        3. Compute W_global = Σ (n_k / n) * W_k
        """
        if len(self.pending_weights) < self.min_robots:
            self.get_logger().warning(
                f'Not enough weights for aggregation '
                f'({len(self.pending_weights)}/{self.min_robots})'
            )
            return

        self.get_logger().info(
            f'Starting FedAvg aggregation (round {self.current_round + 1}) '
            f'with {len(self.pending_weights)} robots'
        )

        try:
            # Collect weights and sample counts
            weights_list = list(self.pending_weights.values())
            sample_counts = [
                self.pending_samples.get(rid, 1)
                for rid in self.pending_weights.keys()
            ]

            # Compute gradient divergence before averaging (for metrics)
            divergences = compute_gradient_divergence(weights_list, self.global_weights)
            self.divergence_history.append({
                'round': self.current_round,
                'divergences': divergences,
                'mean_divergence': np.mean(divergences),
                'max_divergence': np.max(divergences)
            })

            # Perform federated averaging
            start_time = time.time()
            aggregated_weights = federated_averaging(weights_list, sample_counts)
            aggregation_time = time.time() - start_time

            # Update global model
            self.global_weights = aggregated_weights
            self.global_model.set_weights(aggregated_weights)

            # Update round counter
            self.current_round += 1

            # Record aggregation metrics
            metrics = {
                'round': self.current_round,
                'num_participants': len(weights_list),
                'total_samples': sum(sample_counts),
                'aggregation_time': aggregation_time,
                'mean_divergence': np.mean(divergences),
                'timestamp': time.time()
            }
            self.aggregation_history.append(metrics)

            self.get_logger().info(
                f'Aggregation complete (round {self.current_round}): '
                f'{len(weights_list)} participants, '
                f'{sum(sample_counts)} total samples, '
                f'divergence: {np.mean(divergences):.4f}'
            )

            # Publish global model and metrics
            self._publish_global_model()
            self._publish_aggregation_metrics(metrics)

            # Clear pending weights for next round
            self.pending_weights.clear()
            self.pending_samples.clear()

            # Trigger next training round
            self._send_training_command('start_training')

        except Exception as e:
            self.get_logger().error(f'Aggregation failed: {e}')

    def _publish_global_model(self):
        """Publish the global model to all robots."""
        weights_serializable = {
            name: arr.tolist() for name, arr in self.global_weights.items()
        }

        data = {
            'type': 'global_model',
            'round': self.current_round,
            'weights': weights_serializable,
            'timestamp': time.time()
        }

        msg = String()
        msg.data = json.dumps(data)
        self.global_model_publisher.publish(msg)

        self.get_logger().info(f'Published global model (round {self.current_round})')

    def _publish_aggregation_metrics(self, metrics: Dict[str, Any]):
        """Publish aggregation metrics for monitoring."""
        msg = String()
        msg.data = json.dumps(metrics)
        self.aggregation_metrics_publisher.publish(msg)

    def _send_training_command(self, command: str):
        """Send training command to all robots."""
        data = {
            'command': command,
            'round': self.current_round,
            'timestamp': time.time()
        }

        msg = String()
        msg.data = json.dumps(data)
        self.training_command_publisher.publish(msg)

        self.get_logger().info(f'Sent training command: {command}')

    def health_check_callback(self):
        """Check health of registered robots."""
        current_time = time.time()
        inactive_threshold = 60.0  # seconds

        with self.state_lock:
            for robot_id, state in self.robots.items():
                if current_time - state.last_seen > inactive_threshold:
                    if state.is_active:
                        state.is_active = False
                        self.get_logger().warning(f'Robot {robot_id} became inactive')

            active_count = sum(1 for r in self.robots.values() if r.is_active)
            self.get_logger().debug(
                f'Health check: {active_count}/{len(self.robots)} robots active'
            )

    def start_training_round(self):
        """Manually start a new training round."""
        self.get_logger().info('Starting new training round')
        self._send_training_command('start_training')


def main(args=None):
    rclpy.init(args=args)

    aggregator = AggregatorNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(aggregator)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        aggregator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
