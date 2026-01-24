#!/usr/bin/env python3
"""
Robot Agent Node - Federated Learning Client

This node represents a single robot in the federated learning system.
Each robot:
1. Maintains a local model for navigation/obstacle avoidance
2. Trains on locally collected data (simulated sensor readings)
3. Publishes model weights after local training
4. Subscribes to global model updates from the aggregator
5. Provides an action server for coordinated training rounds

ROS2 Concepts Demonstrated:
- Publishers/Subscribers (model weights, sensor data)
- Services (registration, model info)
- Actions (training with progress feedback)
- Parameters (learning rate, batch size, local epochs)
- Timers (periodic publishing)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import String, Float32MultiArray, Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import time
from typing import Dict, Any, Optional
import threading

from fl_robots.models import SimpleNavigationNet


class SyntheticDataGenerator:
    """
    Generates synthetic sensor data for training.
    Simulates LIDAR readings and navigation targets.
    Each robot gets slightly different data distribution (non-IID).
    """

    def __init__(self, robot_id: str, seed: int = None):
        self.robot_id = robot_id
        # Create different data distributions for each robot (non-IID simulation)
        self.seed = seed or hash(robot_id) % 10000
        self.rng = np.random.RandomState(self.seed)

        # Robot-specific bias to simulate non-IID data
        self.obstacle_bias = self.rng.uniform(-0.3, 0.3, size=8)
        self.goal_bias = self.rng.uniform(-0.2, 0.2, size=4)

    def generate_batch(self, batch_size: int = 32) -> tuple:
        """
        Generate a batch of (sensor_readings, labels) for training.

        Returns:
            X: Sensor readings (batch_size, 12)
               - 8 LIDAR distances (normalized 0-1)
               - 4 goal-related features (distance, angle, velocity)
            y: Action labels (batch_size,)
               - 0: Forward, 1: Left, 2: Right, 3: Stop
        """
        X = []
        y = []

        for _ in range(batch_size):
            # Simulate LIDAR readings (8 directions)
            lidar = self.rng.uniform(0.1, 1.0, size=8)
            lidar += self.obstacle_bias * 0.1  # Add robot-specific bias
            lidar = np.clip(lidar, 0.0, 1.0)

            # Simulate goal-related features
            goal_distance = self.rng.uniform(0.0, 1.0)
            goal_angle = self.rng.uniform(-1.0, 1.0)  # Normalized angle
            current_velocity = self.rng.uniform(0.0, 1.0)
            angular_velocity = self.rng.uniform(-1.0, 1.0)
            goal_features = np.array([goal_distance, goal_angle,
                                     current_velocity, angular_velocity])
            goal_features += self.goal_bias * 0.1
            goal_features = np.clip(goal_features, -1.0, 1.0)

            sensor_reading = np.concatenate([lidar, goal_features])

            # Generate label based on simple rules + noise
            label = self._compute_action(lidar, goal_features)

            X.append(sensor_reading)
            y.append(label)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def _compute_action(self, lidar: np.ndarray, goal: np.ndarray) -> int:
        """Simple rule-based labeling with noise."""
        front_clear = lidar[0] > 0.4 and lidar[1] > 0.3 and lidar[7] > 0.3
        left_clear = lidar[2] > 0.4 and lidar[3] > 0.4
        right_clear = lidar[5] > 0.4 and lidar[6] > 0.4

        goal_angle = goal[1]  # Normalized angle to goal

        # Add some noise to make it interesting
        if self.rng.random() < 0.1:
            return self.rng.randint(0, 4)

        if not front_clear:
            if left_clear and (not right_clear or goal_angle > 0):
                return 1  # Turn left
            elif right_clear:
                return 2  # Turn right
            else:
                return 3  # Stop
        else:
            if abs(goal_angle) < 0.2:
                return 0  # Forward
            elif goal_angle > 0:
                return 1  # Turn left
            else:
                return 2  # Turn right


class RobotAgentNode(Node):
    """
    Robot Agent Node for Federated Learning.

    This node demonstrates comprehensive ROS2 usage:
    - Topic publishing/subscribing for model weight exchange
    - Service for robot registration
    - Action server for training coordination
    - Dynamic parameters for hyperparameter tuning
    """

    def __init__(self):
        super().__init__('robot_agent')

        # Callback groups for concurrent execution
        self.cb_group_timers = MutuallyExclusiveCallbackGroup()
        self.cb_group_actions = ReentrantCallbackGroup()
        self.cb_group_subs = ReentrantCallbackGroup()

        # Declare and get parameters
        self._declare_parameters()
        self.robot_id = self.get_parameter('robot_id').value
        self.learning_rate = self.get_parameter('learning_rate').value
        self.batch_size = self.get_parameter('batch_size').value
        self.local_epochs = self.get_parameter('local_epochs').value
        self.samples_per_round = self.get_parameter('samples_per_round').value

        self.get_logger().info(f'Initializing Robot Agent: {self.robot_id}')

        # Initialize model and optimizer
        self.model = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Data generator with robot-specific distribution
        self.data_generator = SyntheticDataGenerator(self.robot_id)

        # Training state
        self.is_training = False
        self.training_round = 0
        self.local_loss_history = []
        self.accuracy_history = []
        self.training_lock = threading.Lock()

        # QoS profiles
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.weights_publisher = self.create_publisher(
            String,
            f'/fl/{self.robot_id}/model_weights',
            qos_reliable
        )

        self.status_publisher = self.create_publisher(
            String,
            '/fl/robot_status',
            qos_reliable
        )

        self.metrics_publisher = self.create_publisher(
            String,
            f'/fl/{self.robot_id}/metrics',
            qos_best_effort
        )

        # Publish velocity commands (for demonstration)
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            f'/{self.robot_id}/cmd_vel',
            qos_best_effort
        )

        # Subscribers
        self.global_weights_subscriber = self.create_subscription(
            String,
            '/fl/global_model',
            self.global_model_callback,
            qos_reliable,
            callback_group=self.cb_group_subs
        )

        self.training_command_subscriber = self.create_subscription(
            String,
            '/fl/training_command',
            self.training_command_callback,
            qos_reliable,
            callback_group=self.cb_group_subs
        )

        # Timers
        self.status_timer = self.create_timer(
            5.0,  # Publish status every 5 seconds
            self.publish_status,
            callback_group=self.cb_group_timers
        )

        # Register with the system
        self._publish_registration()

        self.get_logger().info(f'Robot Agent {self.robot_id} initialized successfully')
        self.get_logger().info(f'Model parameters: {self.model.count_parameters()}')

    def _declare_parameters(self):
        """Declare all ROS2 parameters with defaults."""
        self.declare_parameter('robot_id', 'robot_0')
        self.declare_parameter('learning_rate', 0.001)
        self.declare_parameter('batch_size', 32)
        self.declare_parameter('local_epochs', 5)
        self.declare_parameter('samples_per_round', 256)

    def _publish_registration(self):
        """Publish registration message to aggregator."""
        registration = {
            'type': 'registration',
            'robot_id': self.robot_id,
            'model_params': self.model.count_parameters(),
            'timestamp': time.time()
        }
        msg = String()
        msg.data = json.dumps(registration)
        self.status_publisher.publish(msg)
        self.get_logger().info(f'Published registration for {self.robot_id}')

    def publish_status(self):
        """Periodically publish robot status."""
        status = {
            'type': 'status',
            'robot_id': self.robot_id,
            'is_training': self.is_training,
            'training_round': self.training_round,
            'last_loss': self.local_loss_history[-1] if self.local_loss_history else None,
            'last_accuracy': self.accuracy_history[-1] if self.accuracy_history else None,
            'timestamp': time.time()
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_publisher.publish(msg)

    def global_model_callback(self, msg: String):
        """
        Callback for receiving global model updates.
        Updates local model with aggregated weights.
        """
        try:
            data = json.loads(msg.data)
            if data.get('type') != 'global_model':
                return

            self.get_logger().info(f'{self.robot_id}: Received global model update')

            # Deserialize weights
            weights = {}
            for name, values in data['weights'].items():
                weights[name] = np.array(values, dtype=np.float32)

            # Update local model
            with self.training_lock:
                self.model.set_weights(weights)
                self.training_round = data.get('round', self.training_round)

            self.get_logger().info(f'{self.robot_id}: Updated to global model (round {self.training_round})')

        except Exception as e:
            self.get_logger().error(f'Error processing global model: {e}')

    def training_command_callback(self, msg: String):
        """Handle training commands from coordinator."""
        try:
            data = json.loads(msg.data)
            command = data.get('command')

            if command == 'start_training':
                if not self.is_training:
                    self.get_logger().info(f'{self.robot_id}: Starting local training')
                    self._execute_local_training(data.get('round', 0))

            elif command == 'stop_training':
                self.is_training = False
                self.get_logger().info(f'{self.robot_id}: Stopping training')

            elif command == 'publish_weights':
                self._publish_weights()

        except Exception as e:
            self.get_logger().error(f'Error processing training command: {e}')

    def _execute_local_training(self, round_num: int):
        """
        Execute local training on synthetic data.

        This demonstrates the federated learning local training phase:
        1. Generate local data (simulating robot's own sensor data)
        2. Train for multiple epochs
        3. Publish updated weights
        """
        with self.training_lock:
            if self.is_training:
                return
            self.is_training = True

        try:
            self.training_round = round_num
            self.model.train()

            # Generate training data
            X, y = self.data_generator.generate_batch(self.samples_per_round)
            dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            total_loss = 0.0
            correct = 0
            total = 0

            for epoch in range(self.local_epochs):
                epoch_loss = 0.0

                for batch_X, batch_y in dataloader:
                    self.optimizer.zero_grad()

                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)

                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

                total_loss += epoch_loss

                # Publish training progress
                self._publish_training_progress(epoch, epoch_loss / len(dataloader))

            avg_loss = total_loss / (self.local_epochs * len(dataloader))
            accuracy = 100.0 * correct / total

            self.local_loss_history.append(avg_loss)
            self.accuracy_history.append(accuracy)

            self.get_logger().info(
                f'{self.robot_id}: Training complete - '
                f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%'
            )

            # Publish updated weights
            self._publish_weights()

        except Exception as e:
            self.get_logger().error(f'Training error: {e}')

        finally:
            self.is_training = False

    def _publish_training_progress(self, epoch: int, loss: float):
        """Publish training progress metrics."""
        metrics = {
            'type': 'training_progress',
            'robot_id': self.robot_id,
            'round': self.training_round,
            'epoch': epoch,
            'loss': loss,
            'timestamp': time.time()
        }
        msg = String()
        msg.data = json.dumps(metrics)
        self.metrics_publisher.publish(msg)

    def _publish_weights(self):
        """Publish local model weights to aggregator."""
        weights = self.model.get_weights()

        # Convert numpy arrays to lists for JSON serialization
        weights_serializable = {
            name: arr.tolist() for name, arr in weights.items()
        }

        data = {
            'type': 'local_weights',
            'robot_id': self.robot_id,
            'round': self.training_round,
            'weights': weights_serializable,
            'samples_trained': self.samples_per_round,
            'loss': self.local_loss_history[-1] if self.local_loss_history else None,
            'accuracy': self.accuracy_history[-1] if self.accuracy_history else None,
            'timestamp': time.time()
        }

        msg = String()
        msg.data = json.dumps(data)
        self.weights_publisher.publish(msg)

        self.get_logger().info(f'{self.robot_id}: Published local weights (round {self.training_round})')

    def inference(self, sensor_data: np.ndarray) -> tuple:
        """
        Run inference on sensor data and return action.

        Args:
            sensor_data: Array of shape (12,) with sensor readings

        Returns:
            action: Predicted action (0-3)
            confidence: Confidence score
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(sensor_data, dtype=torch.float32)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs).item()
            confidence = probs[0, action].item()

        return action, confidence


def main(args=None):
    rclpy.init(args=args)

    robot_agent = RobotAgentNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(robot_agent)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        robot_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
