#!/usr/bin/env python3
"""
Robot Agent Node — Federated Learning Client.

This node represents a single robot in the federated learning system.
Each robot:
1. Maintains a local model for navigation/obstacle avoidance
2. Trains on locally collected data (simulated sensor readings)
3. Publishes model weights after local training
4. Subscribes to global model updates from the aggregator
5. Provides an action server for coordinated training rounds
6. Offers services for model info and hyperparameter updates

ROS2 Concepts Demonstrated:
- Publishers/Subscribers (model weights, sensor data, status heartbeats)
- Services (GetModelInfo, UpdateHyperparameters)
- Actions (TrainRound with real-time progress feedback and cancellation)
- Parameters with dynamic reconfigure callback
- Timers (periodic status publishing)
- QoS Profiles (reliable + transient local for critical data, best effort for metrics)
- Callback Groups (reentrant for concurrent action/service handling)
- Multi-threaded Executor
"""

from __future__ import annotations

import json
import threading
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from fl_robots.models import SimpleNavigationNet

from .ros_compat import (
    ActionServer,
    CancelResponse,
    DurabilityPolicy,
    GoalResponse,
    HistoryPolicy,
    MultiThreadedExecutor,
    MutuallyExclusiveCallbackGroup,
    Node,
    QoSProfile,
    ReentrantCallbackGroup,
    ReliabilityPolicy,
    String,
    Twist,
    rclpy,
)
from .utils.determinism import derive_seed, seed_everything

# Try importing custom interfaces; fall back to String-based protocol
try:
    from fl_robots_interfaces.action import TrainRound
    from fl_robots_interfaces.msg import ModelWeights, RobotStatus, TrainingMetrics
    from fl_robots_interfaces.srv import GetModelInfo, RegisterRobot, UpdateHyperparameters

    CUSTOM_INTERFACES = True
except ImportError:
    CUSTOM_INTERFACES = False


class SyntheticDataGenerator:
    """
    Generates synthetic sensor data for training.
    Simulates LIDAR readings and navigation targets.
    Each robot gets slightly different data distribution (non-IID).
    """

    def __init__(self, robot_id: str, seed: int | None = None):
        self.robot_id = robot_id
        self.seed = seed if seed is not None else derive_seed(robot_id, 0)
        self.rng = np.random.RandomState(self.seed)
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
            goal_features = np.array(
                [goal_distance, goal_angle, current_velocity, angular_velocity]
            )
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
    - Service servers for model info and hyperparameter tuning
    - Action server for training coordination with real-time feedback
    - Dynamic parameters for hyperparameter tuning
    - Multiple QoS profiles for different message criticality
    - Callback groups for concurrent execution
    """

    def __init__(self):
        super().__init__("robot_agent")

        # Callback groups for concurrent execution
        self.cb_group_timers = MutuallyExclusiveCallbackGroup()
        self.cb_group_actions = ReentrantCallbackGroup()
        self.cb_group_services = ReentrantCallbackGroup()
        self.cb_group_subs = ReentrantCallbackGroup()

        # Declare and get parameters
        self._declare_parameters()
        self.robot_id = self.get_parameter("robot_id").value
        self.learning_rate = self.get_parameter("learning_rate").value
        self.batch_size = self.get_parameter("batch_size").value
        self.local_epochs = self.get_parameter("local_epochs").value
        self.samples_per_round = self.get_parameter("samples_per_round").value
        self.seed = int(self.get_parameter("seed").value)

        seed_everything(self.seed)
        data_seed = derive_seed(self.robot_id, self.seed)

        self.get_logger().info(f"Initializing Robot Agent: {self.robot_id}")

        # Initialize model and optimizer
        self.model = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Data generator with robot-specific distribution
        self.data_generator = SyntheticDataGenerator(self.robot_id, seed=data_seed)

        # Training state
        self.is_training = False
        self.training_round = 0
        self.local_loss_history: list[float] = []
        self.accuracy_history: list[float] = []
        self._max_history = 200  # prevent unbounded memory growth
        self.training_lock = threading.Lock()
        self._cancel_requested = False

        # FL algorithm config received from the aggregator's global-model
        # broadcast. Defaults here are used until the first global model
        # lands, so a cold-start robot behaves like FedAvg (baseline).
        self._fl_algorithm: str = "fedavg"
        self._fl_proximal_mu: float = 0.0
        # Snapshot of the last global parameters, keyed by name matching
        # ``named_parameters()``. Only populated when the aggregator is
        # running in ``algorithm=fedprox`` mode — saves memory for FedAvg.
        self._fl_global_snapshot: dict[str, torch.Tensor] | None = None

        # QoS profiles
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
        )

        # ── Publishers ──────────────────────────────────────────────
        self.weights_publisher = self.create_publisher(
            String, f"/fl/{self.robot_id}/model_weights", qos_reliable
        )

        self.status_publisher = self.create_publisher(String, "/fl/robot_status", qos_reliable)

        self.metrics_publisher = self.create_publisher(
            String, f"/fl/{self.robot_id}/metrics", qos_best_effort
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist, f"/{self.robot_id}/cmd_vel", qos_best_effort
        )

        # Custom typed publishers (if interfaces available)
        if CUSTOM_INTERFACES:
            self.typed_status_publisher = self.create_publisher(
                RobotStatus, f"/fl/{self.robot_id}/typed_status", qos_reliable
            )
            self.typed_metrics_publisher = self.create_publisher(
                TrainingMetrics, f"/fl/{self.robot_id}/typed_metrics", qos_best_effort
            )

        # ── Subscribers ─────────────────────────────────────────────
        self.global_weights_subscriber = self.create_subscription(
            String,
            "/fl/global_model",
            self.global_model_callback,
            qos_reliable,
            callback_group=self.cb_group_subs,
        )

        self.training_command_subscriber = self.create_subscription(
            String,
            "/fl/training_command",
            self.training_command_callback,
            qos_reliable,
            callback_group=self.cb_group_subs,
        )

        # ── Action Server: TrainRound ───────────────────────────────
        if CUSTOM_INTERFACES:
            self._train_action_server = ActionServer(
                self,
                TrainRound,
                f"/fl/{self.robot_id}/train_round",
                execute_callback=self._execute_train_action,
                goal_callback=self._handle_train_goal,
                cancel_callback=self._handle_train_cancel,
                callback_group=self.cb_group_actions,
            )
            self.get_logger().info(f"Action server: /fl/{self.robot_id}/train_round")

        # ── Service Servers ─────────────────────────────────────────
        if CUSTOM_INTERFACES:
            self._get_model_info_srv = self.create_service(
                GetModelInfo,
                f"/fl/{self.robot_id}/get_model_info",
                self._handle_get_model_info,
                callback_group=self.cb_group_services,
            )
            self._update_hyperparams_srv = self.create_service(
                UpdateHyperparameters,
                f"/fl/{self.robot_id}/update_hyperparameters",
                self._handle_update_hyperparameters,
                callback_group=self.cb_group_services,
            )
            self.get_logger().info("Services: get_model_info, update_hyperparameters")

        # ── Dynamic parameter callback ──────────────────────────────
        self.add_on_set_parameters_callback(self._on_parameter_change)

        # ── Timers ──────────────────────────────────────────────────
        self.status_timer = self.create_timer(
            5.0, self.publish_status, callback_group=self.cb_group_timers
        )

        # Register with the system
        self._publish_registration()

        self.get_logger().info(f"Robot Agent {self.robot_id} initialized successfully")
        self.get_logger().info(f"Model parameters: {self.model.count_parameters()}")
        self.get_logger().info(
            f"Custom interfaces: {'ENABLED' if CUSTOM_INTERFACES else 'DISABLED (fallback)'}"
        )

    # ────────────────────────────────────────────────────────────────
    # Parameter Management
    # ────────────────────────────────────────────────────────────────

    def _declare_parameters(self):
        """Declare all ROS2 parameters with defaults."""
        self.declare_parameter("robot_id", "robot_0")
        self.declare_parameter("learning_rate", 0.001)
        self.declare_parameter("batch_size", 32)
        self.declare_parameter("local_epochs", 5)
        self.declare_parameter("samples_per_round", 256)
        self.declare_parameter("seed", 42)

    def _on_parameter_change(self, params):
        """Handle dynamic parameter changes at runtime."""
        from .ros_compat import SetParametersResult

        for param in params:
            if param.name == "learning_rate" and param.value > 0:
                self.learning_rate = param.value
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.learning_rate
                self.get_logger().info(f"Learning rate updated to {self.learning_rate}")
            elif param.name == "batch_size" and param.value > 0:
                self.batch_size = param.value
            elif param.name == "local_epochs" and param.value > 0:
                self.local_epochs = param.value
            elif param.name == "samples_per_round" and param.value > 0:
                self.samples_per_round = param.value
        return SetParametersResult(successful=True)

    # ────────────────────────────────────────────────────────────────
    # Action Server: TrainRound
    # ────────────────────────────────────────────────────────────────

    def _handle_train_goal(self, goal_request):
        """Accept or reject a training goal."""
        if self.is_training:
            self.get_logger().warning(f"{self.robot_id}: Rejecting goal - already training")
            return GoalResponse.REJECT
        self.get_logger().info(
            f"{self.robot_id}: Accepting training goal for round {goal_request.round_number}"
        )
        return GoalResponse.ACCEPT

    def _handle_train_cancel(self, goal_handle):
        """Handle cancellation request."""
        self.get_logger().info(f"{self.robot_id}: Cancel requested")
        self._cancel_requested = True
        return CancelResponse.ACCEPT

    async def _execute_train_action(self, goal_handle):
        """
        Execute a training round via the Action Server.
        Publishes real-time feedback during training.
        """
        self.get_logger().info(f"{self.robot_id}: Executing TrainRound action")
        self._cancel_requested = False

        goal = goal_handle.request
        round_num = goal.round_number
        epochs = goal.local_epochs if goal.local_epochs > 0 else self.local_epochs
        batch_sz = goal.batch_size if goal.batch_size > 0 else self.batch_size
        lr = goal.learning_rate if goal.learning_rate > 0.0 else self.learning_rate

        # Temporarily adjust optimizer LR if overridden
        original_lr = self.learning_rate
        if lr != self.learning_rate:
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

        result = TrainRound.Result()
        result.robot_id = self.robot_id
        result.round_number = round_num

        with self.training_lock:
            if self.is_training:
                result.success = False
                goal_handle.abort()
                return result
            self.is_training = True

        try:
            self.training_round = round_num
            self.model.train()
            start_time = time.time()

            # Generate training data
            X, y = self.data_generator.generate_batch(self.samples_per_round)
            dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
            dataloader = DataLoader(dataset, batch_size=batch_sz, shuffle=True)
            total_batches = len(dataloader)
            if total_batches == 0:
                self.get_logger().error(
                    f"{self.robot_id}: empty dataloader "
                    f"(samples={self.samples_per_round}, batch_size={batch_sz})"
                )
                result.success = False
                result.training_duration_seconds = time.time() - start_time
                goal_handle.abort()
                return result

            total_loss = 0.0
            # Track last-epoch accuracy (reset per epoch) plus a running final.
            correct = 0
            total = 0

            for epoch in range(epochs):
                epoch_loss = 0.0
                correct = 0
                total = 0

                for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                    if self._cancel_requested:
                        self.get_logger().info(f"{self.robot_id}: Training cancelled")
                        goal_handle.canceled()
                        result.success = False
                        result.training_duration_seconds = time.time() - start_time
                        return result

                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

                    # Publish feedback
                    feedback = TrainRound.Feedback()
                    feedback.robot_id = self.robot_id
                    feedback.round_number = round_num
                    feedback.current_epoch = epoch + 1
                    feedback.total_epochs = epochs
                    feedback.current_batch = batch_idx + 1
                    feedback.total_batches = total_batches
                    feedback.current_loss = loss.item()
                    feedback.current_accuracy = 100.0 * correct / total if total > 0 else 0.0
                    feedback.elapsed_seconds = time.time() - start_time
                    progress = (epoch * total_batches + batch_idx + 1) / (epochs * total_batches)
                    feedback.estimated_remaining_seconds = (
                        (feedback.elapsed_seconds / progress * (1 - progress))
                        if progress > 0
                        else 0.0
                    )
                    goal_handle.publish_feedback(feedback)

                total_loss += epoch_loss

            avg_loss = total_loss / (epochs * total_batches)
            accuracy = 100.0 * correct / total
            duration = time.time() - start_time

            self._record_metrics(avg_loss, accuracy)

            layer_norms = [float(torch.norm(p.data).item()) for p in self.model.parameters()]

            result.success = True
            result.final_loss = avg_loss
            result.final_accuracy = accuracy
            result.total_samples_trained = self.samples_per_round
            result.training_duration_seconds = duration
            result.layer_norms = layer_norms

            goal_handle.succeed()

            self.get_logger().info(
                f"{self.robot_id}: Action complete - "
                f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, Time: {duration:.2f}s"
            )

            self._publish_weights()

        except Exception as e:
            self.get_logger().error(f"Training action error: {e}\n{traceback.format_exc()}")
            result.success = False
            goal_handle.abort()
        finally:
            self.is_training = False
            if lr != original_lr:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = original_lr

        return result

    # ────────────────────────────────────────────────────────────────
    # Service Handlers
    # ────────────────────────────────────────────────────────────────

    def _handle_get_model_info(self, request, response):
        """Service: Return current model information."""
        response.success = True
        response.robot_id = self.robot_id
        response.parameter_count = self.model.count_parameters()
        response.current_round = self.training_round
        response.last_loss = self.local_loss_history[-1] if self.local_loss_history else 0.0
        response.last_accuracy = self.accuracy_history[-1] if self.accuracy_history else 0.0
        response.model_architecture = json.dumps(
            {
                "type": "SimpleNavigationNet",
                "input_dim": 12,
                "hidden_dim": 64,
                "output_dim": 4,
                "layers": ["fc1", "bn1", "fc2", "bn2", "fc3", "bn3", "fc_out"],
            }
        )
        response.training_history = json.dumps(
            {
                "losses": self.local_loss_history[-20:],
                "accuracies": self.accuracy_history[-20:],
            }
        )
        self.get_logger().info(f"{self.robot_id}: GetModelInfo service called")
        return response

    def _handle_update_hyperparameters(self, request, response):
        """Service: Dynamically update training hyperparameters."""
        response.robot_id = self.robot_id
        if request.learning_rate > 0:
            self.learning_rate = request.learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.learning_rate
        if request.batch_size > 0:
            self.batch_size = request.batch_size
        if request.local_epochs > 0:
            self.local_epochs = request.local_epochs
        if request.samples_per_round > 0:
            self.samples_per_round = request.samples_per_round

        response.success = True
        response.message = f"Hyperparameters updated for {self.robot_id}"
        response.learning_rate = self.learning_rate
        response.batch_size = self.batch_size
        response.local_epochs = self.local_epochs
        self.get_logger().info(
            f"{self.robot_id}: Hyperparameters updated - "
            f"LR={self.learning_rate}, BS={self.batch_size}, Epochs={self.local_epochs}"
        )
        return response

    # ────────────────────────────────────────────────────────────────
    # Topic Callbacks & Publishing
    # ────────────────────────────────────────────────────────────────

    def _publish_registration(self):
        """Publish registration message to aggregator."""
        registration = {
            "type": "registration",
            "robot_id": self.robot_id,
            "model_params": self.model.count_parameters(),
            "timestamp": time.time(),
        }
        msg = String()
        msg.data = json.dumps(registration)
        self.status_publisher.publish(msg)
        self.get_logger().info(f"Published registration for {self.robot_id}")

    def publish_status(self):
        """Periodically publish robot status."""
        status = {
            "type": "status",
            "robot_id": self.robot_id,
            "is_training": self.is_training,
            "training_round": self.training_round,
            "last_loss": self.local_loss_history[-1] if self.local_loss_history else None,
            "last_accuracy": self.accuracy_history[-1] if self.accuracy_history else None,
            "timestamp": time.time(),
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_publisher.publish(msg)

        # Also publish typed status if custom interfaces available
        if CUSTOM_INTERFACES:
            typed_msg = RobotStatus()
            typed_msg.stamp = self.get_clock().now().to_msg()
            typed_msg.robot_id = self.robot_id
            typed_msg.status = (
                RobotStatus.STATUS_TRAINING if self.is_training else RobotStatus.STATUS_IDLE
            )
            typed_msg.current_round = self.training_round
            typed_msg.total_rounds_participated = len(self.local_loss_history)
            typed_msg.last_loss = (
                float(self.local_loss_history[-1]) if self.local_loss_history else 0.0
            )
            typed_msg.last_accuracy = (
                float(self.accuracy_history[-1]) if self.accuracy_history else 0.0
            )
            typed_msg.model_parameter_count = self.model.count_parameters()
            typed_msg.is_active = True
            self.typed_status_publisher.publish(typed_msg)

    def global_model_callback(self, msg: String):
        """Callback for receiving global model updates."""
        try:
            data = json.loads(msg.data)
            if data.get("type") != "global_model":
                return

            self.get_logger().info(f"{self.robot_id}: Received global model update")

            # Deserialize weights
            weights = {}
            for name, values in data["weights"].items():
                weights[name] = np.array(values, dtype=np.float32)

            # Latch algorithm config from the broadcast config block. Missing
            # config → assume FedAvg. Legacy aggregators that don't send
            # ``config`` keep the existing FedAvg behaviour — the whole
            # addition is backwards compatible.
            fl_cfg = data.get("config") or {}
            algorithm = str(fl_cfg.get("algorithm", "fedavg")).lower()
            proximal_mu = float(fl_cfg.get("proximal_mu", 0.0) or 0.0)

            # Update local model
            with self.training_lock:
                self.model.set_weights(weights)
                self.training_round = data.get("round", self.training_round)
                self._fl_algorithm = algorithm
                self._fl_proximal_mu = proximal_mu
                # Take a snapshot of the trainable parameters **only** — the
                # proximal term is over model weights, not BN running stats.
                if algorithm == "fedprox" and proximal_mu > 0.0:
                    named = dict(self.model.named_parameters())
                    self._fl_global_snapshot = {
                        name: p.detach().clone() for name, p in named.items()
                    }
                else:
                    self._fl_global_snapshot = None

            self.get_logger().info(
                f"{self.robot_id}: Updated to global model (round {self.training_round})"
            )

        except Exception as e:
            self.get_logger().error(f"Error processing global model: {e}")

    def training_command_callback(self, msg: String):
        """Handle training commands from coordinator (topic-based fallback)."""
        try:
            data = json.loads(msg.data)
            command = data.get("command")

            if command == "start_training":
                if not self.is_training:
                    self.get_logger().info(f"{self.robot_id}: Starting local training (topic cmd)")
                    self._execute_local_training(data.get("round", 0))
            elif command == "stop_training":
                self.is_training = False
                self._cancel_requested = True
                self.get_logger().info(f"{self.robot_id}: Stopping training")
            elif command == "publish_weights":
                self._publish_weights()

        except Exception as e:
            self.get_logger().error(f"Error processing training command: {e}")

    def _execute_local_training(self, round_num: int):
        """Execute local training on synthetic data (topic-based path)."""
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

                    # FedProx: proximal term. See ``_proximal_penalty`` —
                    # returns None under FedAvg so this is a hot-path no-op.
                    prox = self._proximal_penalty()
                    if prox is not None:
                        loss = loss + prox

                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

                total_loss += epoch_loss
                self._publish_training_progress(epoch, epoch_loss / len(dataloader))

            avg_loss = total_loss / (self.local_epochs * len(dataloader))
            accuracy = 100.0 * correct / total

            self._record_metrics(avg_loss, accuracy)

            self.get_logger().info(
                f"{self.robot_id}: Training complete - "
                f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )

            self._publish_weights()

        except Exception as e:
            self.get_logger().error(f"Training error: {e}")
        finally:
            self.is_training = False

    def _publish_training_progress(self, epoch: int, loss: float):
        """Publish training progress metrics."""
        metrics = {
            "type": "training_progress",
            "robot_id": self.robot_id,
            "round": self.training_round,
            "epoch": epoch,
            "loss": loss,
            "timestamp": time.time(),
        }
        msg = String()
        msg.data = json.dumps(metrics)
        self.metrics_publisher.publish(msg)

    def _publish_weights(self):
        """Publish local model weights to aggregator."""
        weights = self.model.get_trainable_weights()

        # Convert numpy arrays to lists for JSON serialization
        weights_serializable = {name: arr.tolist() for name, arr in weights.items()}

        data = {
            "type": "local_weights",
            "robot_id": self.robot_id,
            "round": self.training_round,
            "weights": weights_serializable,
            "samples_trained": self.samples_per_round,
            "loss": self.local_loss_history[-1] if self.local_loss_history else None,
            "accuracy": self.accuracy_history[-1] if self.accuracy_history else None,
            "timestamp": time.time(),
        }

        msg = String()
        msg.data = json.dumps(data)
        self.weights_publisher.publish(msg)

        self.get_logger().info(
            f"{self.robot_id}: Published local weights (round {self.training_round})"
        )

    def _record_metrics(self, loss: float, accuracy: float) -> None:
        """Append loss/accuracy and trim to bounded history."""
        self.local_loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        if len(self.local_loss_history) > self._max_history:
            self.local_loss_history = self.local_loss_history[-self._max_history :]
            self.accuracy_history = self.accuracy_history[-self._max_history :]

    def _proximal_penalty(self) -> torch.Tensor | None:
        """Compute the FedProx proximal term ``½·μ·‖w − w_global‖²``.

        Returns ``None`` when running FedAvg or when no global snapshot
        has been received yet (cold start). Called from inside both
        training loops; designed to be a no-op on the hot path for
        FedAvg (see ``if snapshot is None: return None`` guard).

        Raises ``RuntimeError`` if the stored FedProx snapshot is missing
        trainable parameters or no longer matches the current model shape.
        Partial regularisation would silently degrade the algorithm, so we
        fail loudly instead of pretending FedProx is still applied.
        """
        snapshot = self._fl_global_snapshot
        mu = self._fl_proximal_mu
        if snapshot is None or mu <= 0.0 or self._fl_algorithm != "fedprox":
            return None
        prox: torch.Tensor | None = None
        missing: list[str] = []
        mismatched: list[str] = []
        for name, param in self.model.named_parameters():
            g = snapshot.get(name)
            if g is None:
                missing.append(name)
                continue
            if g.shape != param.shape:
                mismatched.append(f"{name}: snapshot {tuple(g.shape)} != model {tuple(param.shape)}")
                continue
            term = (param - g).pow(2).sum()
            prox = term if prox is None else prox + term
        if missing or mismatched:
            details: list[str] = []
            if missing:
                details.append(f"missing params [{', '.join(missing[:3])}]")
            if mismatched:
                details.append(f"shape mismatch [{'; '.join(mismatched[:3])}]")
            raise RuntimeError(f"Invalid FedProx snapshot: {'; '.join(details)}")
        if prox is None:
            return None
        return 0.5 * mu * prox

    # ────────────────────────────────────────────────────────────────
    # Inference
    # ────────────────────────────────────────────────────────────────

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
            if x.dim() == 1:
                x = x.unsqueeze(0)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0, action].item())

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


if __name__ == "__main__":
    main()
