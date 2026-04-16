"""
Standalone simulation engine that replicates the full ROS2 federated-learning
pipeline in a single process.

The engine spawns lightweight "virtual" robot agents, an aggregator, and a
coordinator — all communicating through :class:`ros_web.message_bus.MessageBus`
instead of DDS.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ros_web.message_bus import MessageBus
from ros_web.models import (
    SimpleNavigationNet,
    federated_averaging,
    compute_gradient_divergence,
)
from ros_web.mpc import DistributedMPCPlanner, MPCConfig


# ── Synthetic data (same logic as robot_agent.py) ───────────────────

class _SyntheticDataGenerator:
    def __init__(self, robot_id: str):
        self.rng = np.random.RandomState(hash(robot_id) % 10000)
        self.obstacle_bias = self.rng.uniform(-0.3, 0.3, size=8)
        self.goal_bias = self.rng.uniform(-0.2, 0.2, size=4)

    def generate_batch(self, n: int = 256):
        X, y = [], []
        for _ in range(n):
            lidar = np.clip(self.rng.uniform(0.1, 1.0, 8) + self.obstacle_bias * 0.1, 0, 1)
            goal = np.clip(
                np.array([self.rng.uniform(0, 1), self.rng.uniform(-1, 1),
                          self.rng.uniform(0, 1), self.rng.uniform(-1, 1)])
                + self.goal_bias * 0.1, -1, 1)
            X.append(np.concatenate([lidar, goal]))
            y.append(self._label(lidar, goal))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def _label(self, lidar, goal):
        if self.rng.random() < 0.1:
            return self.rng.randint(0, 4)
        front = lidar[0] > 0.4 and lidar[1] > 0.3 and lidar[7] > 0.3
        left = lidar[2] > 0.4 and lidar[3] > 0.4
        right = lidar[5] > 0.4 and lidar[6] > 0.4
        angle = goal[1]
        if not front:
            return 1 if left and (not right or angle > 0) else (2 if right else 3)
        return 0 if abs(angle) < 0.2 else (1 if angle > 0 else 2)


# ── Virtual Robot Agent ─────────────────────────────────────────────

@dataclass
class _VirtualRobot:
    robot_id: str
    model: SimpleNavigationNet
    optimizer: Any = None
    data_gen: Any = None
    loss_history: list = field(default_factory=list)
    acc_history: list = field(default_factory=list)
    is_training: bool = False
    training_round: int = 0
    # MPC state
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


# ── Simulation Engine ───────────────────────────────────────────────

class SimulationEngine:
    """
    Runs the full FL pipeline in-process.

    Public surface consumed by :func:`ros_web.web.create_app`:
    - ``bus``            – the message bus
    - ``status``         – dict snapshot for ``/api/status``
    - ``issue_command``  – inject a command string
    - ``shutdown``       – stop background threads
    """

    def __init__(self, num_robots: int = 4, auto_start: bool = True) -> None:
        self.bus = MessageBus()
        self.mpc = DistributedMPCPlanner(MPCConfig(formation_radius=3.0))

        # Global model
        self.global_model = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)
        self.global_weights = self.global_model.get_weights()
        self.current_round = 0
        self.total_aggregations = 0
        self.mean_divergence = 0.0
        self.coordinator_state = "IDLE"
        self.start_time = time.time()

        # Robots
        self.robots: Dict[str, _VirtualRobot] = {}
        for i in range(num_robots):
            rid = f"robot_{i}"
            model = SimpleNavigationNet(input_dim=12, hidden_dim=64, output_dim=4)
            model.set_weights(self.global_weights)
            angle = 2 * math.pi * i / num_robots
            self.robots[rid] = _VirtualRobot(
                robot_id=rid,
                model=model,
                optimizer=optim.Adam(model.parameters(), lr=0.001),
                data_gen=_SyntheticDataGenerator(rid),
                x=3.0 * math.cos(angle),
                y=3.0 * math.sin(angle),
                theta=angle + math.pi,
            )
            self.mpc.update_pose(rid, self.robots[rid].x, self.robots[rid].y, self.robots[rid].theta)

        # History for charts
        self.loss_history: List[dict] = []
        self.acc_history: List[dict] = []
        self.events: list = []

        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._autopilot = True
        self._step_event = threading.Event()

        if auto_start:
            self.start()

    # ── Control ─────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._running = False
        self._step_event.set()

    def issue_command(self, cmd: str) -> dict:
        """Process a dashboard command string."""
        if cmd == "start_training":
            self._step_event.set()
            return {"success": True, "message": "Training step triggered"}
        if cmd == "stop_training":
            self._autopilot = False
            return {"success": True, "message": "Autopilot disabled"}
        if cmd == "toggle_autopilot":
            self._autopilot = not self._autopilot
            return {"success": True, "message": f"Autopilot {'ON' if self._autopilot else 'OFF'}"}
        if cmd == "publish_weights":
            self._aggregate()
            return {"success": True, "message": "Forced aggregation"}
        return {"success": False, "message": f"Unknown command: {cmd}"}

    # ── Status snapshot ─────────────────────────────────────────────

    @property
    def status(self) -> dict:
        with self._lock:
            robots_data = {}
            total_loss = total_acc = count = best_acc = 0
            for rid, r in self.robots.items():
                robots_data[rid] = {
                    "is_training": r.is_training,
                    "loss": r.loss_history[-1] if r.loss_history else None,
                    "accuracy": r.acc_history[-1] if r.acc_history else None,
                    "rounds": r.training_round,
                }
                if r.loss_history:
                    total_loss += r.loss_history[-1]; count += 1
                if r.acc_history:
                    total_acc += r.acc_history[-1]
                    best_acc = max(best_acc, r.acc_history[-1])

            return {
                "coordinator_state": self.coordinator_state,
                "current_round": self.current_round,
                "total_aggregations": self.total_aggregations,
                "active_robots": len(self.robots),
                "mean_divergence": self.mean_divergence,
                "avg_loss": total_loss / count if count else None,
                "avg_accuracy": total_acc / count if count else None,
                "best_accuracy": best_acc if best_acc else None,
                "training_time": time.time() - self.start_time,
                "robots": robots_data,
                "events": self.events[-200:],
                "loss_history": self.loss_history[-50:],
                "acc_history": self.acc_history[-50:],
                "autopilot": self._autopilot,
            }

    # ── Background loop ─────────────────────────────────────────────

    def _loop(self) -> None:
        self._add_event("Simulation started")
        self.coordinator_state = "WAITING_FOR_ROBOTS"
        time.sleep(1)
        self.coordinator_state = "TRAINING_ROUND"

        while self._running:
            if self._autopilot:
                self._step()
                time.sleep(2)          # pace the simulation
            else:
                self._step_event.wait(timeout=5)
                if self._step_event.is_set():
                    self._step_event.clear()
                    self._step()

    def _step(self) -> None:
        """Execute one FL round: local training → aggregation → broadcast."""
        self.current_round += 1
        self.coordinator_state = "TRAINING_ROUND"
        self._add_event(f"Round {self.current_round} – local training")

        # Local training
        for rid, robot in self.robots.items():
            self._train_local(robot)

        # Aggregation
        self._aggregate()

        # MPC step (move robots slightly)
        for rid, robot in self.robots.items():
            goal = (0.0, 0.0)  # converge to center
            lin, ang = self.mpc.plan(rid, goal)
            robot.x += lin * 0.1 * math.cos(robot.theta)
            robot.y += lin * 0.1 * math.sin(robot.theta)
            robot.theta += ang * 0.1
            self.mpc.update_pose(rid, robot.x, robot.y, robot.theta)

        self.coordinator_state = "AGGREGATING"
        self.bus.publish("/fl/coordinator_status", {
            "state": self.coordinator_state,
            "current_round": self.current_round,
        })

    def _train_local(self, robot: _VirtualRobot) -> None:
        robot.is_training = True
        robot.training_round = self.current_round
        criterion = nn.CrossEntropyLoss()

        X, y = robot.data_gen.generate_batch(256)
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        dl = DataLoader(ds, batch_size=32, shuffle=True)

        robot.model.train()
        total_loss = correct = total = 0
        for epoch in range(5):
            for bx, by in dl:
                robot.optimizer.zero_grad()
                out = robot.model(bx)
                loss = criterion(out, by)
                loss.backward()
                robot.optimizer.step()
                total_loss += loss.item()
                _, pred = torch.max(out, 1)
                total += by.size(0)
                correct += (pred == by).sum().item()

        avg_loss = total_loss / (5 * len(dl))
        accuracy = 100.0 * correct / total
        robot.loss_history.append(avg_loss)
        robot.acc_history.append(accuracy)
        robot.is_training = False

        self.bus.publish(f"/fl/{robot.robot_id}/model_weights", {
            "type": "local_weights",
            "robot_id": robot.robot_id,
            "round": self.current_round,
            "loss": avg_loss,
            "accuracy": accuracy,
        })

    def _aggregate(self) -> None:
        weights_list = [r.model.get_weights() for r in self.robots.values()]
        sample_counts = [256] * len(weights_list)

        divergences = compute_gradient_divergence(weights_list, self.global_weights)
        self.mean_divergence = float(np.mean(divergences))

        aggregated = federated_averaging(weights_list, sample_counts)
        self.global_weights = aggregated
        self.global_model.set_weights(aggregated)

        for r in self.robots.values():
            r.model.set_weights(aggregated)

        self.total_aggregations += 1
        self._add_event(f"Aggregation round {self.total_aggregations} – divergence {self.mean_divergence:.4f}")

        # Record chart history
        snap_loss, snap_acc = {}, {}
        for rid, r in self.robots.items():
            snap_loss[rid] = {"loss": r.loss_history[-1] if r.loss_history else None}
            snap_acc[rid] = {"accuracy": r.acc_history[-1] if r.acc_history else None}
        self.loss_history.append({"round": self.total_aggregations, "robots": snap_loss})
        self.acc_history.append({"round": self.total_aggregations, "robots": snap_acc})

        self.bus.publish("/fl/aggregation_metrics", {
            "round": self.total_aggregations,
            "num_participants": len(weights_list),
            "mean_divergence": self.mean_divergence,
        })

    # ── Helpers ─────────────────────────────────────────────────────

    def _add_event(self, message: str) -> None:
        with self._lock:
            self.events.append({"time": time.strftime("%H:%M:%S"), "message": message})

