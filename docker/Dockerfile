# ── Stage 1: Build ROS2 interfaces ──────────────────────────────────
FROM ros:humble-ros-base AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-lifecycle-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws
COPY src/ /ros2_ws/src/

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --packages-select fl_robots_interfaces && \
    source install/setup.bash && \
    colcon build --symlink-install --packages-select fl_robots"

# ── Stage 2: Slim runtime image ────────────────────────────────────
FROM python:3.10-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DOMAIN_ID=42 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (CPU-only PyTorch ~200MB vs 2GB GPU)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    numpy==1.24.3 \
    pydantic \
    scipy \
    scikit-learn \
    flask \
    flask-cors \
    flask-socketio \
    python-socketio \
    eventlet \
    pytest

WORKDIR /ros2_ws

# Copy built workspace + ROS2 runtime from builder (no dev tools)
COPY --from=builder /ros2_ws/install /ros2_ws/install
COPY --from=builder /ros2_ws/src /ros2_ws/src
COPY --from=builder /opt/ros/humble /opt/ros/humble

COPY tests/ /ros2_ws/tests/
RUN mkdir -p /ros2_ws/results /ros2_ws/logs

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
