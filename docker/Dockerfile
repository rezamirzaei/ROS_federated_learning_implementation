# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS standalone-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md main.py /app/
COPY src/fl_robots /app/src/fl_robots

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
        "flask>=3.0.0" \
        "pydantic>=2.0.0" && \
    python -m pip install --no-deps .

EXPOSE 5000

CMD ["python", "main.py", "run", "--host", "0.0.0.0", "--port", "5000"]

FROM standalone-runtime AS standalone-test

COPY tests /app/tests

RUN python -m pip install "pytest>=7.4.0"

CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]

# ── Stage 1: Build the ROS workspace ────────────────────────────────
FROM ros:humble-ros-base AS ros-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-lifecycle-msgs \
    ros-humble-ros2launch \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws
COPY src/fl_robots /ros2_ws/src/fl_robots
COPY src/fl_robots_interfaces /ros2_ws/src/fl_robots_interfaces

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --packages-select fl_robots_interfaces && \
    source /ros2_ws/install/setup.bash && \
    colcon build --packages-select fl_robots"

# ── Stage 2: Runtime image ──────────────────────────────────────────
# ROS2 Humble targets Ubuntu 22.04, so the runtime image still needs an
# Ubuntu/ROS base for ABI compatibility. We keep it lean by using ros-core.
FROM ros:humble-ros-core AS ros-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DOMAIN_ID=42 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    python3-pip \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-lifecycle-msgs \
    ros-humble-ros2cli \
    ros-humble-ros2launch \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.0.0,<=2.2.2" \
    "numpy>=1.24.0,<2.0" \
    "pydantic>=2.0.0" \
    "scipy>=1.10.0" \
    "scikit-learn>=1.3.0" \
    "flask>=3.0.0" \
    "flask-cors>=4.0.0" \
    "flask-socketio>=5.3.0" \
    "python-socketio>=5.0.0" \
    "jinja2>=3.1.0" \
    "eventlet>=0.33.0" \
    "matplotlib>=3.7.0"

WORKDIR /ros2_ws

COPY --from=ros-builder /ros2_ws/install /ros2_ws/install
RUN mkdir -p /ros2_ws/results /ros2_ws/logs /ros2_ws/models

COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh && \
    echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
