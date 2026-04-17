# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS standalone-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md main.py /app/
COPY src/fl_robots /app/src/fl_robots

# Install CPU-only torch wheel plus the `ml` extra so the aggregation /
# simple_nn code paths (and their tests) work inside the lightweight image.
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        ".[ml]"

# Drop privileges. All artefacts live under /app (writable by appuser) and
# /tmp (world-writable). Avoids the "running containers as root" OWASP
# finding in standard scans.
RUN groupadd --system --gid 1000 appuser && \
    useradd --system --uid 1000 --gid 1000 --home /app appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/api/health', timeout=2)" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "8", "fl_robots.wsgi:app"]

FROM standalone-runtime AS standalone-test

USER root
COPY tests /app/tests
RUN python -m pip install "pytest>=7.4.0" "pytest-cov>=4.1.0" && \
    chown -R appuser:appuser /app/tests
USER appuser

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

# Install Python runtime deps straight from pyproject.toml extras so there is
# a SINGLE source of truth for pins — no drift between the ROS image and the
# standalone image. We copy only the metadata files first to maximise layer
# cache reuse.
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src/fl_robots/fl_robots/__init__.py /app/src/fl_robots/fl_robots/__init__.py
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        ".[ml,ros]"

WORKDIR /ros2_ws

COPY --from=ros-builder /ros2_ws/install /ros2_ws/install
RUN mkdir -p /ros2_ws/results /ros2_ws/logs /ros2_ws/models

COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh && \
    echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

# Non-root runtime user. The ROS entrypoint still sources system setup
# from /opt/ros (world-readable), so appuser can launch nodes fine.
RUN groupadd --system --gid 1000 appuser && \
    useradd --system --uid 1000 --gid 1000 --home /ros2_ws appuser && \
    chown -R appuser:appuser /ros2_ws && \
    echo "source /opt/ros/humble/setup.bash" >> /home/appuser/.bashrc 2>/dev/null || true && \
    echo "source /ros2_ws/install/setup.bash" >> /home/appuser/.bashrc 2>/dev/null || true
USER appuser

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
