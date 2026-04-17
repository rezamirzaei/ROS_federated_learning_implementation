# syntax=docker/dockerfile:1.7

ARG PYTHON_SLIM_DIGEST=sha256:ad65166ad2583036804a3f772ec8f1cad733cf77d631c6840ab1ea9231b6b350
ARG ROS_HUMBLE_BASE_DIGEST=sha256:ec4ea4a624a8ea2aa3e304695731570cd47e4424980d7f00da4d172dc08a8625
ARG ROS_HUMBLE_CORE_DIGEST=sha256:a115d56b8bbbfd9f7f95e3b8c40da8a677403521a514386ceb7bdc48e18a76aa
ARG UV_VERSION=0.9.13
ARG JAMMY_BASH_VERSION=5.1-6ubuntu1.1
ARG JAMMY_PYTHON3_PIP_VERSION=22.0.2+dfsg-1ubuntu0.7
ARG ROS_COLCON_EXTENSIONS_VERSION=0.3.0-100
ARG ROS_RCLPY_VERSION=3.3.21-1jammy.20260310.163951
ARG ROS_STD_MSGS_VERSION=4.9.1-1jammy.20260304.212507
ARG ROS_GEOMETRY_MSGS_VERSION=4.9.1-1jammy.20260304.213757
ARG ROS_SENSOR_MSGS_VERSION=4.9.1-1jammy.20260304.214918
ARG ROS_LIFECYCLE_MSGS_VERSION=1.2.2-1jammy.20260304.211354
ARG ROS_ROS2CLI_VERSION=0.18.18-1jammy.20260310.165616
ARG ROS_ROS2LAUNCH_VERSION=0.19.13-1jammy.20260310.172117

FROM python:3.11-slim@${PYTHON_SLIM_DIGEST} AS standalone-runtime

ARG UV_VERSION

COPY --from=ghcr.io/astral-sh/uv:0.9.13 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# Layer 1: Copy only dependency metadata for caching.
COPY pyproject.toml uv.lock README.md /app/

# Layer 2: Install runtime deps (cached unless pyproject.toml / uv.lock change).
RUN uv sync --locked --extra ml --no-dev --no-editable --no-install-project --python python

# Layer 3: Copy source and install the project itself.
COPY src/fl_robots /app/src/fl_robots
COPY main.py /app/
RUN uv sync --locked --extra ml --no-dev --no-editable --python python

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
RUN uv sync --locked --extra ml --extra dev --no-editable --python python && \
    chown -R appuser:appuser /app/tests
USER appuser

CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]

# ── Stage 1: Build the ROS workspace ────────────────────────────────
FROM ros:humble-ros-base@${ROS_HUMBLE_BASE_DIGEST} AS ros-builder

ARG JAMMY_PYTHON3_PIP_VERSION
ARG ROS_COLCON_EXTENSIONS_VERSION
ARG ROS_RCLPY_VERSION
ARG ROS_STD_MSGS_VERSION
ARG ROS_GEOMETRY_MSGS_VERSION
ARG ROS_SENSOR_MSGS_VERSION
ARG ROS_LIFECYCLE_MSGS_VERSION
ARG ROS_ROS2LAUNCH_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip=${JAMMY_PYTHON3_PIP_VERSION} \
    python3-colcon-common-extensions=${ROS_COLCON_EXTENSIONS_VERSION} \
    ros-humble-rclpy=${ROS_RCLPY_VERSION} \
    ros-humble-std-msgs=${ROS_STD_MSGS_VERSION} \
    ros-humble-geometry-msgs=${ROS_GEOMETRY_MSGS_VERSION} \
    ros-humble-sensor-msgs=${ROS_SENSOR_MSGS_VERSION} \
    ros-humble-lifecycle-msgs=${ROS_LIFECYCLE_MSGS_VERSION} \
    ros-humble-ros2launch=${ROS_ROS2LAUNCH_VERSION} \
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
FROM ros:humble-ros-core@${ROS_HUMBLE_CORE_DIGEST} AS ros-runtime

ARG UV_VERSION
ARG JAMMY_BASH_VERSION
ARG ROS_RCLPY_VERSION
ARG ROS_STD_MSGS_VERSION
ARG ROS_GEOMETRY_MSGS_VERSION
ARG ROS_SENSOR_MSGS_VERSION
ARG ROS_LIFECYCLE_MSGS_VERSION
ARG ROS_ROS2CLI_VERSION
ARG ROS_ROS2LAUNCH_VERSION

COPY --from=ghcr.io/astral-sh/uv:0.9.13 /uv /usr/local/bin/uv

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DOMAIN_ID=42 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash=${JAMMY_BASH_VERSION} \
    ros-humble-rclpy=${ROS_RCLPY_VERSION} \
    ros-humble-std-msgs=${ROS_STD_MSGS_VERSION} \
    ros-humble-geometry-msgs=${ROS_GEOMETRY_MSGS_VERSION} \
    ros-humble-sensor-msgs=${ROS_SENSOR_MSGS_VERSION} \
    ros-humble-lifecycle-msgs=${ROS_LIFECYCLE_MSGS_VERSION} \
    ros-humble-ros2cli=${ROS_ROS2CLI_VERSION} \
    ros-humble-ros2launch=${ROS_ROS2LAUNCH_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# Install Python runtime deps from the checked-in uv.lock so the ROS image and
# standalone image consume the same locked Python dependency graph.
WORKDIR /app
COPY pyproject.toml uv.lock README.md /app/
RUN uv sync --locked --extra ml --extra ros --no-dev --no-editable --no-install-project --python /usr/bin/python3
COPY src/fl_robots /app/src/fl_robots
RUN uv sync --locked --extra ml --extra ros --no-dev --no-editable --python /usr/bin/python3

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
