#!/bin/bash
set -e

# Setup ROS2 environment
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash 2>/dev/null || true

exec "$@"
