# ROS2 Humble + PyTorch for Federated Learning
FROM ros:humble-ros-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DOMAIN_ID=42

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-nav-msgs \
    ros-humble-visualization-msgs \
    ros-humble-lifecycle-msgs \
    ros-humble-rclcpp-lifecycle \
    ros-humble-rviz2 \
    ros-humble-plotjuggler-ros \
    ros-humble-ros2launch \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version)
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install other Python ML dependencies
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    scipy \
    scikit-learn \
    matplotlib \
    pandas \
    pytest \
    flask \
    flask-cors \
    flask-socketio \
    python-socketio \
    eventlet

# Create workspace
WORKDIR /ros2_ws
RUN mkdir -p /ros2_ws/src

# Copy source code
COPY src/ /ros2_ws/src/

# Build the workspace (interfaces first, then fl_robots)
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --packages-select fl_robots_interfaces && \
    source install/setup.bash && \
    colcon build --symlink-install --packages-select fl_robots"

# Copy tests and scripts
COPY tests/ /ros2_ws/tests/

# Create results directory
RUN mkdir -p /ros2_ws/results

# Source the workspace in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# Set entrypoint
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
