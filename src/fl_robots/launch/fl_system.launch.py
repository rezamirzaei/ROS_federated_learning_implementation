#!/usr/bin/env python3
"""
Launch file for the complete Federated Learning system.

This launch file demonstrates:
- Launching multiple nodes
- Parameter configuration
- Node namespacing
- Launch arguments
- Conditional launching
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, TimerAction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, PushRosNamespace
from launch.conditions import IfCondition


def generate_launch_description():
    # Declare launch arguments
    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        default_value='3',
        description='Number of robot agents to spawn'
    )

    total_rounds_arg = DeclareLaunchArgument(
        'total_rounds',
        default_value='20',
        description='Total number of training rounds'
    )

    learning_rate_arg = DeclareLaunchArgument(
        'learning_rate',
        default_value='0.001',
        description='Learning rate for local training'
    )

    batch_size_arg = DeclareLaunchArgument(
        'batch_size',
        default_value='32',
        description='Batch size for local training'
    )

    local_epochs_arg = DeclareLaunchArgument(
        'local_epochs',
        default_value='5',
        description='Number of local epochs per round'
    )

    enable_monitor_arg = DeclareLaunchArgument(
        'enable_monitor',
        default_value='true',
        description='Enable monitoring node'
    )

    enable_ui_arg = DeclareLaunchArgument(
        'enable_ui',
        default_value='true',
        description='Enable web dashboard and digital twin'
    )

    # Aggregator node
    aggregator_node = Node(
        package='fl_robots',
        executable='aggregator',
        name='aggregator',
        output='screen',
        parameters=[{
            'min_robots': 2,
            'aggregation_timeout': 30.0,
            'auto_aggregate': True,
            'participation_threshold': 0.5,
        }]
    )

    # Robot agents (delayed start) — supports up to 6 robots via num_robots arg
    # Note: LaunchConfiguration is evaluated at runtime; we pre-generate
    # nodes for the maximum count and gate each with a condition.
    max_robots = 6
    robot_nodes = []
    for i in range(max_robots):
        robot_node = TimerAction(
            period=5.0 + i * 2.0,  # Stagger robot startup
            actions=[
                Node(
                    package='fl_robots',
                    executable='robot_agent',
                    name=f'robot_agent_{i}',
                    output='screen',
                    parameters=[{
                        'robot_id': f'robot_{i}',
                        'learning_rate': LaunchConfiguration('learning_rate'),
                        'batch_size': 32,
                        'local_epochs': LaunchConfiguration('local_epochs'),
                        'samples_per_round': 256,
                    }],
                    condition=IfCondition(
                        PythonExpression([str(i), ' < ', LaunchConfiguration('num_robots')])
                    ),
                )
            ]
        )
        robot_nodes.append(robot_node)

    # Coordinator node (delayed to allow robots to register)
    coordinator_node = TimerAction(
        period=15.0,
        actions=[
            Node(
                package='fl_robots',
                executable='coordinator',
                name='coordinator',
                output='screen',
                parameters=[{
                    'total_rounds': LaunchConfiguration('total_rounds'),
                    'min_robots': 2,
                    'round_timeout': 60.0,
                    'evaluation_interval': 5,
                }]
            )
        ]
    )

    # Monitor node (conditional)
    monitor_node = TimerAction(
        period=18.0,
        actions=[
            Node(
                package='fl_robots',
                executable='monitor',
                name='monitor',
                output='screen',
                parameters=[{
                    'output_dir': '/ros2_ws/results',
                    'save_interval': 30.0,
                }],
                condition=IfCondition(LaunchConfiguration('enable_monitor'))
            )
        ]
    )

    # Digital Twin Visualization node
    digital_twin_node = TimerAction(
        period=20.0,
        actions=[
            Node(
                package='fl_robots',
                executable='digital_twin',
                name='digital_twin',
                output='screen',
                parameters=[{
                    'output_dir': '/ros2_ws/results',
                    'update_interval': 5.0,
                }],
                condition=IfCondition(LaunchConfiguration('enable_ui'))
            )
        ]
    )

    # Web Dashboard node
    web_dashboard_node = TimerAction(
        period=22.0,
        actions=[
            Node(
                package='fl_robots',
                executable='web_dashboard',
                name='web_dashboard',
                output='screen',
                parameters=[{
                    'port': 5000,
                    'host': '0.0.0.0',
                    'output_dir': '/ros2_ws/results',
                }],
                condition=IfCondition(LaunchConfiguration('enable_ui'))
            )
        ]
    )

    return LaunchDescription([
        # Launch arguments
        num_robots_arg,
        total_rounds_arg,
        learning_rate_arg,
        batch_size_arg,
        local_epochs_arg,
        enable_monitor_arg,
        enable_ui_arg,

        # Nodes
        aggregator_node,
        *robot_nodes,
        coordinator_node,
        monitor_node,
        digital_twin_node,
        web_dashboard_node,
    ])
