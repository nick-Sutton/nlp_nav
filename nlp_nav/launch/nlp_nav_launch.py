#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    nav2_bringup_pkg = get_package_share_directory('nav2_bringup')
    pf_pkg = get_package_share_directory('nlp_nav')

    nav2_rviz_config = os.path.join(nav2_bringup_pkg, 'rviz', 'nav2_default_view.rviz')
    map_yaml = os.path.join(pf_pkg, 'maps', 'map.yaml')
    nav2_params = os.path.join(pf_pkg, 'config', 'nav2_params.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Gazebo + TurtleBot3 — uses our custom world with dynamic obstacle actors
    pre_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pf_pkg, 'launch', 'turtlebot3_house_custom.launch.py')
        )
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', nav2_rviz_config],
        output='screen',
    )

    # Static map with its own lifecycle manager
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'yaml_filename': map_yaml}],
    )

    map_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': True},
            {'node_names': ['map_server']},
        ],
    )

    # Particle filter localiser — broadcasts dynamic map→odom TF
    particle_filter = Node(
        package='nlp_nav',
        executable='particleFilter.py',
        name='particle_filter_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Nav2 controller (brings local costmap).
    # Publishes Twist to /cmd_vel_raw; the bridge wraps it as TwistStamped on
    # /cmd_vel to match ros_gz_bridge's subscriber type.
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_params, {'use_sim_time': use_sim_time}],
        remappings=[('cmd_vel', 'cmd_vel_raw')],
    )

    cmd_vel_bridge = Node(
        package='nlp_nav',
        executable='cmd_vel_bridge.py',
        name='cmd_vel_bridge',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    nav2_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': True},
            {'node_names': ['controller_server']},
        ],
    )

    obstacle_mover = Node(
        package='nlp_nav',
        executable='obstacle_mover.py',
        name='obstacle_mover',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Kill any lingering gz sim processes when the launch is shut down (Ctrl+C)
    kill_gz_on_shutdown = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[
                ExecuteProcess(
                    cmd=['bash', '-c', 'pkill -9 -f "gz sim" || true'],
                    output='screen',
                )
            ]
        )
    )

    ld = LaunchDescription()
    ld.add_action(kill_gz_on_shutdown)
    ld.add_action(pre_launch)
    ld.add_action(map_server)
    ld.add_action(map_lifecycle)
    ld.add_action(particle_filter)
    ld.add_action(controller_server)
    ld.add_action(cmd_vel_bridge)
    ld.add_action(nav2_lifecycle)
    ld.add_action(obstacle_mover)
    ld.add_action(rviz)

    return ld
