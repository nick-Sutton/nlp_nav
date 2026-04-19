#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    turtlebot3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    nav2_bringup_pkg = get_package_share_directory('nav2_bringup')
    nav2_rviz_config = os.path.join(nav2_bringup_pkg, 'rviz', 'nav2_default_view.rviz')
    pf_pkg = get_package_share_directory('nlp_nav')
    map_yaml = os.path.join(pf_pkg, 'maps', 'map.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    pre_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(turtlebot3_gazebo_pkg, 'launch', 'turtlebot3_house.launch.py')
            )
        )

    # Note: /scan, /cmd_vel, /clock, /odom, /tf are already bridged by
    # turtlebot3_house.launch.py via turtlebot3_burger_bridge.yaml

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', nav2_rviz_config],
        output='screen'
    )

    transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml}],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[
            {'autostart': True},
            {'node_names': ['map_server']},
        ],
    )

    ld = LaunchDescription()
    ld.add_action(pre_launch)

    ld.add_action(map_server)
    ld.add_action(lifecycle_manager)

    ld.add_action(rviz)
    ld.add_action(transform)

    return ld
