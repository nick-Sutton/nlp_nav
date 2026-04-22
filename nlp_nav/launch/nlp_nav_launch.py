#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    turtlebot3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')
    nav2_bringup_pkg      = get_package_share_directory('nav2_bringup')
    pf_pkg                = get_package_share_directory('nlp_nav')

    map_yaml      = os.path.join(pf_pkg, 'maps', 'map.yaml')
    nav2_params   = os.path.join(pf_pkg, 'config', 'nav2_params.yaml')
    nav2_rviz_cfg = os.path.join(nav2_bringup_pkg, 'rviz', 'nav2_default_view.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Gazebo + TurtleBot3
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_gazebo_pkg, 'launch', 'turtlebot3_house.launch.py')
        )
    )

    # Static map
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

    # Particle filter localiser — broadcasts dynamic map->odom TF and publishes /amcl_pose
    particle_filter = Node(
        package='nlp_nav',
        executable='particleFilter.py',
        name='particle_filter_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Nav2 stack (controller_server with DWA, planner_server, bt_navigator, behaviours)
    nav2_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_pkg, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file':  nav2_params,
        }.items(),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', nav2_rviz_cfg],
        output='screen',
    )

    return LaunchDescription([
        gazebo,
        map_server,
        map_lifecycle,
        particle_filter,
        nav2_navigation,
        rviz,
    ])
