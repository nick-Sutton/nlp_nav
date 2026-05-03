#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    turtlebot3_gazebo_pkg = get_package_share_directory('turtlebot3_gazebo')
    launch_file_dir = os.path.join(turtlebot3_gazebo_pkg, 'launch')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    nlp_nav_pkg = get_package_share_directory('nlp_nav')

    use_sim_time    = LaunchConfiguration('use_sim_time',    default='true')
    x_pose          = LaunchConfiguration('x_pose',          default='-3.0')
    y_pose          = LaunchConfiguration('y_pose',          default='1.0')
    yaw             = LaunchConfiguration('yaw',             default='0.0')
    moving_obstacles = LaunchConfiguration('moving_obstacles', default='false')

    world_static  = os.path.join(nlp_nav_pkg, 'worlds', 'turtlebot3_house_static.world')
    world_dynamic = os.path.join(nlp_nav_pkg, 'worlds', 'turtlebot3_house.world')

    gzserver_static = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s -v2 ', world_static], 'on_exit_shutdown': 'true'}.items(),
        condition=IfCondition(PythonExpression(["'", moving_obstacles, "' == 'false'"]))
    )

    gzserver_dynamic = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s -v2 ', world_dynamic], 'on_exit_shutdown': 'true'}.items(),
        condition=IfCondition(PythonExpression(["'", moving_obstacles, "' == 'true'"]))
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v2 ', 'on_exit_shutdown': 'true'}.items()
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose,
            'yaw':    yaw,
        }.items()
    )

    set_env_vars_resources = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(turtlebot3_gazebo_pkg, 'models')
    )

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument('moving_obstacles', default_value='false',
                                        description='Launch world with moving obstacles'))
    ld.add_action(gzserver_static)
    ld.add_action(gzserver_dynamic)
    ld.add_action(gzclient_cmd)
    ld.add_action(spawn_turtlebot_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(set_env_vars_resources)

    return ld
