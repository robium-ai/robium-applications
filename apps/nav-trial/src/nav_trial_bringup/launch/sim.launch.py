"""Headless sim bringup: gz server (software rendering) + TB3 burger + bridge.

Path B composition (Task 3 Step 1 evidence): upstream
turtlebot3_world.launch.py hardcodes a gz GUI client and non-overridable
server gz_args, so we include ros_gz_sim's gz_sim.launch.py ourselves with
`-s -r --headless-rendering` and reuse the upstream sub-launch files:
- spawn_turtlebot3.launch.py: spawns the model AND starts the ros_gz
  parameter_bridge (params/turtlebot3_burger_bridge.yaml: /clock, /odom,
  /tf, /cmd_vel, /imu, /scan, /joint_states).
- robot_state_publisher.launch.py: rsp with the TB3 urdf.
Plus foxglove_bridge on :8765. Everything runs with use_sim_time.
Tasks 5/6 IncludeLaunchDescription this file.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = os.path.join(tb3_gazebo, 'worlds', 'turtlebot3_world.world')

    set_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', os.path.join(tb3_gazebo, 'models'))

    # Server only (-s), running (-r), software rendering for the gpu_lidar
    # in a GPU-less/headless container (R1).
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': ['-r -s --headless-rendering -v2 ', world],
            'on_exit_shutdown': 'true',
        }.items(),
    )

    # Spawns TURTLEBOT3_MODEL (burger via container env) and starts the
    # ros_gz parameter_bridge from turtlebot3_burger_bridge.yaml.
    spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo, 'launch', 'spawn_turtlebot3.launch.py')),
        launch_arguments={'x_pose': '-2.0', 'y_pose': '-0.5'}.items(),
    )

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo, 'launch',
                         'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    foxglove = Node(
        package='foxglove_bridge', executable='foxglove_bridge',
        parameters=[{'port': 8765, 'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription([
        set_resource_path,
        gzserver,
        spawn,
        rsp,
        foxglove,
    ])
