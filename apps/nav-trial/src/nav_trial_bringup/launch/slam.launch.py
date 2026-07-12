"""SLAM scenario: sim + slam_toolbox (online_async) + Nav2 navigation servers.

Why not nav2_bringup's launch files (both were tried live and failed):

1. bringup_launch.py with slam:=True — on jazzy its slam_launch.py starts
   its OWN slam_toolbox (online_sync) + map_saver_server + a second
   lifecycle manager, so adding online_async alongside duplicates SLAM:
   two map->odom publishers, lifecycle transition errors, bond deaths,
   0/9 waypoints.
2. navigation_launch.py — correct node set, but it hard-codes the
   lifecycle manager's parameters (autostart + node_names only), so
   `bond_timeout` cannot be configured. On Docker Desktop/macOS the
   whole container stalls for several seconds under the activation load
   spike (gz + ceres + DWB in one 8 GB VM); the default 4 s bond timeout
   then declares controller_server dead and shuts the stack down
   ("CRITICAL FAILURE: SERVER controller_server IS DOWN"), 0/9 again.

So the Nav2 servers are launched directly (same nodes/remappings as
navigation_launch.py jazzy, minus route_server and docking_server which
this app does not use) with our own lifecycle manager and
`bond_timeout: 0.0` (bond checking off — Nav2's documented escape hatch
for platforms with scheduling hiccups). respawn stays on per the nav2
skill's iterate-with-standalone-nodes gotcha.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile


def generate_launch_description():
    pkg = get_package_share_directory('nav_trial_bringup')
    slam_toolbox = get_package_share_directory('slam_toolbox')

    # allow_substs expands the $(find-pkg-share ...) substitutions the TB3
    # params file uses for BT xml paths (navigation_launch.py does the same
    # via RewrittenYaml; without it bt_navigator gets the literal string).
    params = ParameterFile(
        os.path.join(pkg, 'config', 'nav2_params.yaml'), allow_substs=True)
    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sim.launch.py')))

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox, 'launch', 'online_async_launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'slam_params_file': os.path.join(pkg, 'config', 'slam_params.yaml'),
        }.items())

    def server(package, executable, name, extra_remaps=()):
        return Node(
            package=package, executable=executable, name=name,
            output='screen', respawn=True, respawn_delay=2.0,
            parameters=[params],
            remappings=remappings + list(extra_remaps))

    nav2_nodes = [
        server('nav2_controller', 'controller_server', 'controller_server',
               [('cmd_vel', 'cmd_vel_nav')]),
        server('nav2_smoother', 'smoother_server', 'smoother_server'),
        server('nav2_planner', 'planner_server', 'planner_server'),
        server('nav2_behaviors', 'behavior_server', 'behavior_server',
               [('cmd_vel', 'cmd_vel_nav')]),
        server('nav2_bt_navigator', 'bt_navigator', 'bt_navigator'),
        server('nav2_waypoint_follower', 'waypoint_follower',
               'waypoint_follower'),
        server('nav2_velocity_smoother', 'velocity_smoother',
               'velocity_smoother', [('cmd_vel', 'cmd_vel_nav')]),
        server('nav2_collision_monitor', 'collision_monitor',
               'collision_monitor'),
        Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_navigation', output='screen',
            parameters=[{
                'use_sim_time': True,
                'autostart': True,
                'bond_timeout': 0.0,
                'node_names': [
                    'controller_server',
                    'smoother_server',
                    'planner_server',
                    'behavior_server',
                    'velocity_smoother',
                    'collision_monitor',
                    'bt_navigator',
                    'waypoint_follower',
                ],
            }]),
    ]

    return LaunchDescription([sim, slam] + nav2_nodes)
