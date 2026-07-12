"""Navigation scenario: sim + map_server + AMCL + Nav2 navigation servers.

Same direct-server composition as slam.launch.py (see its docstring for why
nav2_bringup's launch files were rejected: no bond_timeout control in
navigation_launch.py, and the TB3 params' $(find-pkg-share) substitutions
need ParameterFile(allow_substs=True)). This file swaps the SLAM source for
localization on the saved map: nav2_map_server + nav2_amcl, lifecycle-managed
by the same manager (map_server and amcl first in node_names, so the map and
localization exist before the costmaps/planner activate).

Map path: nav2_params.yaml ships `yaml_filename: "map.yaml"` (relative →
resolved against the node's cwd → broken). Overridden here with the absolute
installed path via a per-node params dict after the ParameterFile.

AMCL publishes map->odom only after it gets an initial pose — send_goals.py's
setInitialPose is mandatory, this launch alone leaves the robot unlocalized.
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

    # allow_substs expands the $(find-pkg-share ...) substitutions the TB3
    # params file uses for BT xml paths (navigation_launch.py does the same
    # via RewrittenYaml; without it bt_navigator gets the literal string).
    params = ParameterFile(
        os.path.join(pkg, 'config', 'nav2_params.yaml'), allow_substs=True)
    map_yaml = os.path.join(pkg, 'maps', 'map.yaml')
    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sim.launch.py')))

    def server(package, executable, name, extra_remaps=(), extra_params=()):
        return Node(
            package=package, executable=executable, name=name,
            output='screen', respawn=True, respawn_delay=2.0,
            parameters=[params] + list(extra_params),
            remappings=remappings + list(extra_remaps))

    nav2_nodes = [
        # Localization on the saved map (replaces slam.launch.py's
        # slam_toolbox include).
        server('nav2_map_server', 'map_server', 'map_server',
               extra_params=[{'yaml_filename': map_yaml}]),
        server('nav2_amcl', 'amcl', 'amcl'),
        # Navigation servers — identical set to slam.launch.py.
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
                    'map_server',
                    'amcl',
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

    return LaunchDescription([sim] + nav2_nodes)
