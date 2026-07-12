"""Demo scenario: the nav stack (nav.launch.py: sim + bridge + Nav2 on the
saved map) plus demo_init, which auto-sets AMCL's initial pose so a Foxglove
visitor can click goals immediately. foxglove_bridge starts with the launch
(listens on :8765 within seconds) — Cloud Run's startup probe passes while
gz/Nav2 are still booting, and the visitor watches topics come alive.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('nav_trial_bringup')
    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'nav.launch.py')))
    init = Node(
        package='nav_trial_bringup', executable='demo_init', name='demo_init',
        output='screen', parameters=[{'use_sim_time': True}])
    return LaunchDescription([nav, init])
