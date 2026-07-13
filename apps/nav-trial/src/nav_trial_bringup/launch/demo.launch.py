"""Demo scenario: the nav stack (nav.launch.py: sim + bridge + Nav2 on the
saved map) plus demo_init (auto initial pose + live status file) plus the
session gateway that owns the public port.

Port layout: gateway on $PORT (8765, Cloud Run's routed port) tunnels
WebSocket upgrades to foxglove_bridge on internal :8766 and serves
/status + /shutdown itself. The gateway listens within ~1 s of launch —
Cloud Run's startup probe passes while gz/Nav2 are still booting, and the
visitor watches topics come alive.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('nav_trial_bringup')
    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'nav.launch.py')),
        launch_arguments={'bridge_port': '8766'}.items())
    init = Node(
        package='nav_trial_bringup', executable='demo_init', name='demo_init',
        output='screen', parameters=[{'use_sim_time': True}])
    gateway = ExecuteProcess(
        cmd=['python3', '/ws/scripts/demo_gateway.py'], output='screen')
    return LaunchDescription([nav, init, gateway])
