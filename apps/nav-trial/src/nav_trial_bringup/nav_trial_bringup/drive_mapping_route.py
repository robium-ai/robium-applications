#!/usr/bin/env python3
"""Drive an exploration ring around turtlebot3_world under SLAM, then save the map."""
import subprocess
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

# (x, y) WORLD coords. turtlebot3_world: 9 pillars (r=0.15) on a 3x3 grid
# at {-1.1, 0, 1.1}^2 inside a hexagon wall. The naive octagon's diagonal
# points (+-1.2, +-1.2) sit INSIDE the pillars (0.14 m from their centers)
# -> unplannable goals in unknown space, recovery-spin map smear (verified
# live, run 6). This route alternates outer axis points (>=0.45 m clear of
# pillars and wall) with inner diagonal corridor midpoints (0.63 m clear),
# covering both the outer ring and the interior for the lidar.
RING = [
    (-1.7, 0.0), (-0.55, 0.55), (0.0, 1.7), (0.55, 0.55),
    (1.7, 0.0), (0.55, -0.55), (0.0, -1.7), (-0.55, -0.55), (-1.7, 0.0),
]
# slam_toolbox's map frame origin is the robot's STARTING pose, not the
# world origin (verified live: planner reported start (0,0) and "Goal
# Coordinates ... outside bounds" for world-frame goals). Shift the world
# ring by the spawn pose from sim.launch.py.
SPAWN = (-2.0, -0.5)
MAP_OUT = '/ws/maps/map'


def make_pose(nav, x, y):
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x = float(x)
    p.pose.position.y = float(y)
    p.pose.orientation.w = 1.0
    return p


def main():
    rclpy.init()
    nav = BasicNavigator()
    # SLAM provides map->odom; no AMCL, no initial pose needed.
    nav.waitUntilNav2Active(localizer='slam_toolbox')

    reached = 0
    for i, (wx, wy) in enumerate(RING):
        x, y = wx - SPAWN[0], wy - SPAWN[1]  # world -> map frame
        for attempt in (1, 2):  # one retry: map may still be growing
            nav.goToPose(make_pose(nav, x, y))
            while not nav.isTaskComplete():
                rclpy.spin_once(nav, timeout_sec=1.0)
            result = nav.getResult()
            print(f'waypoint {i} world({wx},{wy}) map({x},{y}) '
                  f'attempt {attempt}: {result}', flush=True)
            if result == TaskResult.SUCCEEDED:
                reached += 1
                break

    if reached < len(RING) - 2:  # tolerate a couple of failed waypoints
        print(f'FAIL: only {reached}/{len(RING)} waypoints reached', flush=True)
        sys.exit(1)

    print(f'{reached}/{len(RING)} waypoints reached; saving map...', flush=True)
    save = subprocess.run(
        ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', MAP_OUT,
         '--ros-args', '-p', 'use_sim_time:=true'],
        timeout=60)
    sys.exit(save.returncode)


if __name__ == '__main__':
    main()
