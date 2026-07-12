#!/usr/bin/env python3
"""Send nav goals on the saved map; exit 0 iff all SUCCEEDED within --timeout.

Frame convention (matters — Task 5 learning): the saved map's frame origin is
the robot's SLAM STARTING pose, not the Gazebo world origin. The robot spawns
at world (-2.0, -0.5), which is map (0, 0) yaw 0 — that is the initial pose
sent to AMCL below. All goals (--goals and DEFAULT_GOALS) are in MAP frame;
to convert a Gazebo world coordinate: (mx, my) = (wx + 2.0, wy + 0.5).
Default goals are known-good free cells from the Task 5 SLAM run.
"""
import argparse
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

INITIAL_POSE = (0.0, 0.0)  # map frame == SLAM start == world (-2.0, -0.5)
DEFAULT_GOALS = '3.7,0.5;0.3,0.5'  # map frame


def make_pose(nav, x, y):
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x = float(x)
    p.pose.position.y = float(y)
    p.pose.orientation.w = 1.0
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--goals', default=DEFAULT_GOALS,
                    help='semicolon-separated x,y pairs in MAP frame '
                         '(map = world + (2.0, 0.5))')
    ap.add_argument('--timeout', type=float, default=300.0,
                    help='wall-clock seconds for the WHOLE run')
    args = ap.parse_args(argv)
    goals = [tuple(map(float, g.split(','))) for g in args.goals.split(';')]

    rclpy.init()
    nav = BasicNavigator()
    nav.setInitialPose(make_pose(nav, *INITIAL_POSE))  # AMCL is silent without this
    nav.waitUntilNav2Active()  # default localizer 'amcl' — republishes initial pose until amcl_pose arrives

    deadline = time.monotonic() + args.timeout
    for i, (x, y) in enumerate(goals):
        nav.goToPose(make_pose(nav, x, y))
        while not nav.isTaskComplete():
            if time.monotonic() > deadline:
                print(f'FAIL: timeout during goal {i} ({x},{y})', flush=True)
                nav.cancelTask()
                return 1
            rclpy.spin_once(nav, timeout_sec=1.0)
        result = nav.getResult()
        print(f'goal {i} ({x},{y}): {result}', flush=True)
        if result != TaskResult.SUCCEEDED:
            print('FAIL: goal not reached', flush=True)
            return 1

    print('PASS: all goals reached', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
