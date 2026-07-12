"""Auto-initialize the demo session: set AMCL's initial pose, wait for Nav2,
measure RTF, then log the readiness line the demo smoke greps for.

Runs inside demo.launch.py. Exits 0 when ready (the launch keeps running);
without this node the stack sits unlocalized forever — the documented
interactive-bringup abort (learnings 2026-07-10).
"""
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator

INITIAL_POSE = (0.0, 0.0)  # map frame == SLAM start == world (-2.0, -0.5)


def main():
    rclpy.init()
    nav = BasicNavigator()
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = nav.get_clock().now().to_msg()
    pose.pose.position.x, pose.pose.position.y = INITIAL_POSE
    pose.pose.orientation.w = 1.0
    nav.setInitialPose(pose)
    nav.waitUntilNav2Active()  # republishes initial pose until /amcl_pose

    # RTF over ~10 s wall: sim clock (use_sim_time) vs monotonic.
    sim0 = nav.get_clock().now().nanoseconds
    wall0 = time.monotonic()
    while time.monotonic() - wall0 < 10.0:
        rclpy.spin_once(nav, timeout_sec=0.5)
    rtf = (nav.get_clock().now().nanoseconds - sim0) / 1e9 / (time.monotonic() - wall0)

    print(f'DEMO READY rtf={rtf:.2f}', flush=True)
    nav.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
