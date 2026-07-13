"""Auto-initialize the demo session, then keep writing stack status.

Sets AMCL's initial pose (the documented interactive-bringup abort otherwise),
waits for Nav2, measures RTF, then loops forever: subscribes /rosout and
writes /tmp/demo_status.json every 2 s for the gateway's /status endpoint.
"""
import json
import os
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator
from rcl_interfaces.msg import Log

INITIAL_POSE = (0.0, 0.0)  # map frame == SLAM start == world (-2.0, -0.5)
STATUS_PATH = '/tmp/demo_status.json'
LOG_KEEP = 40
START = time.time()


def write_status(nav, ready, rtf, log_ring):
    status = {
        'start': START,
        'ready': ready,
        'rtf': rtf,
        'nodes': len(nav.get_node_names()),
        'log': list(log_ring),
    }
    with open(STATUS_PATH + '.tmp', 'w') as f:
        json.dump(status, f)
    os.replace(STATUS_PATH + '.tmp', STATUS_PATH)


def main():
    rclpy.init()
    nav = BasicNavigator()
    log_ring = []

    def on_log(msg: Log):
        line = f'[{msg.name}] {msg.msg}'
        log_ring.append(line[:160])
        del log_ring[:-LOG_KEEP]

    nav.create_subscription(Log, '/rosout', on_log, 10)

    write_status(nav, False, None, log_ring)
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = nav.get_clock().now().to_msg()
    pose.pose.position.x, pose.pose.position.y = INITIAL_POSE
    pose.pose.orientation.w = 1.0
    nav.setInitialPose(pose)
    nav.waitUntilNav2Active()

    sim0 = nav.get_clock().now().nanoseconds
    wall0 = time.monotonic()
    while time.monotonic() - wall0 < 10.0:
        rclpy.spin_once(nav, timeout_sec=0.5)
    rtf = (nav.get_clock().now().nanoseconds - sim0) / 1e9 / (time.monotonic() - wall0)
    print(f'DEMO READY rtf={rtf:.2f}', flush=True)

    last = 0.0
    while rclpy.ok():
        rclpy.spin_once(nav, timeout_sec=0.5)
        if time.monotonic() - last >= 2.0:
            write_status(nav, True, round(rtf, 2), log_ring)
            last = time.monotonic()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
