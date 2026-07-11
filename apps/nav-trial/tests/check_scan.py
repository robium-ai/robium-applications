#!/usr/bin/env python3
"""Exit 0 iff /scan publishes at least one non-empty LaserScan within TIMEOUT."""
import os
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

TIMEOUT_S = float(os.environ.get('CHECK_SCAN_TIMEOUT', '90'))


class ScanCheck(Node):
    def __init__(self):
        super().__init__('check_scan')
        self.ok = False
        self.create_subscription(LaserScan, '/scan', self._cb, 10)

    def _cb(self, msg):
        if len(msg.ranges) > 0:
            self.ok = True


def main():
    rclpy.init()
    node = ScanCheck()
    end = node.get_clock().now().nanoseconds / 1e9 + TIMEOUT_S
    while rclpy.ok() and not node.ok and (node.get_clock().now().nanoseconds / 1e9) < end:
        rclpy.spin_once(node, timeout_sec=1.0)
    print('PASS: /scan publishing' if node.ok else 'FAIL: no /scan within timeout')
    sys.exit(0 if node.ok else 1)


if __name__ == '__main__':
    main()
