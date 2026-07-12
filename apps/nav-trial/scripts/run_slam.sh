#!/usr/bin/env bash
# ROS setup.bash is not `set -u` clean (AMENT_TRACE_SETUP_FILES unbound) —
# source first, tighten after.
source /opt/ros/${ROS_DISTRO}/setup.bash
source /ws/install/setup.bash
set -uo pipefail

mkdir -p /ws/maps
ros2 launch nav_trial_bringup slam.launch.py &
LAUNCH_PID=$!

ros2 run nav_trial_bringup drive_mapping_route
RC=$?

# Bounded shutdown: gz teardown can hang headless; don't let it wedge the
# container after the scenario result is already decided.
kill -INT ${LAUNCH_PID} 2>/dev/null
for _ in $(seq 1 20); do
  kill -0 ${LAUNCH_PID} 2>/dev/null || break
  sleep 1
done
kill -9 ${LAUNCH_PID} 2>/dev/null
exit ${RC}
