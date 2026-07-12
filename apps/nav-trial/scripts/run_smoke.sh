#!/usr/bin/env bash
# ROS setup.bash is not `set -u` clean (AMENT_TRACE_SETUP_FILES unbound) —
# source first, tighten after.
source /opt/ros/${ROS_DISTRO}/setup.bash
source /ws/install/setup.bash
set -uo pipefail

ros2 launch nav_trial_bringup nav.launch.py &
LAUNCH_PID=$!

# Outer bound is mandatory: send_goals.main() calls waitUntilNav2Active()
# BEFORE its --timeout clock starts, so a never-activating stack hangs forever
# without it. On timeout RC=124 propagates; -k 10 hard-kills a client that
# ignores SIGTERM. Sizing: ~90 s sim @ measured RTF~=0.99 (Task 3) => ~91 s
# wall, x2 margin => 180 default; override with SMOKE_TIMEOUT.
timeout -k 10 "${SMOKE_TIMEOUT:-180}" python3 /ws/tests/smoke_nav.py
RC=$?

# Bounded shutdown: gz teardown can hang headless; don't let it wedge the
# container after the scenario result is already decided.
kill -INT ${LAUNCH_PID} 2>/dev/null
for _ in $(seq 1 20); do
  kill -0 ${LAUNCH_PID} 2>/dev/null || break
  sleep 1
done
kill -9 ${LAUNCH_PID} 2>/dev/null
echo "SMOKE RESULT: ${RC}"
exit ${RC}
