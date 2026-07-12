#!/usr/bin/env python3
"""The nav-trial pass bar: reuse send_goals with a CI-sized timeout.

Runs under the container's /entrypoint.sh (compose `command:` wrapper), which
sources the /ws/install overlay — so the package import below needs no
sys.path surgery.
"""
import os
import sys

from nav_trial_bringup.send_goals import main

# Timeout sizing: expected scenario ~90 s sim time; measured RTF ~= 0.99
# (Task 3: cumulative sim 93.4 s / real 94.3 s) => ~91 s wall; x2 safety
# margin => 180 s default. Override with SMOKE_TIMEOUT.
# NOTE: this bounds only the goal-following phase — send_goals starts its
# clock AFTER waitUntilNav2Active(). run_smoke.sh wraps the whole run in an
# outer `timeout` so a never-activating stack still fails bounded.
DEFAULT_TIMEOUT = '180'

if __name__ == '__main__':
    timeout = os.environ.get('SMOKE_TIMEOUT', DEFAULT_TIMEOUT)
    sys.exit(main(['--timeout', timeout]))
