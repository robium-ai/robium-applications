"""Ground-truth-driven pick-and-place. The demo's "before" and its data source.

This is NOT a policy — it can see the cube's exact pose, which is precisely why
it works and why it proves nothing about learning. It exists to (a) put a moving
arm on screen on day one and (b) generate the demonstrations SmolVLA trains on.

STATUS (Task 5): THE CANARY IS RED — this oracle does NOT reach 10/10. It does
not reliably grasp the cube at all. The cause is a structural mismatch between
the SO-101's pincer and menagerie's `box`, diagnosed in detail in
.superpowers/sdd/task-5-report.md. The short version:

  * The 5-DOF arm can only reach the cube-spawn region with its fingers pointing
    ~32 deg BELOW horizontal (forward-and-down). This is not a solver artifact:
    a wrist orientation with level fingers is genuinely unreachable there
    (position residual 9.4 cm, and the pose stalls even with the cube removed).
  * The cube (menagerie `box`: half-extents 0.02/0.02/0.03, so 4x4x6 cm) is only
    held in the JAW THROAT — gripper-local z ~= -0.035, up near the jaw root —
    NOT at the fingertips (local z ~= -0.101). Verified empirically by teleporting
    the cube to candidate gripper-local offsets, closing, and lifting.
  * Those two facts are incompatible. Putting the throat at the cube's height
    (z=0.03) with down-forward fingers drives the fingertips BELOW the table
    (z ~= -0.007). Aiming the fingertips at the cube instead makes the fixed
    jaw's collision mesh shaft (`wrist_roll_follower_so101_gripper_part0_v1`,
    which spans gripper-local z -0.106..-0.010) pass straight through the cube's
    volume: the arm stalls (joint error 0.13-0.24 rad, shoulder_lift and
    elbow_flex saturated at their +-2.94 N*m forcerange) and the jaws close on
    empty air (gripper settles at its fully-closed command, -0.175).

The waypoint machine below is left in the most principled form found (throat
reference + approach along the finger axis) so the next fix has a base to build
on, but it is NOT a working oracle and MUST NOT be used to record the dataset in
Task 6 until the grasp is solved. See the report for the recommended fixes.
"""

import mujoco
import numpy as np

from vla_trial.config import (
    EE_SITE,
    MAX_EPISODE_STEPS,
    ORACLE_APPROACH_HEIGHT,
    ORACLE_GRASP_CORRECTION_ITERS,
    ORACLE_GRASP_LOCAL,
    ORACLE_GRASP_SETTLE_STEPS,
    ORACLE_LIFT_HEIGHT,
    ORACLE_SETTLE_STEPS,
    ORACLE_STANDOFF,
)
from vla_trial.env.ik import solve_ik

# GRIPPER POLARITY — determined empirically, NOT from any doc (the upstream
# SO-ARM100 repo says 0=closed/100=open but admits its MJCF does not reflect
# that, so it cannot be trusted). Method: drive the gripper joint to each end of
# the actuator's ctrlrange and measure the fingertip gap
# (fixed_jaw_sph_tip1 <-> moving_jaw_sph_tip1):
#
#     ctrl = -0.17453 (ctrlrange LOW)  -> fingertip gap 0.0041 m  => CLOSED
#     ctrl =  0.0                      -> fingertip gap 0.0163 m
#     ctrl = +1.74533 (ctrlrange HIGH) -> fingertip gap 0.1334 m  => OPEN
#
# So: ctrlrange LOW = closed, HIGH = open. Reading `lo, hi = ctrlrange` and
# calling `hi` "open" is therefore correct for this model — no swap needed.


class OraclePolicy:
    """A waypoint state machine: approach -> advance -> grasp -> lift -> place -> release."""

    def __init__(self, env):
        self.env = env
        lo, hi = env.model.actuator_ctrlrange[5]
        self.open_cmd = float(hi)     # see GRIPPER POLARITY note above
        self.closed_cmd = float(lo)

        self._gripper_bid = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
        )
        self.reset()

    def reset(self) -> None:
        self._phase = 0
        self._hold = 0

    # --- kinematics helpers -------------------------------------------------
    def _grasp_point(self, qpos_arm: np.ndarray) -> np.ndarray:
        """World position of the jaw THROAT (the point that actually holds the
        cube) for a given arm configuration. ORACLE_GRASP_LOCAL is a
        gripper-body-local offset, so it rotates with the wrist — it cannot be
        precomputed as a world vector."""
        model = self.env.model
        s = mujoco.MjData(model)
        s.qpos[:5] = qpos_arm
        s.qpos[5] = self.open_cmd
        mujoco.mj_forward(model, s)
        rot = s.xmat[self._gripper_bid].reshape(3, 3)
        return s.xpos[self._gripper_bid] + rot @ ORACLE_GRASP_LOCAL

    def _finger_dir(self, qpos_arm: np.ndarray) -> np.ndarray:
        """World unit vector the fingers point along (gripper-local -z)."""
        model = self.env.model
        s = mujoco.MjData(model)
        s.qpos[:5] = qpos_arm
        mujoco.mj_forward(model, s)
        rot = s.xmat[self._gripper_bid].reshape(3, 3)
        return rot @ np.array([0.0, 0.0, -1.0])

    def _solve_grasp_ik(self, target: np.ndarray) -> np.ndarray:
        """Solve IK so the jaw THROAT (not the `gripperframe` site, and not the
        fingertips) lands on `target`.

        `solve_ik` drives the EE_SITE, but the site is not where the cube is
        held — it sits at gripper-local (0.012, ~0, -0.098), out near the
        fingertips, ~2 cm off the pinch axis. Position-only IK also leaves the
        wrist orientation free, so the site-to-throat offset is rotated by
        whatever orientation the solver lands on and cannot simply be subtracted
        once. Fixed-point correction instead: solve toward a site target, measure
        where the throat actually ended up, shift the site target by the
        residual, repeat. Converges to sub-mm within a few passes.
        """
        model, data = self.env.model, self.env.data
        goal = np.asarray(target, dtype=np.float64)
        site_target = goal.copy()
        q = solve_ik(model, data, EE_SITE, site_target)
        for _ in range(ORACLE_GRASP_CORRECTION_ITERS):
            site_target = site_target + (goal - self._grasp_point(q))
            q = solve_ik(model, data, EE_SITE, site_target)
        return q

    # --- waypoints ----------------------------------------------------------
    def _waypoints(self) -> list[tuple[np.ndarray, float, int]]:
        cube = self.env.cube_pos.astype(np.float64)
        bin_ = self.env.bin_pos.astype(np.float64)

        # Back the pre-grasp off ALONG THE FINGER AXIS, not straight up: the
        # fingers point forward-and-down, so a purely vertical descent drives
        # the fixed jaw's shaft onto the cube. Retreating along -finger_dir
        # keeps the shaft inside the volume the fingertips already swept.
        finger = self._finger_dir(self._solve_grasp_ik(cube))
        pre = cube - finger * ORACLE_STANDOFF

        lifted = cube + np.array([0.0, 0.0, ORACLE_LIFT_HEIGHT])
        # Dead-centre over the bin: Task 4 measured the clear-drop margin at
        # only +-0.035 from bin centre (releasing at +-0.04 perches the cube on
        # the rim), well inside the +-0.055 IK-reachable footprint. So this
        # deliberately does not chase the corners of that footprint.
        above_bin = bin_ + np.array([0.0, 0.0, ORACLE_LIFT_HEIGHT])

        return [
            # 0 approach (above the line-up point)
            (pre + np.array([0.0, 0.0, ORACLE_APPROACH_HEIGHT]), self.open_cmd, ORACLE_SETTLE_STEPS),
            (pre, self.open_cmd, ORACLE_SETTLE_STEPS),           # 1 line up behind the cube
            (cube, self.open_cmd, ORACLE_SETTLE_STEPS),          # 2 advance: cube enters the throat
            (cube, self.closed_cmd, ORACLE_GRASP_SETTLE_STEPS),  # 3 grasp
            (lifted, self.closed_cmd, ORACLE_SETTLE_STEPS),      # 4 lift
            (above_bin, self.closed_cmd, ORACLE_SETTLE_STEPS),   # 5 traverse over the bin
            (above_bin, self.open_cmd, ORACLE_SETTLE_STEPS),     # 6 release
        ]

    def act(self, obs: dict) -> np.ndarray:
        waypoints = self._waypoints()
        self._phase = min(self._phase, len(waypoints) - 1)
        target, gripper, settle_steps = waypoints[self._phase]

        arm = self._solve_grasp_ik(target)

        self._hold += 1
        if self._hold >= settle_steps:
            self._hold = 0
            self._phase += 1

        return np.concatenate([arm, [gripper]]).astype(np.float32)

    @property
    def done(self) -> bool:
        return self._phase >= len(self._waypoints())


def rollout(env, seed: int, logger=None) -> dict:
    """One oracle episode. Returns success + the frames a dataset needs."""
    obs, _ = env.reset(seed=seed)
    policy = OraclePolicy(env)

    frames: list[dict] = []
    success = False

    for step in range(MAX_EPISODE_STEPS):
        action = policy.act(obs)
        frames.append(
            {
                "observation.images.wrist": obs["observation.images.wrist"],
                "observation.images.scene": obs["observation.images.scene"],
                "observation.state": obs["observation.state"],
                "action": action,
            }
        )
        if logger is not None:
            logger.log_step(step, obs, action, task=env.task)

        obs, _reward, terminated, truncated, info = env.step(action)
        success = bool(info["is_success"])
        if terminated or truncated:
            break

    return {"success": success, "frames": frames, "n_steps": len(frames)}
