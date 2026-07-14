"""Ground-truth-driven pick-and-place. The demo's "before" and its data source.

This is NOT a policy — it can see the cube's exact pose, which is precisely why
it works and why it proves nothing about learning. It exists to (a) put a moving
arm on screen on day one and (b) generate the demonstrations SmolVLA trains on.

Getting this to grasp at all took three corrections, each found by measurement
rather than by reading the MJCF. All three are load-bearing; drop any one and the
success rate collapses to zero:

  1. THE PEDESTAL (scene_pick.xml). With the cube on the floor, the 5-DOF arm can
     only reach it with the fingers pitched ~32 deg BELOW horizontal, and at that
     pitch the fixed jaw's collision-mesh shaft sweeps straight through the cube:
     the arm stalls (shoulder_lift/elbow_flex saturated at their +-2.94 N*m
     forcerange) and the jaws close on empty air.

  2. THE WRIST ROLL (`_solve_roll`). Position-only IK leaves `wrist_roll` free and
     it lands near 0 — where the jaws' pinch axis is ~91% VERTICAL. The jaws were
     trying to span the cube's 6 cm HEIGHT with a 4.2 cm aperture: geometrically
     impossible, which is why they always closed on air. Rolling the wrist so the
     pinch axis is HORIZONTAL makes them span the cube's 4 cm WIDTH instead. This
     single fix took the oracle from 0/10 to 8/10.

  3. THE GRASP OFFSET (ORACLE_GRASP_LOCAL). The `gripperframe` site is NOT the
     grasp point; the real one was calibrated empirically. See config.py.

Approach is along the FINGER AXIS, not straight down — descending vertically
lands the jaw shaft on top of the cube.
"""

import mujoco
import numpy as np

from vla_trial.config import (
    EE_SITE,
    IK_N_ARM_JOINTS,
    MAX_EPISODE_STEPS,
    ORACLE_APPROACH_HEIGHT,
    ORACLE_GRASP_CORRECTION_ITERS,
    ORACLE_GRASP_LOCAL,
    ORACLE_GRASP_SETTLE_STEPS,
    ORACLE_LIFT_HEIGHT,
    ORACLE_RELEASE_HEIGHT,
    ORACLE_ROLL_ITERS,
    ORACLE_ROLL_SCAN,
    ORACLE_SETTLE_STEPS,
    ORACLE_STANDOFF,
    SCENE_XML,
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
# So ctrlrange LOW = closed, HIGH = open — no swap needed. Polarity was never the
# bug; the jaws opened and closed correctly all along. Them settling at the
# fully-closed command was the *symptom* of closing on empty air.

# The jaws separate along the gripper body's local +x axis (measured: at the
# cube-width jaw angle the fixed->moving fingertip vector is (+0.042, 0, 0) in
# gripper-local coords — i.e. pure local x).
_PINCH_AXIS_LOCAL = np.array([1.0, 0.0, 0.0])
# The fingers point along the gripper body's local -z axis.
_FINGER_AXIS_LOCAL = np.array([0.0, 0.0, -1.0])


class OraclePolicy:
    """A waypoint state machine: approach → line up → advance → grasp → lift → place → release."""

    def __init__(self, env):
        self.env = env
        lo, hi = env.model.actuator_ctrlrange[5]
        self.open_cmd = float(hi)     # see GRIPPER POLARITY note above
        self.closed_cmd = float(lo)

        self._gripper_bid = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
        )
        # A model clone whose wrist_roll range gets pinned to a chosen angle, so
        # the shared `solve_ik` (which clips to model.jnt_range) doubles as a
        # roll-constrained solver — no second IK implementation needed.
        self._roll_model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self._roll_lo, self._roll_hi = env.model.jnt_range[4]

        self._scratch = mujoco.MjData(env.model)
        self.reset()

    def reset(self) -> None:
        self._phase = 0
        self._hold = 0
        self._cache: tuple | None = None

    # --- forward kinematics helpers ----------------------------------------
    def _fk(self, qpos_arm: np.ndarray, roll: float | None = None):
        """Gripper body origin + rotation for an arm config (gripper held open)."""
        s = self._scratch
        s.qpos[:IK_N_ARM_JOINTS] = qpos_arm
        if roll is not None:
            s.qpos[4] = roll
        s.qpos[5] = self.open_cmd
        mujoco.mj_forward(self.env.model, s)
        return (
            s.xpos[self._gripper_bid].copy(),
            s.xmat[self._gripper_bid].reshape(3, 3).copy(),
        )

    def _grasp_point(self, qpos_arm: np.ndarray) -> np.ndarray:
        """World position of the point that actually holds the cube.
        ORACLE_GRASP_LOCAL is a gripper-body-LOCAL offset, so it rotates with the
        wrist and cannot be precomputed as a world vector."""
        origin, rot = self._fk(qpos_arm)
        return origin + rot @ ORACLE_GRASP_LOCAL

    def _finger_dir(self, qpos_arm: np.ndarray) -> np.ndarray:
        _, rot = self._fk(qpos_arm)
        return rot @ _FINGER_AXIS_LOCAL

    def _pinch_z(self, qpos_arm: np.ndarray, roll: float) -> float:
        """World z-component of the jaws' pinch axis. 0 => the jaws close
        horizontally (spanning the cube's 4 cm width); +-1 => vertically (trying
        to span its 6 cm height with a 4.2 cm aperture, which cannot work)."""
        _, rot = self._fk(qpos_arm, roll=roll)
        return float((rot @ _PINCH_AXIS_LOCAL)[2])

    def _solve_roll(self, qpos_arm: np.ndarray) -> float:
        """Pick the wrist_roll that makes the pinch axis horizontal.

        wrist_roll is the last joint before the gripper, so it spins the jaws
        about the finger axis: pinch_z(roll) is a smooth 1-D sinusoid, and this
        is just a 1-D root find. Solving it jointly with position inside one
        damped-least-squares loop was tried first and diverges into a local
        minimum (|pinch_z| stuck at 0.94 — still vertical); scanning the 1-D
        curve is both cheaper and robust.
        """
        rolls = np.linspace(
            max(self._roll_lo, -np.pi), min(self._roll_hi, np.pi), ORACLE_ROLL_SCAN
        )
        vals = np.abs([self._pinch_z(qpos_arm, r) for r in rolls])
        return float(rolls[int(vals.argmin())])

    def _solve_ik(self, target: np.ndarray) -> np.ndarray:
        """Drive the GRASP POINT (not the EE site) to `target`, with the wrist
        rolled so the jaws pinch horizontally.

        Two nested corrections, because `solve_ik` only knows how to drive the EE
        site's position:
          * outer — re-pick the roll for the arm config we ended up in, then
            re-solve with it pinned (the roll changes the pose, which changes the
            best roll);
          * inner — fixed-point on the site target: measure where the grasp point
            actually landed and shift the site target by the residual. Position-
            only IK leaves orientation free, so the site->grasp-point offset is
            rotated by whatever pose the solver lands on and cannot be subtracted
            once.
        """
        goal = np.asarray(target, dtype=np.float64)
        roll = np.pi / 2                      # a horizontal-pinch roll, as a seed
        q = np.zeros(IK_N_ARM_JOINTS)

        for _ in range(ORACLE_ROLL_ITERS):
            self._roll_model.jnt_range[4] = [roll, roll]   # pin wrist_roll
            site_target = goal.copy()
            seed = mujoco.MjData(self._roll_model)
            seed.qpos[:IK_N_ARM_JOINTS] = q
            seed.qpos[4] = roll
            q = solve_ik(self._roll_model, seed, EE_SITE, site_target)

            for _ in range(ORACLE_GRASP_CORRECTION_ITERS):
                site_target = site_target + (goal - self._grasp_point(q))
                seed = mujoco.MjData(self._roll_model)
                seed.qpos[:IK_N_ARM_JOINTS] = q
                q = solve_ik(self._roll_model, seed, EE_SITE, site_target)

            roll = self._solve_roll(q)
        return q

    # --- waypoints ----------------------------------------------------------
    def _waypoints(self) -> list[tuple[np.ndarray, float, int]]:
        """Recomputed from the LIVE cube pose, but cached while the cube is still
        — the roll scan is not free and re-solving every control step would
        dominate the episode. Re-solving when the cube actually moves is what lets
        the oracle recover if the approach nudges it."""
        cube = self.env.cube_pos.astype(np.float64)
        bin_ = self.env.bin_pos.astype(np.float64)

        if self._cache is not None:
            cached_cube, waypoints = self._cache
            if np.linalg.norm(cached_cube - cube) < 1e-3:
                return waypoints

        q_grasp = self._solve_ik(cube)
        finger = self._finger_dir(q_grasp)
        # Back off ALONG THE FINGER AXIS, not straight up: the fingers point
        # forward-and-down, so a vertical descent lands the jaw shaft on the cube.
        # Retreating along -finger keeps the shaft inside the volume the
        # fingertips already swept through.
        pre = cube - finger * ORACLE_STANDOFF

        lifted = cube + np.array([0.0, 0.0, ORACLE_LIFT_HEIGHT])
        # Dead-centre over the bin. Task 4 measured the clear-drop margin at only
        # +-0.035 from bin centre (releasing at +-0.04 perches the cube on the
        # rim), well inside the +-0.055 IK-reachable footprint — so this
        # deliberately does not chase the corners of that envelope.
        over_bin = bin_ + np.array([0.0, 0.0, ORACLE_RELEASE_HEIGHT])

        waypoints = [
            # 0 approach: high above the line-up point. Must clear the cube —
            #   going straight to the line-up pose sweeps the arm through it.
            (
                pre + np.array([0.0, 0.0, ORACLE_APPROACH_HEIGHT]),
                self.open_cmd,
                ORACLE_SETTLE_STEPS,
            ),
            (pre, self.open_cmd, ORACLE_SETTLE_STEPS),            # 1 line up behind the cube
            (cube, self.open_cmd, ORACLE_GRASP_SETTLE_STEPS),     # 2 advance along the finger axis
            (cube, self.closed_cmd, ORACLE_GRASP_SETTLE_STEPS),   # 3 close on the cube
            (lifted, self.closed_cmd, ORACLE_SETTLE_STEPS),       # 4 lift clear of the pedestal
            (over_bin, self.closed_cmd, ORACLE_SETTLE_STEPS),     # 5 traverse to the bin
            (over_bin, self.open_cmd, ORACLE_SETTLE_STEPS),       # 6 release
        ]
        self._cache = (cube, waypoints)
        return waypoints

    def act(self, obs: dict) -> np.ndarray:
        waypoints = self._waypoints()
        self._phase = min(self._phase, len(waypoints) - 1)
        target, gripper, settle_steps = waypoints[self._phase]

        # Freeze the waypoints once the jaws start closing: from here the cube
        # MOVES WITH the gripper, so re-solving against its live pose would make
        # the arm chase the cube it is already holding.
        if self._phase >= 3:
            self._cache = (self.env.cube_pos.astype(np.float64), waypoints)

        arm = self._solve_ik(target)

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
