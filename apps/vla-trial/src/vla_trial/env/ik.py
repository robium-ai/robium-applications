"""Damped-least-squares (Levenberg-Marquardt) inverse kinematics for the SO-101 arm.

Pulled forward from Task 5 (the scripted oracle needs it to move the gripper to a
target position) because Task 4's Step 3a reachability probe needs it first: the
SO-101 is a small arm and any spawn/bin coordinate that is not actually reachable
will silently degrade into "the arm stops short" much later, deep inside the
oracle, unless it is checked here.

A pure function of (model, data, site_name, target_pos) — no SO101PickEnv
dependency, so it is usable standalone from a throwaway probe script.

IMPORTANT: this solves position only, for the 5 ARM joints (qpos[0:5]); the
gripper (qpos[5], the 6th DOF) is not part of the kinematic chain to the
`gripperframe` site's position and is left untouched. It does NOT raise on an
unreachable target — a target outside the arm's reach simply fails to converge
within `tol` and the best-effort (locally-optimal) joint config is returned. Callers
that need to know whether a target was actually reached must check the resulting
site position against the target themselves (see the Step 3a probe in the task
brief, or `tests/test_ik.py`).
"""

import mujoco
import numpy as np

N_ARM_JOINTS = 5  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll.


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    target_pos: np.ndarray,
    max_iters: int = 200,
    tol: float = 1e-4,
    damping: float = 1e-3,
    step_scale: float = 1.0,
) -> np.ndarray:
    """Solve for the arm qpos (5,) that drives `site_name` to `target_pos`.

    Iterates a scratch copy of `data` in place using the site Jacobian (damped
    least squares, a.k.a. Levenberg-Marquardt) so it never mutates the caller's
    `data`. Joint limits are respected at every step (clipped to `model.jnt_range`).
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"no site named {site_name!r} in the model")

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = 0.0

    jnt_lo = model.jnt_range[:N_ARM_JOINTS, 0]
    jnt_hi = model.jnt_range[:N_ARM_JOINTS, 1]

    jacp = np.zeros((3, model.nv))
    for _ in range(max_iters):
        mujoco.mj_forward(model, scratch)
        err = np.asarray(target_pos, dtype=np.float64) - scratch.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            break

        mujoco.mj_jacSite(model, scratch, jacp, None, site_id)
        j = jacp[:, :N_ARM_JOINTS]

        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 err.
        jjt = j @ j.T + (damping**2) * np.eye(3)
        dq = j.T @ np.linalg.solve(jjt, err)

        scratch.qpos[:N_ARM_JOINTS] = np.clip(
            scratch.qpos[:N_ARM_JOINTS] + step_scale * dq, jnt_lo, jnt_hi
        )

    return scratch.qpos[:N_ARM_JOINTS].copy()
