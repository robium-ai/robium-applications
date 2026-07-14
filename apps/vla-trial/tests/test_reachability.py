"""Regression test for the Task 4 Step 3a workspace reachability probe.

That probe (originally run at the REPL and only recorded as a transcript in
task-4-report.md) found that menagerie's own default box spawn (0.5, 0, 0.03)
is unreachable for the SO-101's short reach, and that the bin's first
candidate position (0.35, 0.22, 0.0) failed at the far (+x,+y) corner at lift
height — which is why the bin ended up at (0.35, 0.16, 0.0). Nothing caught
that automatically; it was found by hand. This commits the same probe as a
test so a future scene change (e.g. a menagerie re-fetch, or hand-editing
scene_pick.xml) that silently breaks reachability again fails loudly here
instead.

`solve_ik` does NOT raise on an unreachable target — it returns its
locally-optimal best effort — so "reachable" must be checked via the residual
between the solved pose's site position and the target, not by absence of an
exception.
"""

import itertools

import mujoco
import numpy as np
import pytest

from vla_trial.config import (
    BIN_BODY,
    CUBE_SPAWN_CENTER,
    CUBE_SPAWN_HALF_EXTENT,
    EE_SITE,
    SCENE_XML,
)
from vla_trial.env.ik import solve_ik

# "Reachable" per the brief's Step 3a definition: IK residual under 1 cm.
REACHABLE_TOL = 0.01

# Task 4's Step 3a verified the bin's full footprint at these lift heights
# (0.03 = resting on the floor pad, up to 0.20 = well above the walls).
BIN_LIFT_HEIGHTS = (0.03, 0.10, 0.15, 0.20)

# Bin inner half-width (scene_pick.xml's bin_wall_* geoms; see config.py's
# SUCCESS_XY_TOL comment). Not itself a config constant — the wall geometry
# is scene-authored, not a run parameter — so it's local to this probe.
_BIN_HALF_WIDTH = 0.055


def _load_model_data():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _residual(model, data, target) -> float:
    """IK-solve toward `target` and return the resulting site-position error."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    q = solve_ik(model, data, EE_SITE, np.asarray(target, dtype=np.float64))

    scratch = mujoco.MjData(model)
    scratch.qpos[: len(q)] = q
    mujoco.mj_forward(model, scratch)
    return float(np.linalg.norm(np.asarray(target, dtype=np.float64) - scratch.site_xpos[site_id]))


def _cube_spawn_points():
    cx, cy, cz = CUBE_SPAWN_CENTER
    h = CUBE_SPAWN_HALF_EXTENT
    points = [(cx + dx, cy + dy, cz) for dx, dy in itertools.product((-h, h), (-h, h))]
    points.append((cx, cy, cz))  # center
    return points


_model, _data = _load_model_data()
_bin_bid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, BIN_BODY)
_BIN_X, _BIN_Y, _ = _data.xpos[_bin_bid]

_BIN_XY_POINTS = [(0.0, 0.0)] + [
    (dx, dy) for dx, dy in itertools.product((-_BIN_HALF_WIDTH, _BIN_HALF_WIDTH), repeat=2)
]


@pytest.mark.parametrize("target", _cube_spawn_points())
def test_cube_spawn_square_corner_is_reachable(target):
    """Every corner (and the center) of the cube-spawn square must be
    IK-reachable at cube height — a wide-but-unreachable spawn region would
    fail episodes for a reason that has nothing to do with the policy."""
    err = _residual(_model, _data, target)
    assert err < REACHABLE_TOL, f"spawn point {target} unreachable: err={err:.4f}"


@pytest.mark.parametrize(
    "offset,lift_z",
    list(itertools.product(_BIN_XY_POINTS, BIN_LIFT_HEIGHTS)),
)
def test_bin_footprint_is_reachable_at_lift_heights(offset, lift_z):
    """The bin's footprint (center + all four corners) must be IK-reachable at
    several lift heights, not just at table height — this is the exact check
    that found the (0.35, 0.22, 0.0) placement's far-corner failure at lift
    height and led to moving the bin to (0.35, 0.16, 0.0).

    IMPORTANT — reachable is NOT the same as "safe to release at": this test
    checks IK reach to the bin's full +-0.055 (inner-wall) footprint, but the
    Task 4 fix's physics-driven drop test found that releasing the cube at
    +-0.04 from bin center PERCHES it on the rim (final z=0.0899, not a
    success) instead of dropping it in. The measured clear-drop margin is
    +-0.035, not +-0.055 — the scripted oracle (next task) should aim its
    release point near bin center, well inside this reachability envelope,
    not out at these corners.
    """
    dx, dy = offset
    target = (_BIN_X + dx, _BIN_Y + dy, lift_z)
    err = _residual(_model, _data, target)
    assert err < REACHABLE_TOL, f"bin point {target} unreachable: err={err:.4f}"


def test_menagerie_default_box_spawn_is_unreachable():
    """Sanity check that this probe isn't vacuously green: menagerie's own
    default box position (0.5, 0, 0.03) — the vendored asset author's own
    choice — is genuinely outside the SO-101's reach (err ~0.035 in the
    original probe). If this ever starts passing, the arm's kinematics or the
    IK solver changed and the other assertions in this file deserve a second
    look."""
    err = _residual(_model, _data, (0.5, 0.0, 0.03))
    assert err >= REACHABLE_TOL, f"expected this point to stay unreachable, got err={err:.4f}"
