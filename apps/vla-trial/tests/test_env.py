import mujoco
import numpy as np
import pytest

from vla_trial.config import EE_SITE, IMG_H, IMG_W, N_JOINTS, SUCCESS_SETTLE_STEPS, TASK
from vla_trial.env.ik import solve_ik
from vla_trial.env.so101_pick import SO101PickEnv

CAMERA_KEYS = ("observation.images.wrist", "observation.images.scene")


@pytest.fixture
def env():
    e = SO101PickEnv()
    yield e
    e.close()


def test_reset_returns_well_formed_obs(env):
    obs, info = env.reset(seed=0)
    assert obs["observation.images.wrist"].shape == (IMG_H, IMG_W, 3)
    assert obs["observation.images.wrist"].dtype == np.uint8
    assert obs["observation.images.scene"].shape == (IMG_H, IMG_W, 3)
    assert obs["observation.state"].shape == (N_JOINTS,)
    assert obs["observation.state"].dtype == np.float32
    assert info["task"] == TASK


def test_obs_contains_no_ground_truth(env):
    """The policy sees pixels + proprioception. Never the cube's pose."""
    obs, _ = env.reset(seed=0)
    assert set(obs) == {
        "observation.images.wrist",
        "observation.images.scene",
        "observation.state",
    }


def test_cube_qpos_does_not_overlap_policy_state(env):
    """Structural guarantee behind `test_obs_contains_no_ground_truth`:
    `observation.state` is `qpos[:N_JOINTS]`, so the cube's ground-truth pose
    stays out of it ONLY because the cube's freejoint qpos block starts at or
    after N_JOINTS. This is currently true because of MuJoCo's joint ordering
    in scene_pick.xml, not because of anything that inherently prevents a
    future scene edit from changing that ordering — `SO101PickEnv.__init__`
    now asserts this invariant at construction (so such an edit fails loudly),
    and this test pins it as a regression check independent of that assert.
    """
    assert env._cube_qadr >= N_JOINTS


def test_step_returns_five_tuple_and_respects_action_space(env):
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.action_space.shape == (N_JOINTS,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "is_success" in info


def test_seeding_is_deterministic(env):
    """Both cameras must be bit-identical across same-seed resets on one env
    instance. The scene camera is the one that actually had the reproducibility
    bug (mode="targetbody" re-ran MuJoCo's tracking-smoothing filter on every
    update_scene() call — see task-4-report.md); wrist_cam never exhibited it.
    Asserting only wrist_cam (as the original version of this test did) would
    not have caught that bug, and would not catch a regression of it either —
    e.g. someone reverting scene_pick.xml's camera to a dynamic mode, or
    removing the renderer warm-up in SO101PickEnv.__init__. Task 7 records
    datasets from the scene camera, so it is the one that most needs pinning.
    """
    obs_a, _ = env.reset(seed=42)
    cube_a = env.cube_pos.copy()
    obs_b, _ = env.reset(seed=42)
    assert np.allclose(cube_a, env.cube_pos)
    for key in CAMERA_KEYS:
        assert np.array_equal(obs_a[key], obs_b[key]), (
            f"{key} not bit-identical across same-seed resets on one env instance"
        )


def test_seeding_is_deterministic_across_env_instances():
    """Cross-instance determinism was also part of what the Task 4 fix
    addressed (the renderer's first-render-cycle warm-up in __init__, and the
    static-camera fix, both need to hold per-instance, not just within one
    instance's call history). Two separately-constructed envs, same seed, must
    produce bit-identical frames on both cameras."""
    env_a = SO101PickEnv()
    env_b = SO101PickEnv()
    try:
        obs_a, _ = env_a.reset(seed=42)
        obs_b, _ = env_b.reset(seed=42)
        assert np.allclose(env_a.cube_pos, env_b.cube_pos)
        for key in CAMERA_KEYS:
            assert np.array_equal(obs_a[key], obs_b[key]), (
                f"{key} not bit-identical across separately-constructed env instances"
            )
    finally:
        env_a.close()
        env_b.close()


def test_cube_spawns_inside_the_tight_region(env):
    """Spec §5: a wide spawn region at low episode counts is a documented failure."""
    from vla_trial.config import CUBE_SPAWN_CENTER, CUBE_SPAWN_HALF_EXTENT

    for seed in range(25):
        env.reset(seed=seed)
        offset = np.abs(env.cube_pos[:2] - np.array(CUBE_SPAWN_CENTER[:2]))
        assert np.all(offset <= CUBE_SPAWN_HALF_EXTENT + 1e-6), (
            f"seed {seed}: cube at {env.cube_pos} escaped the spawn region"
        )


def test_truncates_at_max_steps(env):
    from vla_trial.config import MAX_EPISODE_STEPS

    env.reset(seed=0)
    zero = np.zeros(N_JOINTS, dtype=np.float32)
    truncated = False
    for _ in range(MAX_EPISODE_STEPS + 1):
        _, _, terminated, truncated, _ = env.step(zero)
        if terminated or truncated:
            break
    assert truncated


# --- success predicate: settle/debounce (Finding 3) -------------------------
#
# These place the cube directly via env.data (qpos/qvel) rather than driving
# the arm, and call mj_forward (not mj_step) so the cube's state stays exactly
# where set between calls to env._is_success() — that isolates the debounce
# logic itself from actual drop physics, which the real oracle will exercise
# separately.


def _place_cube_in_bin_zone(env, speed: float):
    """Put the cube at the bin's XY, at rest height, moving at `speed` m/s
    straight up (an arbitrary direction — only the magnitude matters to the
    at-rest check)."""
    bx, by, _ = env.bin_pos
    env.data.qpos[env._cube_qadr : env._cube_qadr + 3] = [bx, by, 0.03]
    env.data.qvel[env._cube_vadr : env._cube_vadr + 3] = [0.0, 0.0, speed]
    mujoco.mj_forward(env.model, env.data)


def test_fast_moving_cube_through_the_zone_never_registers_success(env):
    """A cube merely passing through the success zone at speed — the bouncing-
    cube false positive Finding 3 warns about — must never register success,
    no matter how many consecutive steps it stays fast in the zone."""
    env.reset(seed=0)
    from vla_trial.config import SUCCESS_MAX_SPEED

    fast = SUCCESS_MAX_SPEED * 10
    for _ in range(SUCCESS_SETTLE_STEPS + 5):
        _place_cube_in_bin_zone(env, speed=fast)
        assert env._is_success() is False


def test_brief_slow_moment_without_sustained_settle_is_not_success(env):
    """A single slow instant (e.g. the apex of a bounce) must not be enough by
    itself; the debounce requires SUCCESS_SETTLE_STEPS consecutive at-rest
    steps, and speeding back up resets the streak to 0 — exactly what a
    bouncing cube does."""
    from vla_trial.config import SUCCESS_MAX_SPEED

    env.reset(seed=0)
    assert SUCCESS_SETTLE_STEPS > 1, "test assumes a multi-step debounce window"

    _place_cube_in_bin_zone(env, speed=SUCCESS_MAX_SPEED / 10)
    assert env._is_success() is False  # one slow step: streak=1, below threshold

    fast = SUCCESS_MAX_SPEED * 10
    for _ in range(SUCCESS_SETTLE_STEPS):
        _place_cube_in_bin_zone(env, speed=fast)
        assert env._is_success() is False  # streak reset, never reaccumulates


def test_settled_cube_registers_success_after_debounce_window(env):
    """Positive control: a cube that is genuinely in the zone and at rest for
    SUCCESS_SETTLE_STEPS consecutive steps DOES register success — the
    debounce rejects bounces without also rejecting real placements."""
    env.reset(seed=0)
    success = False
    for _ in range(SUCCESS_SETTLE_STEPS + 2):
        _place_cube_in_bin_zone(env, speed=0.0)
        success = env._is_success()
    assert success is True


# --- success predicate: real physics (Finding 1 / Finding 2) ----------------
#
# The tests above pin the debounce *counter logic* via mj_forward + hand-set
# qpos/qvel — they'd pass against any implementation with the same counter,
# calibrated thresholds or not, grasp-aware or not. The two tests below drive
# actual mujoco.mj_step physics (through the real env.step() gym API) so they
# also exercise whether SUCCESS_Z_MAX/SUCCESS_MAX_SPEED are physically
# calibrated, and whether the release (contact) check actually rejects a
# still-grasped cube — the exact gap Finding 1 found.


def test_dropped_cube_settles_and_registers_success_via_real_physics(env):
    """A cube released in free space above the bin, with no gripper contact,
    must fall under real gravity+contact, settle, and register success within
    a handful of control steps of first touching the floor — this is the test
    that guards the 10/10 oracle canary against a miscalibrated SUCCESS_Z_MAX
    or SUCCESS_MAX_SPEED (the debounce-only tests above can't catch that).

    Reference values from manual reproduction (see task-4-report.md, Fix pass
    2): a released cube settles to z=0.0399, resting speed reads 0.0 m/s, and
    success registers within ~5-12 control steps of first floor contact.
    """
    env.reset(seed=0)
    bin_floor_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "bin_floor")
    bx, by, _ = env.bin_pos

    # Teleport the cube to hover directly above the bin, at rest, with no
    # gripper geom anywhere near it — this isolates "does a genuinely
    # released cube register success correctly" from grasp mechanics.
    env.data.qpos[env._cube_qadr : env._cube_qadr + 3] = [bx, by, 0.15]
    env.data.qvel[env._cube_vadr : env._cube_vadr + 3] = [0.0, 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)
    hold_action = env.data.qpos[:N_JOINTS].copy()  # arm's current (home) pose

    contact_step = None
    success_step = None
    for i in range(60):
        _, _, terminated, _, info = env.step(hold_action)
        has_floor_contact = any(
            bin_floor_gid in (c.geom1, c.geom2) and env._cube_geom_id in (c.geom1, c.geom2)
            for c in (env.data.contact[j] for j in range(env.data.ncon))
        )
        if contact_step is None and has_floor_contact:
            contact_step = i
        if terminated:
            success_step = i
            assert info["is_success"] is True
            break

    assert success_step is not None, "cube never registered success after settling"
    assert contact_step is not None, "cube never touched bin_floor"
    assert 0 <= success_step - contact_step <= 12, (
        f"success at step {success_step}, {success_step - contact_step} steps "
        f"after first floor contact (step {contact_step}) — expected ~5-12"
    )
    final_z = float(env.cube_pos[2])
    assert final_z == pytest.approx(0.0399, abs=0.002), f"unexpected rest height {final_z}"
    final_speed = float(
        np.linalg.norm(env.data.qvel[env._cube_vadr : env._cube_vadr + 3])
    )
    assert final_speed < 0.001, f"cube not at rest at success time: speed={final_speed}"


def _grasp_cube_near_bin(env) -> np.ndarray:
    """Solve IK so the gripper's own jaw geoms sit closed around the cube
    *inside the bin's in-zone/low-z footprint*, then hold that pose. Returns
    the ctrl vector that holds the grasp (callers repeat this action).

    Real friction grasp, not a synthetic contact: the cube is placed at the
    geometric midpoint of the closed fixed/moving jaw box geoms for the
    solved arm pose, so `mujoco.mj_step`'s own collision pipeline is what
    puts cube-vs-gripper contacts in `data.contact` — nothing is injected.
    The IK target's xy offset from the bin (empirically ~(+0.06, +0.03) in
    site space) was found by probing so the resulting jaw midpoint lands
    within SUCCESS_XY_TOL of bin center and below SUCCESS_Z_MAX -- i.e.
    already inside the success zone without any further arm motion needed.
    """
    bx, by, _ = env.bin_pos
    target = np.array([bx + 0.06, by + 0.03, 0.02])
    q = solve_ik(env.model, env.data, EE_SITE, target)
    env.data.qpos[:5] = q
    env.data.qpos[5] = 1.2  # gripper mostly closed
    mujoco.mj_forward(env.model, env.data)

    fixed_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_box1")
    moving_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_box1")
    midpoint = (env.data.geom_xpos[fixed_gid] + env.data.geom_xpos[moving_gid]) / 2

    env.data.qpos[env._cube_qadr : env._cube_qadr + 3] = midpoint
    env.data.qvel[env._cube_vadr : env._cube_vadr + 3] = 0.0
    mujoco.mj_forward(env.model, env.data)

    hold_action = np.zeros(N_JOINTS, dtype=np.float32)
    hold_action[:5] = q
    hold_action[5] = 1.4  # keep closing pressure
    return hold_action


def test_grasped_cube_in_zone_never_registers_success_via_real_physics(env):
    """The exact Finding 1 bug, reproduced with real physics rather than
    hand-set qpos/qvel: a cube genuinely held by the (real, friction-based)
    gripper grasp, sitting within the bin's xy/z success zone and (per the
    printed trace) at-rest speed for long stretches, must NOT register
    success while the grasp holds contact.

    This test passes against the current (fixed) predicate because
    `_is_released()` sees the real cube-vs-gripper contact MuJoCo's collision
    pipeline reports every step. It was verified (not merely asserted) to
    fail against the pre-fix predicate: patching `env._is_released` to always
    return True (simulating the old in_xy-and-low-and-at_rest-only check)
    against this exact same physical trajectory registers success at control
    step 11 — well inside SUCCESS_SETTLE_STEPS's reach, since in_xy/low/
    at_rest all hold simultaneously for many consecutive steps here.
    """
    env.reset(seed=0)
    hold_action = _grasp_cube_near_bin(env)
    assert env._is_released() is False, "test setup didn't establish a real grasp contact"

    for i in range(40):
        _, _, terminated, _, info = env.step(hold_action)
        assert terminated is False, (
            f"success falsely registered at step {i} while cube is still "
            f"grasped (released={env._is_released()}, z={env.cube_pos[2]:.4f})"
        )
        assert info["is_success"] is False
