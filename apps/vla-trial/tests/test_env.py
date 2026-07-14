import numpy as np
import pytest

from vla_trial.config import IMG_H, IMG_W, N_JOINTS, TASK
from vla_trial.env.so101_pick import SO101PickEnv


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
    obs_a, _ = env.reset(seed=42)
    cube_a = env.cube_pos.copy()
    obs_b, _ = env.reset(seed=42)
    assert np.allclose(cube_a, env.cube_pos)
    assert np.array_equal(
        obs_a["observation.images.wrist"], obs_b["observation.images.wrist"]
    )


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
