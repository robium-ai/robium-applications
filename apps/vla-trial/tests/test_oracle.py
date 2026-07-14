"""The day-1 canary: the oracle must be deterministic and near-perfect.

It sees the cube's ground-truth pose, so anything below 10/10 means the env,
the IK, or the grasp is broken — not that "the policy needs more data".
"""

import pytest

from vla_trial.env.so101_pick import SO101PickEnv
from vla_trial.oracle.scripted_pick import OraclePolicy, rollout


@pytest.fixture
def env():
    e = SO101PickEnv()
    yield e
    e.close()


def test_oracle_succeeds_on_one_seed(env):
    result = rollout(env, seed=0)
    assert result["success"], f"oracle failed in {result['n_steps']} steps"


def test_oracle_succeeds_10_of_10_seeded(env):
    successes = sum(rollout(env, seed=s)["success"] for s in range(10))
    assert successes == 10, f"oracle succeeded {successes}/10 — env or IK is broken"


def test_oracle_is_deterministic(env):
    a = rollout(env, seed=7)
    b = rollout(env, seed=7)
    assert a["n_steps"] == b["n_steps"]
    assert a["success"] == b["success"]


def test_rollout_emits_frames_shaped_for_the_dataset(env):
    result = rollout(env, seed=0)
    frame = result["frames"][0]
    assert set(frame) >= {
        "observation.images.wrist",
        "observation.images.scene",
        "observation.state",
        "action",
    }
