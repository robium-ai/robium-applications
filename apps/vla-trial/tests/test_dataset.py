"""The recorded dataset must be loadable by LeRobot exactly as training will load it."""

import pytest

from vla_trial.config import CONTROL_FPS, N_JOINTS, TASK


@pytest.mark.slow
def test_recorded_dataset_loads_and_is_well_formed(tmp_path, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from vla_trial.data.record import record

    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    root = record(n_episodes=2, repo_id="test/so101_pick_smoke", push=False)

    ds = LeRobotDataset("test/so101_pick_smoke", root=root)
    assert ds.num_episodes == 2
    assert ds.fps == CONTROL_FPS

    frame = ds[0]
    assert frame["observation.state"].shape == (N_JOINTS,)
    assert frame["action"].shape == (N_JOINTS,)
    assert "observation.images.wrist" in frame
    # The language condition is what makes this a VLA dataset at all.
    assert ds.meta.tasks.iloc[0].name == TASK or TASK in str(ds.meta.tasks)
