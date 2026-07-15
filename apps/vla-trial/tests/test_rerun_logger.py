import numpy as np

from vla_trial.config import IMG_H, IMG_W, N_JOINTS, TASK
from vla_trial.viz.rerun_logger import RerunLogger


def test_logger_writes_a_nonempty_recording(tmp_path):
    path = tmp_path / "test.rrd"
    logger = RerunLogger(app_id="test", save_path=path)

    obs = {
        "observation.images.wrist": np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8),
        "observation.images.scene": np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8),
        "observation.state": np.zeros(N_JOINTS, dtype=np.float32),
    }
    for step in range(3):
        logger.log_step(step, obs, np.zeros(N_JOINTS, dtype=np.float32), task=TASK)
    logger.close()

    assert path.is_file()
    assert path.stat().st_size > 0
