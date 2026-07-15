"""Roll a SmolVLA checkpoint out in the sim and score it.

Our own loop rather than `lerobot-eval`: SO101PickEnv is not a registered
LeRobot env type, and registering it buys nothing here.

The policy sees ONLY pixels + joint state + the task string. No ground truth.
That is the whole claim being tested.

Inference contract (verified against the installed lerobot==0.6.0, Task 9):
  - `SmolVLAPolicy.select_action(batch)` does NOT tokenize/normalize/batch a
    raw observation dict itself — that happens in a separate processor
    pipeline built by `make_pre_post_processors`, which must be run over the
    batch BEFORE `select_action` and over its output action AFTER (mirrors
    `spike/bench_policy.py`'s `_dummy_batch` + `lerobot/scripts/lerobot_eval.py`).
  - Loading the pipelines `from` the checkpoint path (`pretrained_path=`)
    pulls the checkpoint's own saved `policy_preprocessor.json` /
    `policy_postprocessor.json` rather than building fresh ones.

Camera renaming (empirically determined against the local Task 8 smoke
checkpoint, `outputs/train/smolvla_so101_smoke/checkpoints/000005/pretrained_model`):
  `policy_preprocessor.json`'s FIRST step is a `rename_observations_processor`
  with CAMERA_RENAME_MAP baked in from training. So the checkpoint's saved
  preprocessor ALREADY renames wrist->camera1 / scene->camera2 internally —
  we feed it the env's raw wrist/scene keys, not pre-renamed ones. Renaming
  ourselves first would just hand the pipeline's own rename step keys it no
  longer recognizes. `CAMERA_RENAME_MAP` is still passed explicitly via
  `preprocessor_overrides`, matching how lerobot's own `lerobot_eval.py`
  re-asserts `cfg.rename_map` even after loading pretrained processors — this
  keeps the rename sourced from `config.py`, not from trusting a JSON blob.
  The missing third camera (`camera3`) is padded automatically by
  `SmolVLAPolicy.prepare_images` from `policy.config.empty_cameras` — no
  batch key needed for it.
"""

import json
from pathlib import Path

import numpy as np
import torch

from vla_trial.config import (
    CAMERA_RENAME_MAP,
    EMPTY_CAMERAS,
    INFERENCE_DEVICE,
    MAX_EPISODE_STEPS,
    SMOKE_EVAL_OUTPUT_DIR,
    TASK,
)
from vla_trial.env.so101_pick import SO101PickEnv


def _to_batch(obs: dict, task: str) -> dict:
    """env obs (HWC uint8) -> raw policy batch (CHW float32 in [0,1]).

    Camera keys stay as the env emits them (wrist/scene), un-renamed and
    un-batched — the preprocessor pipeline's own rename step and
    `to_batch_processor` step handle both (see module docstring).
    """
    batch = {}
    for key in ("observation.images.wrist", "observation.images.scene"):
        img = torch.from_numpy(obs[key])
        batch[key] = img.permute(2, 0, 1).float() / 255.0
    batch["observation.state"] = torch.from_numpy(obs["observation.state"]).float()
    batch["task"] = task
    return batch


def evaluate(
    policy_path: str,
    n_episodes: int = 10,
    seed: int = 1000,
    task: str = TASK,
    output_dir: Path = SMOKE_EVAL_OUTPUT_DIR,
    device: str = INFERENCE_DEVICE,
    logger=None,
) -> dict:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.to(device)
    policy.eval()

    assert policy.config.empty_cameras == EMPTY_CAMERAS, (
        f"checkpoint's empty_cameras ({policy.config.empty_cameras}) != "
        f"config.EMPTY_CAMERAS ({EMPTY_CAMERAS}) — the checkpoint was trained "
        "with a different camera layout than this app now assumes."
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": CAMERA_RENAME_MAP},
        },
    )

    env = SO101PickEnv(task=task)
    successes = 0
    per_episode = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        policy.reset()
        success = False

        for step in range(MAX_EPISODE_STEPS):
            batch = preprocessor(_to_batch(obs, task))
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocessor(action)
            action = action.squeeze(0).cpu().numpy().astype(np.float32)

            if logger is not None:
                logger.log_step(step, obs, action, task=task)

            obs, _reward, terminated, truncated, info = env.step(action)
            success = bool(info["is_success"])
            if terminated or truncated:
                break

        successes += success
        per_episode.append({"seed": seed + i, "success": success})
        print(f"episode {i + 1}/{n_episodes}: {'OK' if success else 'FAIL'}")

    env.close()

    result = {
        "policy": str(policy_path),
        "device": device,
        "task": task,
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": successes / n_episodes,
        "episodes": per_episode,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_info.json").write_text(json.dumps(result, indent=2))
    print(f"success_rate: {result['success_rate']:.0%}")
    return result
