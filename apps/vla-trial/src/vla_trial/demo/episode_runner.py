"""One-episode runner for the demo page — the piece between the Gradio UI
and the proven env/oracle/policy code.

Loads the model ONCE (SmolVLA checkpoint + processors) so a Run click costs
an episode, not a model load — but constructs a FRESH SO101PickEnv inside
each run, in the thread that executes it. That is deliberate, not waste:
macOS MuJoCo GL contexts (CGL) are thread-affine, and Gradio executes each
request on whatever worker thread is free — a renderer created on the boot
thread DEADLOCKS (hangs forever in cgl `make_current`, no error) the first
time another thread renders with it. Verified 2026-07-15 with a faulthandler
stack dump; env construction is ~100 ms, invisible next to the episode.

Logs each step to a caller-provided `rr.RecordingStream` (NOT the global
recording — every Run gets its own recording id so the embedded viewer's
timeline is per-run) and yields after every control step so the caller can
stream incremental bytes to the browser.

The inference path deliberately mirrors policy/evaluate.py — same
preprocessor/postprocessor pipelines, same raw-key batch contract (the
checkpoint's own preprocessor renames wrist/scene internally; see that
module's docstring). It reuses evaluate's `_to_batch` rather than copying it.
"""

import itertools
import threading
from dataclasses import dataclass

import numpy as np
import rerun as rr
import torch

from vla_trial.config import (
    CAMERA_RENAME_MAP,
    DEMO_CHECKPOINT,
    DEMO_ORACLE_SEEDS,
    DEMO_TRAINED_SEED_BASE,
    EMPTY_CAMERAS,
    INFERENCE_DEVICE,
    MAX_EPISODE_STEPS,
    TASK,
)
from vla_trial.env.so101_pick import SO101PickEnv
from vla_trial.policy.evaluate import _to_batch

CONTROLLERS = ("oracle", "trained")


def _log_step(rec: rr.RecordingStream, step: int, obs: dict, action, task: str) -> None:
    """RerunLogger.log_step's schema, on an explicit RecordingStream.

    Same entity paths as viz/rerun_logger.py so demo recordings and the
    offline .rrd spot-checks read identically in the viewer.
    """
    rec.set_time("step", sequence=step)
    # .compress(): raw RGB at 2 cams x 256x256 x 300 steps is >100 MB down the
    # browser stream; JPEG at q75 is ~20x smaller and visually fine for a demo.
    rec.log("camera/wrist", rr.Image(obs["observation.images.wrist"]).compress(jpeg_quality=75))
    rec.log("camera/scene", rr.Image(obs["observation.images.scene"]).compress(jpeg_quality=75))
    rec.log("task", rr.TextLog(task))
    for i, q in enumerate(obs["observation.state"]):
        rec.log(f"state/joint_{i}", rr.Scalars([float(q)]))
    for i, a in enumerate(action):
        rec.log(f"action/joint_{i}", rr.Scalars([float(a)]))


@dataclass
class StepEvent:
    step: int
    total: int
    done: bool = False
    success: bool = False


class EpisodeRunner:
    """Owns the (single) env + policy; serializes runs with a lock.

    MuJoCo's Renderer and the policy's action queue are not safe for
    concurrent rollouts — `busy`/`acquire` make a second Run wait its turn
    (the UI surfaces a friendly error instead).
    """

    def __init__(self, device: str = INFERENCE_DEVICE, checkpoint: str = DEMO_CHECKPOINT):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.device = device
        self.checkpoint = checkpoint

        self.policy = SmolVLAPolicy.from_pretrained(checkpoint)
        self.policy.to(device)
        self.policy.eval()
        assert self.policy.config.empty_cameras == EMPTY_CAMERAS, (
            f"checkpoint empty_cameras ({self.policy.config.empty_cameras}) != "
            f"config.EMPTY_CAMERAS ({EMPTY_CAMERAS})"
        )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "rename_observations_processor": {"rename_map": CAMERA_RENAME_MAP},
            },
        )

        # Boot probe: prove env construction + a render works IN THIS PROCESS
        # (GL backend present, assets vendored), then close it — runs use
        # fresh per-thread envs (see module docstring for why).
        probe = SO101PickEnv(task=TASK)
        probe.reset(seed=0)
        probe.close()

        self._lock = threading.Lock()
        self._oracle_seeds = itertools.cycle(DEMO_ORACLE_SEEDS)
        self._trained_counter = itertools.count()

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def run(self, controller: str, instruction: str, rec: rr.RecordingStream):
        """Generator: one episode, yielding a StepEvent after each step."""
        if controller not in CONTROLLERS:
            raise ValueError(f"unknown controller {controller!r}")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("a run is already in progress")
        try:
            yield from self._run_locked(controller, instruction, rec)
        finally:
            self._lock.release()

    def _run_locked(self, controller: str, instruction: str, rec: rr.RecordingStream):
        from vla_trial.oracle.scripted_pick import OraclePolicy

        if controller == "oracle":
            seed = next(self._oracle_seeds)
        else:
            seed = DEMO_TRAINED_SEED_BASE + next(self._trained_counter)

        # Fresh env in THIS thread — the GL context must be created by the
        # thread that renders with it (module docstring).
        env = SO101PickEnv(task=instruction)
        try:
            obs, _ = env.reset(seed=seed)

            oracle = OraclePolicy(env) if controller == "oracle" else None
            if oracle is None:
                self.policy.reset()

            success = False
            step = 0
            for step in range(MAX_EPISODE_STEPS):
                if oracle is not None:
                    action = oracle.act(obs)
                else:
                    batch = self.preprocessor(_to_batch(obs, instruction))
                    with torch.inference_mode():
                        action = self.policy.select_action(batch)
                    action = self.postprocessor(action)
                    action = action.squeeze(0).cpu().numpy().astype(np.float32)

                _log_step(rec, step, obs, action, task=instruction)

                obs, _reward, terminated, truncated, info = env.step(action)
                success = bool(info["is_success"])
                yield StepEvent(step=step, total=MAX_EPISODE_STEPS)
                if terminated or truncated:
                    break

            yield StepEvent(step=step, total=MAX_EPISODE_STEPS, done=True, success=success)
        finally:
            env.close()
