"""One-episode runner — the piece between the Gradio UI and the trained rungs.

Mirrors vla-trial's demo/episode_runner.py shape (lock-serialized runs, abort
event, fresh env per run) minus the MuJoCo GL thread-affinity machinery:
gym-pusht renders with pygame/pymunk on the CPU, so there is no GL context to
worry about. A fresh env per run is kept anyway — construction is milliseconds
and guarantees no state leaks between runs.

Policies load lazily per rung (ACT is ~200 MB on disk, seconds to load) and
stay cached; the gateway boots only the default rung so DEMO READY is fast.

The inference path is the same contract lerobot's own eval loop uses
(verified against lerobot 0.6.0's scripts/lerobot_eval.py rollout()):
preprocess_observation -> preprocessor pipeline -> policy.select_action ->
postprocessor pipeline. Single sync env — immune to the forkserver/async-env
gotcha by construction.
"""

import itertools
import json
import threading
from dataclasses import dataclass

import gymnasium as gym
import gym_pusht  # noqa: F401 — registers gym_pusht/PushT-v0
import numpy as np
import rerun as rr
import torch
from lerobot.envs.utils import preprocess_observation

from manip_trial import config

MAX_EPISODE_STEPS = 300  # gym_pusht registration default; lerobot's PushtEnv config agrees


def _log_step(rec: rr.RecordingStream, step: int, obs: dict, action, reward: float, max_reward: float) -> None:
    rec.set_time("step", sequence=step)
    # obs["pixels"] is the 96x96 frame the policy actually sees — honest by
    # construction. JPEG q85: tiny over the browser stream, fine to look at.
    rec.log("sim", rr.Image(obs["pixels"]).compress(jpeg_quality=85))
    rec.log("reward/coverage", rr.Scalars([float(reward)]))
    rec.log("reward/max_so_far", rr.Scalars([float(max_reward)]))
    rec.log("action/x", rr.Scalars([float(action[0])]))
    rec.log("action/y", rr.Scalars([float(action[1])]))


@dataclass
class StepEvent:
    step: int
    total: int
    done: bool = False
    success: bool = False
    aborted: bool = False
    max_reward: float = 0.0


class EpisodeRunner:
    """Owns the rung policies; serializes runs with a lock."""

    def __init__(self, device: str | None = None):
        self.device = device or config.demo_device()
        self.manifest = json.loads(config.DEMO_LADDER_MANIFEST.read_text())
        self.rungs = {r["name"]: r for r in self.manifest["rungs"]}
        self._policies: dict[str, tuple] = {}
        self._lock = threading.Lock()
        self._abort = threading.Event()
        self._seed_counter = itertools.count()

        self._load(config.DEMO_DEFAULT_RUNG)  # boot cost: one rung, not four
        # Boot probe: prove the env constructs + renders in this process.
        probe = gym.make(
            "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"
        )
        probe.reset(seed=0)
        probe.close()

    def _load(self, rung: str) -> tuple:
        if rung not in self._policies:
            from lerobot.policies.act.modeling_act import ACTPolicy
            from lerobot.policies.factory import make_pre_post_processors

            path = str(config.APP_ROOT / self.rungs[rung]["checkpoint"])
            policy = ACTPolicy.from_pretrained(path)
            policy.to(self.device)
            policy.eval()
            pre, post = make_pre_post_processors(
                policy_cfg=policy.config,
                pretrained_path=path,
                preprocessor_overrides={"device_processor": {"device": self.device}},
            )
            self._policies[rung] = (policy, pre, post)
        return self._policies[rung]

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def request_abort(self) -> None:
        """Stop the in-flight episode at its next control step (page-refresh path)."""
        self._abort.set()

    def run(self, rung: str, rec: rr.RecordingStream):
        """Generator: one episode, yielding a StepEvent after each step."""
        if rung not in self.rungs:
            raise ValueError(f"unknown rung {rung!r}")
        # Wait, don't fail: an aborted predecessor exits within one control step.
        if not self._lock.acquire(timeout=30):
            raise RuntimeError("a run is already in progress")
        self._abort.clear()
        try:
            yield from self._run_locked(rung, rec)
        finally:
            self._lock.release()

    def _run_locked(self, rung: str, rec: rr.RecordingStream):
        policy, pre, post = self._load(rung)
        policy.reset()
        # Fresh seed per run so repeat runs show different starts; offset from
        # the eval SEED so the demo never replays the gallery's exact episodes.
        seed = config.SEED + 10_000 + next(self._seed_counter)

        env = gym.make(
            "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"
        )
        try:
            obs, _ = env.reset(seed=seed)
            success = False
            max_reward = 0.0
            step = 0
            for step in range(MAX_EPISODE_STEPS):
                if self._abort.is_set():
                    yield StepEvent(step=step, total=MAX_EPISODE_STEPS, done=True,
                                    aborted=True, max_reward=max_reward)
                    return
                batch = pre(preprocess_observation(obs))
                with torch.inference_mode():
                    action = policy.select_action(batch)
                action = post(action)
                action = action.squeeze(0).cpu().numpy().astype(np.float32)

                obs, reward, terminated, truncated, info = env.step(action)
                max_reward = max(max_reward, float(reward))
                success = bool(info["is_success"])
                _log_step(rec, step, obs, action, float(reward), max_reward)
                yield StepEvent(step=step, total=MAX_EPISODE_STEPS, max_reward=max_reward)
                if terminated or truncated:
                    break

            yield StepEvent(step=step, total=MAX_EPISODE_STEPS, done=True,
                            success=success, max_reward=max_reward)
        finally:
            env.close()
