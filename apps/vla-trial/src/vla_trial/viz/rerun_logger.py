"""Rerun logging — one place that knows the archetype API.

Why Rerun and not an MJPEG stream: a video rectangle shows what happened but
discards the joint states, the language instruction, and the action chunk the
policy predicted. Rerun puts all of them on one scrubable timeline, which IS the
explanation of what the VLA did — the whole point of the demo.

Archetype check against the pinned rerun-sdk==0.34.1 (Task 6, Step 1): this
release exposes `rr.Scalars` (plural) — NOT the older singular `rr.Scalar` —
taking a `Float64ArrayLike`, so a single scalar is logged as `rr.Scalars([x])`.

A second, unwarned-about drift: the brief's `rr.set_time_sequence("step", step)`
does not exist in 0.34.1 either — it was replaced by the unified
`rr.set_time("step", sequence=step)`.
"""

from pathlib import Path

import numpy as np
import rerun as rr


class RerunLogger:
    def __init__(
        self,
        app_id: str = "vla_trial",
        spawn: bool = False,
        save_path: Path | None = None,
    ):
        rr.init(app_id, spawn=spawn)
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(str(save_path))
        self._save_path = save_path

    def log_step(
        self,
        step: int,
        obs: dict,
        action: np.ndarray,
        task: str,
        chunk: np.ndarray | None = None,
    ) -> None:
        rr.set_time("step", sequence=step)

        # What the policy actually sees.
        rr.log("camera/wrist", rr.Image(obs["observation.images.wrist"]))
        rr.log("camera/scene", rr.Image(obs["observation.images.scene"]))

        # What it was told.
        rr.log("task", rr.TextLog(task))

        # Where it is, and what it decided.
        for i, q in enumerate(obs["observation.state"]):
            rr.log(f"state/joint_{i}", rr.Scalars([float(q)]))
        for i, a in enumerate(action):
            rr.log(f"action/joint_{i}", rr.Scalars([float(a)]))

        # The predicted chunk, if the caller has one: the VLA's lookahead.
        if chunk is not None:
            for i in range(chunk.shape[-1]):
                rr.log(f"chunk/joint_{i}", rr.Scalars(chunk[:, i].tolist()))

    def close(self) -> None:
        rr.disconnect()
