"""M0 spike: how fast can MuJoCo render offscreen on this machine?

Spec risk 2. On macOS the only headless backend is CGL (`MUJOCO_GL=cgl`);
osmesa and egl are Linux-only. Upstream has an open discussion titled
"Offscreen rendering with mjr_render is extremely slow", so this is measured,
not assumed.

Renders `config.WRIST_CAM` (a named camera) rather than the default free
camera, since that is what the real data-collection/eval pipeline renders.

Run: make spike-render
"""

import json
import os
import platform
import time

import mujoco

from vla_trial.config import (
    IMG_H,
    IMG_W,
    RENDER_SPIKE_JSON,
    RENDER_SPIKE_N_FRAMES,
    RENDER_SPIKE_WARMUP_FRAMES,
    SCENE_XML,
    SPIKE_OUTPUT_DIR,
    WRIST_CAM,
)


def _default_backend() -> str:
    return "cgl" if platform.system() == "Darwin" else "egl"


def bench_render(n_frames: int = RENDER_SPIKE_N_FRAMES, width: int = IMG_W, height: int = IMG_H) -> dict:
    os.environ.setdefault("MUJOCO_GL", _default_backend())

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        # warm-up: discard frames until GL/driver state reaches steady state
        for _ in range(RENDER_SPIKE_WARMUP_FRAMES):
            renderer.update_scene(data, camera=WRIST_CAM)
            renderer.render()

        start = time.perf_counter()
        for _ in range(n_frames):
            renderer.update_scene(data, camera=WRIST_CAM)
            renderer.render()
        elapsed = time.perf_counter() - start

    result = {
        "fps": n_frames / elapsed,
        "n_frames": n_frames,
        "elapsed_s": elapsed,
        "width": width,
        "height": height,
        "camera": WRIST_CAM,
        "backend": os.environ["MUJOCO_GL"],
        "platform": platform.platform(),
    }

    SPIKE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_SPIKE_JSON.write_text(json.dumps(result, indent=2))
    return result
