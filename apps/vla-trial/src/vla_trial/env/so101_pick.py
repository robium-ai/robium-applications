"""Language-conditioned SO-101 pick-and-place in MuJoCo.

Hand-built because no SO-101 Gymnasium env exists in any maintained package,
and the one ready-made language-conditioned MuJoCo benchmark (LIBERO) is
Linux-only. Structurally modelled on gym-hil, but deliberately NOT importing it:
gym-hil pins mujoco<3.9 and would downgrade us below the menagerie SO-101 floor
of >=3.1.3.

The policy observes pixels + joint state. The cube's pose is ground truth and is
exposed ONLY via `cube_pos` for the scripted oracle — it never enters an obs.
"""

import os
import platform

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from vla_trial.config import (
    BIN_BODY,
    CONTROL_FPS,
    CUBE_BODY,
    CUBE_SPAWN_CENTER,
    CUBE_SPAWN_HALF_EXTENT,
    IMG_H,
    IMG_W,
    MAX_EPISODE_STEPS,
    N_JOINTS,
    SCENE_CAM,
    SCENE_XML,
    SUCCESS_MAX_SPEED,
    SUCCESS_SETTLE_STEPS,
    SUCCESS_XY_TOL,
    SUCCESS_Z_MAX,
    TASK,
    WRIST_CAM,
)

# macOS headless rendering is CGL-only; osmesa/egl are Linux-only.
os.environ.setdefault("MUJOCO_GL", "cgl" if platform.system() == "Darwin" else "egl")


class SO101PickEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_FPS}

    def __init__(self, task: str = TASK):
        self.task = task
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)

        self._renderer = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
        # A freshly-constructed Renderer's very FIRST (mj_resetData -> mj_forward
        # -> render) cycle on macOS CGL is not repeatable — observed up to 6 LSB
        # drift over ~12% of pixels vs. a later render of the IDENTICAL physics
        # state. Every render from the *second* such cycle onward is bit-exact
        # for identical state (confirmed over 8 consecutive identical-state
        # renders: only index 0->1 differed; 1->2, 2->3, ... were all zero-diff).
        # So the warm-up must reproduce the real reset()/render sequence (not a
        # render of the data's raw pre-mj_forward zeros) to actually consume
        # that first cold cycle — otherwise the first *real* reset() is still
        # the unsettled one. Cheap: happens once per env, not per episode.
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        for cam in (WRIST_CAM, SCENE_CAM):
            self._renderer.update_scene(self.data, camera=cam)
            self._renderer.render()
        # NB: the pick target's BODY name is "box" (menagerie's), even though the
        # task string calls it "the cube". The bin is ours, added in scene_pick.xml.
        self._cube_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY
        )
        self._bin_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BIN_BODY)
        if self._cube_bid < 0 or self._bin_bid < 0:
            raise RuntimeError(
                f"scene is missing {CUBE_BODY!r} or {BIN_BODY!r} — did you add the bin "
                "to scene_pick.xml?"
            )
        # The cube's freejoint qpos address (7 dof: 3 pos + 4 quat).
        self._cube_qadr = self.model.jnt_qposadr[self.model.body_jntadr[self._cube_bid]]
        # The cube's freejoint qvel address (6 dof: 3 linear + 3 angular), used
        # by the success predicate's at-rest check.
        self._cube_vadr = self.model.jnt_dofadr[self.model.body_jntadr[self._cube_bid]]

        # Structural guarantee behind `_obs()`'s `qpos[:N_JOINTS]` slice (see
        # docstring above): this is what actually keeps the cube's ground-truth
        # pose out of the policy's observation. It holds today only because the
        # cube's freejoint happens to be ordered after the 6 actuated joints in
        # the MJCF. A future scene edit that inserts a body/joint before the cube
        # would silently shift this and leak ground truth into `observation.state`
        # — assert it here so that edit fails loudly at construction instead.
        assert self._cube_qadr >= N_JOINTS, (
            f"cube qpos address ({self._cube_qadr}) is before N_JOINTS ({N_JOINTS}) "
            "— qpos[:N_JOINTS] would leak the cube's ground-truth pose into "
            "observation.state. A body/joint was inserted before the cube in the "
            "scene (or N_JOINTS grew to include it); fix the scene ordering or "
            "revisit this invariant deliberately."
        )

        self._steps = 0
        self._settle_steps = 0

        # Actuator ranges are the ground truth for the action space.
        lo = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        hi = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(low=lo, high=hi, shape=(N_JOINTS,), dtype=np.float32)

        self.observation_space = spaces.Dict(
            {
                "observation.images.wrist": spaces.Box(
                    0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8
                ),
                "observation.images.scene": spaces.Box(
                    0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8
                ),
                "observation.state": spaces.Box(
                    -np.inf, np.inf, (N_JOINTS,), dtype=np.float32
                ),
            }
        )

    # --- ground truth: oracle only, never observed --------------------------
    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self._cube_bid].astype(np.float32)

    @property
    def box_pos(self) -> np.ndarray:
        """Alias for `cube_pos` — the body's own name is `box` (see CUBE_BODY)."""
        return self.cube_pos

    @property
    def bin_pos(self) -> np.ndarray:
        return self.data.xpos[self._bin_bid].astype(np.float32)

    # --- gym api ------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        cx, cy, cz = CUBE_SPAWN_CENTER
        dx, dy = self.np_random.uniform(
            -CUBE_SPAWN_HALF_EXTENT, CUBE_SPAWN_HALF_EXTENT, size=2
        )
        self.data.qpos[self._cube_qadr : self._cube_qadr + 3] = [cx + dx, cy + dy, cz]

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._settle_steps = 0
        return self._obs(), {"task": self.task, "is_success": False}

    def step(self, action: np.ndarray):
        self.data.ctrl[:] = np.clip(
            action, self.action_space.low, self.action_space.high
        )
        n_substeps = max(1, int(round(1.0 / (CONTROL_FPS * self.model.opt.timestep))))
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._steps += 1
        success = self._is_success()
        truncated = self._steps >= MAX_EPISODE_STEPS and not success
        return (
            self._obs(),
            float(success),
            bool(success),
            bool(truncated),
            {"task": self.task, "is_success": bool(success)},
        )

    def close(self):
        self._renderer.close()

    # --- internals ----------------------------------------------------------
    def _obs(self) -> dict:
        return {
            "observation.images.wrist": self._render(WRIST_CAM),
            "observation.images.scene": self._render(SCENE_CAM),
            "observation.state": self.data.qpos[:N_JOINTS].astype(np.float32),
        }

    def _render(self, camera: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def _in_zone_and_at_rest(self) -> bool:
        """XY+Z zone check AND a genuine at-rest check (linear speed below
        threshold) for the CURRENT physics step only. Does not by itself imply
        success — see `_is_success`'s debounce."""
        cube, bin_ = self.cube_pos, self.bin_pos
        in_xy = np.linalg.norm(cube[:2] - bin_[:2]) < SUCCESS_XY_TOL
        low = cube[2] < SUCCESS_Z_MAX
        speed = float(np.linalg.norm(self.data.qvel[self._cube_vadr : self._cube_vadr + 3]))
        at_rest = speed < SUCCESS_MAX_SPEED
        return bool(in_xy and low and at_rest)

    def _is_success(self) -> bool:
        # The bin's walls are shallow (wall height ~= cube half-height), so a
        # cube can bounce THROUGH the XY+Z zone on its way back out. Requiring
        # the in-zone-AND-at-rest condition to hold for SUCCESS_SETTLE_STEPS
        # consecutive control steps (not just once) is what rejects that: a
        # bouncing cube's speed only dips below SUCCESS_MAX_SPEED momentarily
        # (near the top of a bounce) and the streak resets to 0 the instant it
        # speeds back up, so it can never accumulate a full debounce window. A
        # cube actually placed/settled in the bin stays slow across consecutive
        # steps and does accumulate one.
        if self._in_zone_and_at_rest():
            self._settle_steps += 1
        else:
            self._settle_steps = 0
        return self._settle_steps >= SUCCESS_SETTLE_STEPS
