"""Single source of truth for vla-trial run parameters.

The Makefile targets and the tests both build their invocations from these
values, so a hand-run stage and the pass-bar run can never drift apart.
(Pattern lifted from apps/manip-trial.)
"""

import os
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[2]
# Vendored menagerie dump (~18MB of binary STL meshes) — gitignored except
# for scene_pick.xml (see apps/vla-trial/.gitignore), repopulated by
# `make assets`. Never edited in place.
ASSETS = Path(__file__).resolve().parent / "env" / "assets"

# menagerie's scene_box.xml (arm + box) ships only a wrist-mounted camera and
# no overview camera. scene_pick.xml is our own file, tracked in git (not
# vendored — see fetch_assets.sh), that <include>s scene_box.xml unmodified
# and adds the missing "scene" camera, so `make assets` stays idempotent. It
# must live alongside scene_box.xml — MuJoCo resolves mesh/include paths
# relative to the including file's own directory.
SCENE_XML = ASSETS / "scene_pick.xml"

# The vendored MJCF's actual camera names (enumerated from the loaded model;
# menagerie does not name an overview camera "scene", only the wrist one).
WRIST_CAM = "wrist_cam"
SCENE_CAM = "scene"

# --- rendering -------------------------------------------------------------
# Policy observation size. 256x256 is SmolVLA's expected image scale.
IMG_W = 256
IMG_H = 256

SPIKE_OUTPUT_DIR = APP_ROOT / "outputs" / "spike"
RENDER_SPIKE_JSON = SPIKE_OUTPUT_DIR / "render.json"
POLICY_SPIKE_JSON = SPIKE_OUTPUT_DIR / "policy.json"

# Task 6: where RerunLogger recordings (.rrd) land.
VIZ_DIR = APP_ROOT / "outputs" / "viz"

# M0 gate: the render rate the sim must clear for a laptop demo to be viable.
# One control step renders 2 cameras; at 30 FPS sim that is 60 renders/sec.
RENDER_FPS_FLOOR = 60.0

# Spike: number of frames to render in the benchmark run.
RENDER_SPIKE_N_FRAMES = 1000

# Spike: number of warm-up frames to discard before timing starts.
# GL/driver state takes ~4 renders to reach steady state; set to 5 for margin.
RENDER_SPIKE_WARMUP_FRAMES = 5

# M0 gate: SmolVLA action chunking means one forward pass covers ~50 actions
# (~1.7 s of robot motion at 30 FPS). So the bar is ~1 pass/sec, not 30 Hz.
POLICY_LATENCY_CEILING_S = 1.0

# Spike: number of timed passes in the policy-latency benchmark run.
POLICY_SPIKE_N_PASSES = 20

# Spike: number of warm-up passes to discard before timing starts. The first
# timed pass pays lazy-init/kernel-compile costs far beyond the rest — 2
# warm-up passes, as sketched in the brief, left no headroom; 5 does.
POLICY_SPIKE_WARMUP_PASSES = 5

# --- policy ------------------------------------------------------------
# 450M. The only VLA in the "runs without a GPU" class. Its pretraining corpus
# is SO-100 data exclusively, so an SO-101 fine-tune is in-embodiment.
BASE_POLICY_ID = "lerobot/smolvla_base"

# SO-101: 5 arm joints + 1 gripper.
N_JOINTS = 6

TASK = "put the green cube in the bin"

# Ground truth about the vendored scene, established in Task 1 (do not guess):
#   bodies: world, base, shoulder, upper_arm, lower_arm, wrist, gripper,
#           camera_mount, moving_jaw_so101_v1, box
#   sites:  baseframe, gripperframe          (there is NO fingertip site)
#   cams:   wrist_cam                        (the "scene" cam is OURS, added in
#                                             scene_pick.xml — menagerie has none)
#   actuators: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
#              gripper
# The menagerie scene ships ONE manipulable object, body+geom both named "box"
# (green, half-extents 0.02 0.02 0.03, at 0.5 0 0.03). There is NO container —
# Task 4 adds one to scene_pick.xml. So the pick target's body name is "box"
# even though we call it "the cube" in the task string.
CUBE_BODY = "box"
BIN_BODY = "bin"
EE_SITE = "gripperframe"

# --- env -------------------------------------------------------------------
CONTROL_FPS = 30  # SmolVLA's pretraining corpus is standardized to 30 FPS.
MAX_EPISODE_STEPS = 300  # 10 s at 30 FPS.

# THE constraint. Spec §5: 50 episodes over a 30 cm workspace is a documented
# outright failure ("learned the general motion but couldn't pin down grasp
# locations"); 75 episodes over ~10 cm gives 60-80%. Density beats count.
# Half-extent 0.05 m => a 10 cm x 10 cm spawn square.
#
# The CENTRE below is a starting guess and MUST be validated by Step 3a's
# reachability check — every corner of the spawn square has to be IK-reachable,
# or episodes will fail for a reason that has nothing to do with the policy.
# The cube rests on the PEDESTAL (scene_pick.xml), not the floor: z is the
# pedestal's top surface (PEDESTAL_HEIGHT) plus the cube's half-height (0.03).
# The pedestal is what makes the pick physically possible at all — see Task 5
# and scene_pick.xml's comment. Half-extent stays 0.05 (a 10cm spawn square).
PEDESTAL_HEIGHT = 0.06
CUBE_SPAWN_CENTER = (0.32, -0.06, PEDESTAL_HEIGHT + 0.03)
CUBE_SPAWN_HALF_EXTENT = 0.05

# Success = the cube is inside the bin's footprint and low enough to be resting
# in it rather than being carried above it. Bin inner half-width is 0.055.
SUCCESS_XY_TOL = 0.05
SUCCESS_Z_MAX = 0.06

# The bin walls are shallow (wall height ~= cube half-height), so a cube can
# bounce THROUGH the XY+Z success zone on its way back out. A single-step
# position check alone would register that bounce as success. So success also
# requires the cube to be genuinely at rest: linear speed below
# SUCCESS_MAX_SPEED, AND the in-zone-and-slow condition held for
# SUCCESS_SETTLE_STEPS consecutive control steps (debounce). A cube merely
# passing through the zone can't satisfy both at once for that many steps in a
# row; a cube actually placed/settled in the bin can.
SUCCESS_SETTLE_STEPS = 5
SUCCESS_MAX_SPEED = 0.05  # m/s, cube linear speed threshold to count as "at rest".

# Position and speed alone can't tell "held" from "let go": a cube lowered
# slowly enough (descent speed < SUCCESS_MAX_SPEED) drifts through the
# in-zone/at-rest window while STILL fully grasped -- traversing
# SUCCESS_Z_MAX down to resting height at ~0.03 m/s takes ~19 control steps,
# far more than SUCCESS_SETTLE_STEPS needs to accumulate. Contact against the
# gripper's own collision geoms is the only signal that actually
# distinguishes "held" from "released", so success also requires NO contact
# between the cube geom and any of these. (A bin_floor-contact check was
# considered instead but rejected: a cube can be set down flush on the floor
# while the jaws are still closed around it, which would pass a
# floor-contact check while still being grasped.)
# Gripper geoms are resolved by BODY MEMBERSHIP, not by a name list.
#
# This used to be a hand-maintained tuple of geom NAMES. That was a latent
# false-success bug, found in Task 5: the geom that actually collides with the
# cube -- the fixed jaw's collision mesh
# (`wrist_roll_follower_so101_gripper_part0_v1`, which spans gripper-local
# z -0.106..-0.010) -- has NO `name` attribute in the vendored MJCF, so a
# name-based list could never see it. `_is_released()` would therefore have
# reported "released" for a cube resting against that mesh while it was still
# very much in the gripper's grasp, registering a FALSE SUCCESS the moment the
# grasp started working. Names are optional in MJCF and menagerie leaves most
# mesh geoms unnamed, so any name list is structurally incomplete.
#
# Body membership has none of that failure mode: every collision geom of the
# gripper belongs to one of these two bodies by construction, named or not.
# `SO101PickEnv` expands these to geom ids via `model.geom_bodyid`.
GRIPPER_BODIES = ("gripper", "moving_jaw_so101_v1")
# The cube's GEOM name (as opposed to CUBE_BODY, its body name) -- MuJoCo
# keeps geom and body names in separate namespaces, so both happen to be "box".
CUBE_GEOM = "box"

# --- ik ----------------------------------------------------------------
# Damped-least-squares IK hyperparameters (env/ik.py's solve_ik). Project
# rule: run parameters live in config.py, not hardcoded in the solver — the
# scripted oracle (next task) may need to retune these.
IK_N_ARM_JOINTS = 5  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll.
IK_MAX_ITERS = 200
IK_TOL = 1e-4
IK_DAMPING = 1e-3
IK_STEP_SCALE = 1.0

# --- oracle --------------------------------------------------------------
# THE GRASP POINT — where the jaws actually hold the cube, in GRIPPER-BODY-LOCAL
# coordinates. Empirically calibrated, and NOT where you would guess.
#
# The `gripperframe` site (EE_SITE) is NOT the grasp point. Aiming IK at it puts
# the real jaws several cm from the cube. Calibrated by brute force instead:
# park the arm, teleport the cube to a grid of candidate gripper-local offsets,
# close the gripper, lift, and keep the offsets whose cube came up with the arm.
# The true-pinch holds (jaws settling at ~0.32 rad, i.e. exactly the cube-width
# jaw angle, rather than wedging) clustered tightly at local x +0.010..+0.025,
# z -0.100..-0.115. This is the pinch point BETWEEN the fingertips — which only
# becomes reachable once the wrist is rolled so the jaws close horizontally (see
# scripted_pick.py's _solve_roll); before that fix nothing grasped anywhere.
#
# This value is SENSITIVE — it was swept end-to-end through the real oracle:
#   (0.015, -0.102) -> 9/10     (0.015, -0.108) -> 8/10
#   (0.018, -0.108) -> 8/10     (0.015, -0.114) -> 6/10
#   (0.012, -0.108) -> 3/10
# Do not "tidy" it without re-running the sweep.
ORACLE_GRASP_LOCAL = np.array([0.015, 0.0, -0.102])

# Fixed-point passes that drive the GRASP POINT (not the EE site) onto a target.
# Position-only IK leaves the wrist orientation free, so the site->grasp-point
# offset is rotated by whatever pose the solver lands on and cannot be
# subtracted once — it has to be corrected iteratively.
ORACLE_GRASP_CORRECTION_ITERS = 4

# Outer passes that re-pick wrist_roll for the arm config the solver landed in
# (roll changes the pose, which changes the best roll), and the resolution of the
# 1-D scan over roll used to zero the pinch axis' vertical component.
ORACLE_ROLL_ITERS = 4
ORACLE_ROLL_SCAN = 361

# How far to back the pre-grasp off ALONG THE FINGER AXIS (not straight up). The
# fingers point below horizontal, so a purely vertical descent lands the fixed
# jaw's shaft on top of the cube; retreating along -finger_dir keeps that shaft
# inside the volume the fingertips already swept through. Swept: 0.09 -> 8/10,
# 0.11 -> 7/10.
ORACLE_STANDOFF = 0.09

# APPROACH_HEIGHT is how far ABOVE the line-up point the arm first travels. It
# has to actually clear the cube: the position servo takes an arbitrary
# joint-space path from home, and if this first waypoint is too low the arm
# sweeps THROUGH the cube on its way there and knocks it off the pedestal.
#
# This was the last failing seed. At 0.16 the near-base corner of the spawn
# square (seed 2, cube at x=0.296 — closest to the arm) got clipped during the
# very first move: the cube shifted at control step 5, ended up 0.18 m away on
# the floor, and the episode was a guaranteed loss before the gripper had done
# anything. 0.20 clears it. Measured on that seed: 0.16 -> FAIL, 0.20/0.24/0.28
# -> PASS. Kept at 0.20 (the lowest that works) so the arm is not wasting
# episode budget flying higher than it needs to.
ORACLE_APPROACH_HEIGHT = 0.20

# Lift clear of the pedestal before traversing to the bin.
ORACLE_LIFT_HEIGHT = 0.12

# Release height above the bin. Task 4's real-physics drop test validated that a
# free cube released from directly above bin centre at this height settles into
# the bin and registers success 5-12 control steps after floor contact.
ORACLE_RELEASE_HEIGHT = 0.15

# Control steps to hold at each waypoint before advancing, so the position
# actuators converge before the target moves again. The advance and grasp phases
# get more: sliding the cube into the jaws and then closing a friction grasp on
# it both need longer to settle than a free-space move.
ORACLE_SETTLE_STEPS = 16
ORACLE_GRASP_SETTLE_STEPS = 34

# --- dataset ---------------------------------------------------------------
HF_USER = os.environ.get("HF_USER", "jazarium")
DATASET_REPO_ID = f"{HF_USER}/so101_pick_cube"

# 75, not 50. Spec §5: 50 episodes over a 30 cm workspace is a documented
# failure; 75 over ~10 cm gives 60-80%. Do not lower this to "save time" —
# it is the difference between a demo and a robot that grasps at air.
N_EPISODES = 75

# --- training --------------------------------------------------------------
# Shared seed for training and eval so runs reproduce.
SEED = 1000

# Where the fine-tuned checkpoint is published (private).
POLICY_REPO_ID = f"{HF_USER}/smolvla_so101_pick"

# Two run profiles. THE FULL RUN is the real one (20k steps ~= 4 h on an A100,
# 60-80% success in the reference); it is a separate, deliberate spend.
TRAIN_STEPS = 20_000
TRAIN_BATCH_SIZE = 64
TRAIN_JOB_TARGET = "a10g-small"

# THE PIPE-TEST RUN (user directive 2026-07-14: "test the pipe, don't spend
# much"). Deliberately under-trained — its job is to prove the whole remote loop
# (push -> HF Jobs submit -> log stream -> checkpoint to Hub -> pull -> eval),
# NOT to clear the >=60% bar. Expect a low success rate; that is success here.
# a10g-small (24 GB) not t4-small: batch 32 of a 450M VLA is tight on a T4's
# 16 GB, and a job that OOMs still bills. Reliability beats absolute-cheapest.
PIPE_TEST_STEPS = 2_000
PIPE_TEST_BATCH_SIZE = 32
PIPE_TEST_JOB_TARGET = "a10g-small"

# Local loop-start smoke: a handful of steps on CPU to prove the training loop
# assembles (config/shape/dtype) before paying for a GPU. NOT training — on MPS
# a fine-tune is ~2 h per 20 steps (Task 3 M0).
TRAIN_SMOKE_STEPS = 5
TRAIN_SMOKE_BATCH_SIZE = 2

TRAIN_OUTPUT_DIR = APP_ROOT / "outputs" / "train" / "smolvla_so101"
TRAIN_SMOKE_OUTPUT_DIR = APP_ROOT / "outputs" / "train" / "smolvla_so101_smoke"


# SmolVLA base declares THREE image inputs named observation.images.camera1/2/3
# (Task 3 M0 found this). Our dataset has two semantic cameras, so at both train
# and eval we (a) rename ours onto camera1/camera2 and (b) add one masked
# placeholder for the missing camera3 via --policy.empty_cameras=1. The
# placeholder is filled with a masked dummy tensor — no fake image is recorded.
# (LeRobot rename_map.mdx; supported for SmolVLA/PI0/PI05/PI0Fast/XVLA.)
import json as _json

CAMERA_RENAME_MAP = {
    "observation.images.wrist": "observation.images.camera1",
    "observation.images.scene": "observation.images.camera2",
}
EMPTY_CAMERAS = 1


def _rename_map_arg() -> str:
    return "--rename_map=" + _json.dumps(CAMERA_RENAME_MAP)


def _train_base_args(steps: int, batch_size: int, output_dir) -> list[str]:
    return [
        "lerobot-train",
        f"--policy.path={BASE_POLICY_ID}",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        _rename_map_arg(),
        f"--policy.empty_cameras={EMPTY_CAMERAS}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--seed={SEED}",
        f"--output_dir={output_dir}",
    ]


def train_smoke_cmd() -> list[str]:
    """A few steps on CPU — proves the loop assembles. NOT training, NOT remote."""
    return _train_base_args(
        TRAIN_SMOKE_STEPS, TRAIN_SMOKE_BATCH_SIZE, TRAIN_SMOKE_OUTPUT_DIR
    ) + [
        "--policy.device=cpu",
        "--policy.push_to_hub=false",
        f"--save_freq={TRAIN_SMOKE_STEPS}",
    ]


def train_remote_cmd(pipe_test: bool = True) -> list[str]:
    """The real fine-tune, on HF Jobs. Never runs on this machine.

    pipe_test=True  -> the cheap under-trained run that proves the pipeline.
    pipe_test=False -> the full 20k-step run (a separate, deliberate spend).
    """
    if pipe_test:
        steps, batch, target = (
            PIPE_TEST_STEPS, PIPE_TEST_BATCH_SIZE, PIPE_TEST_JOB_TARGET
        )
    else:
        steps, batch, target = TRAIN_STEPS, TRAIN_BATCH_SIZE, TRAIN_JOB_TARGET
    return _train_base_args(steps, batch, TRAIN_OUTPUT_DIR) + [
        f"--policy.repo_id={POLICY_REPO_ID}",
        "--policy.push_to_hub=true",
        f"--job.target={target}",  # any non-local value submits to HF Jobs
    ]
