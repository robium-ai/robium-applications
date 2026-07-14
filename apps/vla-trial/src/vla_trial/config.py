"""Single source of truth for vla-trial run parameters.

The Makefile targets and the tests both build their invocations from these
values, so a hand-run stage and the pass-bar run can never drift apart.
(Pattern lifted from apps/manip-trial.)
"""

from pathlib import Path

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
