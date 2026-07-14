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
# The cube's z is its half-height (0.03), so it rests on the floor.
CUBE_SPAWN_CENTER = (0.32, -0.06, 0.03)
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
# Names are so101.xml's `collision_gripper`-class geoms (fixed + moving jaw).
GRIPPER_GEOMS = (
    "fixed_jaw_box1",
    "fixed_jaw_box2",
    "fixed_jaw_box3",
    "fixed_jaw_box4",
    "fixed_jaw_box5",
    "fixed_jaw_box6",
    "fixed_jaw_box7",
    "fixed_jaw_sph_tip1",
    "fixed_jaw_sph_tip2",
    "fixed_jaw_sph_tip3",
    "moving_jaw_box1",
    "moving_jaw_box2",
    "moving_jaw_box3",
    "moving_jaw_sph_tip1",
    "moving_jaw_sph_tip2",
    "moving_jaw_sph_tip3",
)
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
