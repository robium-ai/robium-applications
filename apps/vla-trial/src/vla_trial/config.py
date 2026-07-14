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

# M0 gate: SmolVLA action chunking means one forward pass covers ~50 actions
# (~1.7 s of robot motion at 30 FPS). So the bar is ~1 pass/sec, not 30 Hz.
POLICY_LATENCY_CEILING_S = 1.0
