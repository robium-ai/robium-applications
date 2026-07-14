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
