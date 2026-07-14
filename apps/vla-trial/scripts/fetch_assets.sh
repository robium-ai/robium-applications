#!/usr/bin/env bash
# Vendor the mujoco_menagerie SO-101 asset.
#
# Menagerie's robotstudio_so101 — NOT TheRobotStudio/SO-ARM100 upstream, whose
# own README says the gripper linear-joint mapping "is not yet reflected in the
# current URDF and MuJoCo files". Menagerie's derivation adds primitive collision
# geoms, manipulation-tuned solver params for the gripper, and a camera mount.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/src/vla_trial/env/assets"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git "$TMP/menagerie"
git -C "$TMP/menagerie" sparse-checkout set robotstudio_so101

mkdir -p "$DEST"
cp -R "$TMP/menagerie/robotstudio_so101/." "$DEST/"

echo "SO-101 asset vendored to $DEST"
ls "$DEST"
