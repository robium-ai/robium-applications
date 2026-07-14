"""The SO-101 asset must be the menagerie one, and must load under our MuJoCo."""
import mujoco
import pytest

from vla_trial.config import SCENE_XML, SCENE_CAM, WRIST_CAM


def test_scene_xml_exists():
    assert SCENE_XML.is_file(), f"missing MJCF at {SCENE_XML} — run `make assets`"


def test_scene_loads_and_has_expected_actuators():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    # SO-101: 5 arm joints + 1 gripper.
    assert model.nu == 6, f"expected 6 actuators, got {model.nu}"


def test_scene_has_both_cameras():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        for i in range(model.ncam)
    }
    assert {WRIST_CAM, SCENE_CAM} <= names, f"cameras found: {names}"
