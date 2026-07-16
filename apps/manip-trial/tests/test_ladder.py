"""Ladder plumbing tests — pure config/fs logic, no training, fast."""

import json
from pathlib import Path

from manip_trial import config
from manip_trial.ladder import build_manifest
from manip_trial.run import _prune_ladder


def test_ladder_rungs_shape():
    rungs = config.ladder_rungs()
    assert [r["name"] for r in rungs] == ["1k", "3k", "5k", "10k"]
    assert [r["steps"] for r in rungs] == [1000, 3000, 5000, 10000]
    # 1k/3k/5k come from the ladder run; 10k is the reused baseline.
    for r in rungs[:3]:
        assert str(config.LADDER_TRAIN_OUTPUT_DIR) in str(r["checkpoint"])
    assert str(config.BASELINE_TRAIN_OUTPUT_DIR) in str(rungs[3]["checkpoint"])
    for r in rungs:
        assert str(r["checkpoint"]).endswith("pretrained_model")


def test_train_ladder_cmd():
    cmd = config.train_ladder_cmd()
    assert "--steps=5000" in cmd
    assert "--save_freq=1000" in cmd
    assert f"--output_dir={config.LADDER_TRAIN_OUTPUT_DIR}" in cmd


def test_prune_ladder_keeps_only_rungs(tmp_path, monkeypatch):
    ckpt_root = tmp_path / "checkpoints"
    for name in ["001000", "002000", "003000", "004000", "005000"]:
        (ckpt_root / name / "pretrained_model").mkdir(parents=True)
        (ckpt_root / name / "training_state").mkdir()
    (ckpt_root / "last").symlink_to(ckpt_root / "005000")
    monkeypatch.setattr(config, "LADDER_TRAIN_OUTPUT_DIR", tmp_path)

    _prune_ladder()

    kept = sorted(p.name for p in ckpt_root.iterdir())
    assert kept == ["001000", "003000", "005000"]
    for name in kept:
        assert (ckpt_root / name / "pretrained_model").is_dir()
        # training_state (optimizer etc.) is dead weight for inference rungs
        assert not (ckpt_root / name / "training_state").exists()


def test_build_manifest(tmp_path):
    app_root = tmp_path
    eval_root = tmp_path / "outputs" / "eval" / "ladder"
    ckpt = tmp_path / "outputs" / "train" / "x" / "checkpoints" / "001000" / "pretrained_model"
    ckpt.mkdir(parents=True)
    rung_eval = eval_root / "1k"
    (rung_eval / "videos" / "pusht_0").mkdir(parents=True)
    (rung_eval / "videos" / "pusht_0" / "eval_episode_0.mp4").write_bytes(b"")
    (rung_eval / "eval_info.json").write_text(
        json.dumps({"overall": {
            "avg_max_reward": 0.1, "avg_sum_reward": 2.0, "pc_success": 0.0,
            "n_episodes": 10, "eval_s": 9.9, "video_paths": ["ignored"],
        }})
    )
    rungs = [{"name": "1k", "steps": 1000, "run": "ladder", "checkpoint": ckpt}]

    m = build_manifest(rungs, eval_root, app_root)

    r = m["rungs"][0]
    assert r["checkpoint"] == "outputs/train/x/checkpoints/001000/pretrained_model"
    assert r["metrics"] == {
        "avg_max_reward": 0.1, "avg_sum_reward": 2.0, "pc_success": 0.0, "n_episodes": 10,
    }
    assert r["videos"] == ["outputs/eval/ladder/1k/videos/pusht_0/eval_episode_0.mp4"]
