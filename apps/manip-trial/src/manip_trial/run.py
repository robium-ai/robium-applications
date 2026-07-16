"""Config-driven runner for the manual pipeline stages.

Usage: python -m manip_trial.run <train-baseline|baseline-eval|train-smoke|eval-trained|train-ladder|eval-ladder>
Builds each stage's CLI invocation from config.py so manual runs and the
smoke test always agree on parameters.
"""

import shutil
import subprocess
import sys

from manip_trial import config


def _prune_ladder() -> None:
    """Keep only the rung checkpoints (and drop their optimizer state).

    A 5k run with save_freq=1000 leaves 5 checkpoints + a `last` symlink at
    ~400 MB each (half of it training_state). The demo needs 3 rungs of
    pretrained_model only.
    """
    ckpt_root = config.LADDER_TRAIN_OUTPUT_DIR / "checkpoints"
    keep = {f"{s:06d}" for s in config.LADDER_KEEP_STEPS}
    for d in sorted(ckpt_root.iterdir()):
        if d.name not in keep:
            d.unlink() if d.is_symlink() else shutil.rmtree(d)
        else:
            shutil.rmtree(d / "training_state", ignore_errors=True)


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "train-baseline":
        shutil.rmtree(config.BASELINE_TRAIN_OUTPUT_DIR, ignore_errors=True)
        cmd = config.train_baseline_cmd()
    elif stage == "baseline-eval":
        shutil.rmtree(config.BASELINE_EVAL_OUTPUT_DIR, ignore_errors=True)
        cmd = config.eval_cmd(
            str(config.latest_checkpoint(config.BASELINE_TRAIN_OUTPUT_DIR)),
            config.BASELINE_EVAL_EPISODES,
            config.BASELINE_EVAL_BATCH_SIZE,
            config.BASELINE_EVAL_OUTPUT_DIR,
        )
    elif stage == "train-smoke":
        shutil.rmtree(config.TRAIN_OUTPUT_DIR, ignore_errors=True)
        cmd = config.train_smoke_cmd()
    elif stage == "eval-trained":
        shutil.rmtree(config.SMOKE_EVAL_OUTPUT_DIR, ignore_errors=True)
        cmd = config.eval_cmd(
            str(config.latest_checkpoint()),
            config.SMOKE_EVAL_EPISODES,
            config.SMOKE_EVAL_BATCH_SIZE,
            config.SMOKE_EVAL_OUTPUT_DIR,
        )
    elif stage == "train-ladder":
        shutil.rmtree(config.LADDER_TRAIN_OUTPUT_DIR, ignore_errors=True)
        cmd = config.train_ladder_cmd()
        print(f"$ {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            _prune_ladder()
        return rc
    elif stage == "eval-ladder":
        from manip_trial.ladder import eval_ladder

        return eval_ladder()
    else:
        print(__doc__)
        return 2
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
