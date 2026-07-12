"""Config-driven runner for the manual pipeline stages.

Usage: python -m manip_trial.run <train-baseline|baseline-eval|train-smoke|eval-trained>
Builds each stage's CLI invocation from config.py so manual runs and the
smoke test always agree on parameters.
"""

import shutil
import subprocess
import sys

from manip_trial import config


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
    else:
        print(__doc__)
        return 2
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
