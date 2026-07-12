"""manip-trial pass-bar smoke test.

One command: `uv run pytest tests/test_smoke.py`

Proves the trial pass bar end to end at tiny scale:
  1. a training run completes (200-step ACT on lerobot/pusht),
  2. eval of the resulting checkpoint produces metrics (2 episodes).

No success-rate threshold — a 200-step policy is not expected to solve
PushT; the bar is pipeline-completes + numeric metrics produced.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from manip_trial import config  # noqa: E402

TRAIN_TIMEOUT_S = int(os.environ.get("SMOKE_TRAIN_TIMEOUT", "1800"))
EVAL_TIMEOUT_S = int(os.environ.get("SMOKE_EVAL_TIMEOUT", "900"))


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, timeout=timeout, text=True)


@pytest.fixture(scope="module")
def trained_checkpoint():
    shutil.rmtree(config.TRAIN_OUTPUT_DIR, ignore_errors=True)
    proc = _run(config.train_smoke_cmd(), TRAIN_TIMEOUT_S)
    assert proc.returncode == 0, "lerobot-train exited nonzero"
    ckpt = config.latest_checkpoint()
    assert (ckpt / "config.json").is_file(), f"missing config.json in {ckpt}"
    assert (ckpt / "model.safetensors").is_file(), f"missing weights in {ckpt}"
    return ckpt


def test_train_completes(trained_checkpoint):
    assert trained_checkpoint.is_dir()


def test_eval_produces_metrics(trained_checkpoint):
    shutil.rmtree(config.SMOKE_EVAL_OUTPUT_DIR, ignore_errors=True)
    proc = _run(
        config.eval_cmd(
            str(trained_checkpoint),
            config.SMOKE_EVAL_EPISODES,
            config.SMOKE_EVAL_BATCH_SIZE,
            config.SMOKE_EVAL_OUTPUT_DIR,
        ),
        EVAL_TIMEOUT_S,
    )
    assert proc.returncode == 0, "lerobot-eval exited nonzero"

    eval_info = config.SMOKE_EVAL_OUTPUT_DIR / "eval_info.json"
    assert eval_info.is_file(), f"missing {eval_info}"
    # lerobot 0.6.0 eval_info.json schema: {"per_task": [...], "per_group": {...}, "overall": {...}}
    metrics = json.loads(eval_info.read_text())["overall"]
    for key in ("pc_success", "avg_sum_reward"):
        assert isinstance(metrics.get(key), (int, float)), f"{key} not numeric: {metrics}"
    print(f"\nSMOKE METRICS: pc_success={metrics['pc_success']} avg_sum_reward={metrics['avg_sum_reward']}")
