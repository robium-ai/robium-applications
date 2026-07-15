"""THE PIPE-TEST PASS BAR (Task 9, user directive 2026-07-14).

The real fine-tune (20k steps, HF Jobs) hasn't run yet — it's blocked on HF
Jobs credits. The only checkpoint that exists right now is the Task 8
LOCAL SMOKE checkpoint: `TRAIN_SMOKE_STEPS` (5) steps on CPU, essentially
still the base model. It is EXPECTED to score ~0% — SUCCESS_RATE_FLOOR
(0.60, the real pass bar for the eventual 20k-step Hub checkpoint) does NOT
apply here and is deliberately NOT asserted below; asserting it against this
checkpoint would fail for a reason that has nothing to do with the eval
pipeline being broken.

What this test guards instead is PIPELINE MECHANICS: the checkpoint loads,
`evaluate()` rolls out N seeded episodes end to end (camera rename, action
un-normalization, sim step loop), and `eval_info.json` lands on disk with a
numeric success_rate. That is the whole deliverable for this task — proving
record -> train -> eval closes before the real fine-tune exists to grade.

Marked `slow`: it loads the 450M SmolVLA checkpoint. Run via `make smoke`.
"""

import json

import pytest

from vla_trial.config import (
    SEED,
    SMOKE_CHECKPOINT_PATH,
    SMOKE_EVAL_EPISODES,
    SMOKE_EVAL_OUTPUT_DIR,
)
from vla_trial.policy.evaluate import evaluate

pytestmark = pytest.mark.slow


def test_eval_pipeline_runs_end_to_end_on_the_local_smoke_checkpoint():
    result = evaluate(
        policy_path=str(SMOKE_CHECKPOINT_PATH),
        n_episodes=SMOKE_EVAL_EPISODES,
        seed=SEED,
    )

    assert result["n_episodes"] == SMOKE_EVAL_EPISODES
    assert result["successes"] == sum(ep["success"] for ep in result["episodes"])
    # Mechanics only — NOT the >=0.60 pass bar. See module docstring.
    assert 0.0 <= result["success_rate"] <= 1.0

    written = json.loads((SMOKE_EVAL_OUTPUT_DIR / "eval_info.json").read_text())
    assert written["success_rate"] == result["success_rate"]
    assert written["n_episodes"] == SMOKE_EVAL_EPISODES
